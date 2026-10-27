from __future__ import annotations

import json
from collections.abc import Callable

import requests

from odoo.addons.muk_ai.providers.base import ProviderBase

DEFAULT_EFFORTS = ['low', 'medium', 'high']


class ChatCompletionsProvider(ProviderBase):
    """OpenAI-compatible ``/chat/completions`` adapter shared by OpenRouter and Ollama.

    muk_ai speaks the Responses item vocabulary internally; this adapter maps
    those items to chat ``messages`` on the way out and folds the streamed
    chunks back into the muk_ai result on the way in.
    """

    modalities = frozenset({'chat'})

    reasoning_error_tokens = ('reasoning',)

    # ----------------------------------------------------------
    # Contract
    # ----------------------------------------------------------

    def headers(self) -> dict:
        """Return the JSON headers, with a bearer token when one is configured."""
        headers = {'Content-Type': 'application/json'}
        if key := self.api_key:
            headers['Authorization'] = f'Bearer {key}'
        return headers

    def request(
        self,
        inputs,
        tools_schema=None,
        text_schema=None,
        on_delta=None,
        model=None,
        enable_web_search=False,
        enable_code_interpreter=False,
        extra=None,
    ) -> dict:
        """Build a chat completions request and run it, streamed when asked."""
        model = self.model_for(model)
        body = {
            'model': model,
            'messages': self._inputs_to_messages(inputs),
        }
        if self.max_tokens:
            body['max_tokens'] = self.max_tokens
        if text_schema:
            body['response_format'] = {
                'type': 'json_schema',
                'json_schema': {
                    'name': text_schema.get('name', 'response'),
                    'schema': text_schema['schema'],
                    'strict': True,
                },
            }
        if tools := self._tools_to_functions(tools_schema):
            body['tools'] = tools
        if effort := (extra or {}).get('reasoning_effort'):
            self._apply_reasoning(body, effort)
        return self._invoke_with_reasoning_retry(
            model,
            lambda callback: self._invoke(body, callback),
            on_delta,
            ((body, ('reasoning', 'reasoning_effort')),),
        )

    def list_models(self) -> list[dict]:
        """Return the vendor's chat models as normalized catalogue entries."""
        raise NotImplementedError

    # ----------------------------------------------------------
    # Hooks
    # ----------------------------------------------------------

    def _apply_reasoning(self, body: dict, effort: str) -> None:
        """Set the reasoning tier in the wire form the vendor accepts."""
        body['reasoning_effort'] = effort

    def _file_part(self, block: dict) -> dict:
        """Return the message part for a non-image file attachment.

        The plain chat wire has no file part, so the file is named instead;
        a vendor that accepts files overrides this.
        """
        filename = block.get('filename') or 'attachment'
        return {'type': 'text', 'text': f'[Attached file not readable here: {filename}]'}

    # ----------------------------------------------------------
    # Tools
    # ----------------------------------------------------------

    @staticmethod
    def _tools_to_functions(tools_schema: list | None) -> list:
        """Map muk_ai function schemas to chat ``tools`` entries, dropping duplicates."""
        seen = set()
        out = []
        for tool in tools_schema or []:
            name = tool.get('name')
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(
                {
                    'type': 'function',
                    'function': {
                        'name': name,
                        'description': tool.get('description') or '',
                        'parameters': tool.get('parameters')
                        or {'type': 'object', 'properties': {}},
                    },
                }
            )
        return out

    # ----------------------------------------------------------
    # Inputs
    # ----------------------------------------------------------

    def _inputs_to_messages(self, inputs: list | None) -> list:
        """Map muk_ai input items to chat messages.

        System items are joined into one leading message, since several
        vendors only honour a system prompt in first position. A function
        call joins the assistant message right before it, so every ``tool``
        message answers a call of the assistant turn it follows.
        """
        system_parts = []
        messages = []
        for item in self._wire_items(inputs):
            if not isinstance(item, dict):
                continue
            item_type = item.get('type')
            role = item.get('role')
            if role == 'system':
                if text := self._text_from_content(item.get('content')):
                    system_parts.append(text)
            elif item_type == 'function_call':
                arguments = item.get('arguments') or '{}'
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, default=str)
                call = {
                    'id': item.get('call_id') or '',
                    'type': 'function',
                    'function': {'name': item.get('name') or '', 'arguments': arguments},
                }
                last = messages[-1] if messages else None
                if last and last['role'] == 'assistant':
                    last.setdefault('tool_calls', []).append(call)
                else:
                    messages.append(
                        {'role': 'assistant', 'content': None, 'tool_calls': [call]}
                    )
            elif item_type == 'function_call_output':
                output = item.get('output')
                if not isinstance(output, str):
                    output = json.dumps(output, default=str)
                messages.append(
                    {
                        'role': 'tool',
                        'tool_call_id': item.get('call_id') or '',
                        'content': output,
                    }
                )
            elif role == 'user':
                messages.append(
                    {'role': 'user', 'content': self._user_content(item.get('content'))}
                )
            elif role == 'assistant':
                if text := self._text_from_content(item.get('content')):
                    messages.append({'role': 'assistant', 'content': text})
        if system_parts:
            messages.insert(0, {'role': 'system', 'content': '\n\n'.join(system_parts)})
        return messages

    @staticmethod
    def _text_from_content(content) -> str:
        """Flatten a string or list-of-blocks content into plain text."""
        if isinstance(content, str):
            return content
        return '\n\n'.join(
            block['text']
            for block in content or []
            if isinstance(block, dict) and block.get('text')
        )

    def _user_content(self, content):
        """Map user content to a string, or to parts when it carries an image or file."""
        if isinstance(content, str):
            return content
        text_parts = []
        media = []
        for block in content or []:
            if not isinstance(block, dict):
                continue
            block_type = block.get('type') or ''
            if block_type == 'muk_ai_attachment':
                part = self._attachment_part(block)
                if part is None:
                    continue
                if part['type'] == 'text':
                    text_parts.append(part['text'])
                else:
                    media.append(part)
            elif block_type == 'input_image' and block.get('image_url'):
                media.append({'type': 'image_url', 'image_url': {'url': block['image_url']}})
            elif not block_type.startswith('muk_ai_') and block.get('text'):
                text_parts.append(block['text'])
        if not media:
            return '\n\n'.join(text_parts)
        parts = [{'type': 'text', 'text': '\n\n'.join(text_parts)}] if text_parts else []
        return parts + media

    def _attachment_part(self, block: dict) -> dict | None:
        """Convert an attachment block into an image, file or text part."""
        strategy = block.get('strategy')
        mimetype = block.get('mimetype') or 'application/octet-stream'
        data_b64 = block.get('data_b64') or ''
        if strategy == 'image' and data_b64:
            return {
                'type': 'image_url',
                'image_url': {'url': f'data:{mimetype};base64,{data_b64}'},
            }
        if strategy == 'file' and data_b64:
            return self._file_part(block)
        text = block.get('inline_text') or ''
        if not text:
            return None
        filename = block.get('filename') or 'attachment'
        if key := block.get('attachment_id'):
            mimetype = f'{mimetype}, attachment {key}'
        if block.get('truncated'):
            text += '\n[truncated]'
        return {'type': 'text', 'text': f'--- File: {filename} ({mimetype}) ---\n{text}'}

    # ----------------------------------------------------------
    # Invoke
    # ----------------------------------------------------------

    def _invoke(self, body: dict, on_delta: Callable | None) -> dict:
        """Dispatch the request to the streaming or the buffered path."""
        if callable(on_delta):
            return self._stream(body, on_delta)
        return self._parse_response(self._post_json('/chat/completions', body))

    def _get_json(self, url: str) -> dict:
        """GET a JSON document from the vendor, with the token when one is set.

        Catalogue routes are often public, so a missing key is not an error here.

        :raise UserError: on HTTP or transport errors
        """
        headers = {'Content-Type': 'application/json'}
        if key := self._api_key:
            headers['Authorization'] = f'Bearer {key}'
        try:
            response = self._http_session().get(
                url,
                headers=headers,
                timeout=self.request_timeout,
            )
            response.raise_for_status()
        except requests.HTTPError as error:
            self._raise(getattr(error.response, 'text', '') or str(error))
        except requests.RequestException as error:
            self._raise(error)
        return response.json()

    # ----------------------------------------------------------
    # Parse
    # ----------------------------------------------------------

    @staticmethod
    def _reasoning_text(message: dict) -> str:
        """Return the reasoning text a vendor put next to the answer, if any."""
        return message.get('reasoning') or message.get('reasoning_content') or ''

    def _usage_from(self, usage: dict | None) -> dict:
        """Normalize an OpenAI-style usage block."""
        usage = usage or {}
        details = usage.get('prompt_tokens_details') or {}
        return self._usage(
            input_tokens=usage.get('prompt_tokens'),
            output_tokens=usage.get('completion_tokens'),
            cache_read_tokens=details.get('cached_tokens'),
            cache_write_tokens=details.get('cache_write_tokens'),
        )

    def _assemble(
        self,
        text: str,
        calls: list[dict],
        usage: dict,
        finish_reason: str | None,
        on_delta: Callable | None = None,
    ) -> dict:
        """Build the muk_ai result and its carry inputs from the collected parts."""
        tool_calls = []
        carry_inputs = []
        if text.strip():
            carry_inputs.append(
                {'role': 'assistant', 'content': [{'type': 'output_text', 'text': text}]}
            )
        for call in calls:
            args, parse_error = self._parse_tool_arguments(call['arguments'])
            tool_calls.append(
                {
                    'call_id': call['call_id'],
                    'name': call['name'],
                    'arguments': args,
                    '_parse_error': parse_error,
                }
            )
            carry_inputs.append(
                {
                    'type': 'function_call',
                    'call_id': call['call_id'],
                    'name': call['name'],
                    'arguments': call['arguments'] or '{}',
                }
            )
        result = {
            'text': text.strip(),
            'tool_calls': tool_calls,
            'carry_inputs': carry_inputs,
            'usage': self._usage_from(usage),
        }
        if finish_reason == 'length':
            self._apply_truncation(result, on_delta, self.max_tokens)
        return result

    def _parse_response(self, payload: dict) -> dict:
        """Parse a buffered chat completion into the muk_ai result."""
        if error := payload.get('error'):
            self._raise(error.get('message') if isinstance(error, dict) else error)
        choice = next(iter(payload.get('choices') or []), None) or {}
        message = choice.get('message') or {}
        calls = [
            {
                'call_id': call.get('id') or f'call_{index}',
                'name': (call.get('function') or {}).get('name') or '',
                'arguments': (call.get('function') or {}).get('arguments') or '',
            }
            for index, call in enumerate(message.get('tool_calls') or [])
        ]
        return self._assemble(
            message.get('content') or '',
            calls,
            payload.get('usage'),
            choice.get('finish_reason'),
        )

    # ----------------------------------------------------------
    # Streaming
    # ----------------------------------------------------------

    def _stream(self, body: dict, on_delta: Callable) -> dict:
        """Stream the completion, emitting deltas and assembling the final result.

        Tool calls arrive in fragments keyed by ``index``: the first carries
        the id and name, the rest only append to the arguments.
        """
        body = {**body, 'stream': True, 'stream_options': {'include_usage': True}}
        text_parts = []
        calls = {}
        usage = None
        finish_reason = None
        for chunk in self._post_stream('/chat/completions', body):
            if error := chunk.get('error'):
                self._raise(error.get('message') if isinstance(error, dict) else error)
            if chunk.get('usage'):
                usage = chunk['usage']
            for choice in chunk.get('choices') or []:
                delta = choice.get('delta') or {}
                if reasoning := self._reasoning_text(delta):
                    self._call_on_delta(on_delta, 'reasoning', {'delta': reasoning})
                if content := delta.get('content'):
                    text_parts.append(content)
                    self._call_on_delta(on_delta, 'text', {'delta': content})
                for fragment in delta.get('tool_calls') or []:
                    self._fold_tool_call(fragment, calls, on_delta)
                finish_reason = choice.get('finish_reason') or finish_reason
        self._raise_unless_complete(finish_reason is not None)
        return self._assemble(
            ''.join(text_parts),
            [calls[index] for index in sorted(calls)],
            usage,
            finish_reason,
            on_delta,
        )

    def _fold_tool_call(self, fragment: dict, calls: dict, on_delta: Callable) -> None:
        """Merge one streamed tool-call fragment into its pending call."""
        index = fragment.get('index', len(calls))
        function = fragment.get('function') or {}
        call = calls.get(index)
        if call is None:
            call = calls[index] = {
                'call_id': fragment.get('id') or f'call_{index}',
                'name': function.get('name') or '',
                'arguments': '',
            }
            self._call_on_delta(
                on_delta,
                'tool_start',
                {'call_id': call['call_id'], 'name': call['name']},
            )
        elif function.get('name') and not call['name']:
            call['name'] = function['name']
        if arguments := function.get('arguments'):
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, default=str)
            call['arguments'] += arguments
            self._call_on_delta(
                on_delta,
                'tool_args',
                {'call_id': call['call_id'], 'delta': arguments},
            )
