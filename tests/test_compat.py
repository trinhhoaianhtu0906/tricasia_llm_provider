import json
from unittest.mock import patch

from odoo.tests.common import TransactionCase, tagged

from odoo.addons.muk_ai.providers import REGISTRY
from odoo.addons.tricasia_llm_provider.providers.ollama import OllamaProvider
from odoo.addons.tricasia_llm_provider.providers.openrouter import OpenRouterProvider


@tagged('post_install', '-at_install')
class TestChatCompletions(TransactionCase):
    """The chat completions wire: request mapping, streaming and parsing."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.openrouter = cls.env.ref('tricasia_llm_provider.provider_openrouter')
        cls.openrouter.sudo().api_key = 'test-key'
        cls.ollama = cls.env.ref('tricasia_llm_provider.provider_ollama')

    def _run_stream(self, provider, chunks, **kwargs):
        """Stream ``chunks`` through the provider and return (result, body, deltas)."""
        sent = {}
        deltas = []

        def fake_stream(_self, path, body):
            sent.update(path=path, body=body)
            return iter(chunks)

        impl = REGISTRY[provider.name]
        with patch.object(impl, '_post_stream', fake_stream):
            result = provider._get_client().request(
                on_delta=lambda kind, data: deltas.append((kind, data)),
                **kwargs,
            )
        return result, sent, deltas

    # ----------------------------------------------------------
    # Registry
    # ----------------------------------------------------------

    def test_providers_registered(self):
        self.assertIs(REGISTRY['openrouter'], OpenRouterProvider)
        self.assertIs(REGISTRY['ollama'], OllamaProvider)
        self.assertEqual(self.openrouter.api_endpoint, 'https://openrouter.ai/api/v1')

    def test_ollama_serves_without_key(self):
        self.assertTrue(self.ollama._can_serve())
        client = self.ollama._get_client()
        self.assertNotIn('Authorization', client.headers())
        self.assertEqual(client.native_url, 'http://host.docker.internal:11434')

    def test_openrouter_needs_key(self):
        self.openrouter.sudo().api_key = False
        self.assertFalse(self.openrouter._can_serve())

    # ----------------------------------------------------------
    # Inputs
    # ----------------------------------------------------------

    def test_inputs_to_messages(self):
        client = self.openrouter._get_client()
        messages = client._inputs_to_messages(
            [
                {'role': 'system', 'content': [{'type': 'input_text', 'text': 'Be brief.'}]},
                {'role': 'user', 'content': [{'type': 'input_text', 'text': 'Count partners'}]},
                {'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'Checking.'}]},
                {'type': 'function_call', 'call_id': 'c1', 'name': 'search_count', 'arguments': '{"model": "res.partner"}'},
                {'type': 'function_call', 'call_id': 'c2', 'name': 'search_count', 'arguments': {'model': 'res.users'}},
                {'type': 'function_call_output', 'call_id': 'c1', 'output': '42'},
                {'type': 'function_call_output', 'call_id': 'c2', 'output': {'count': 3}},
                {'type': 'reasoning', 'id': 'r1', 'summary': []},
                {'role': 'system', 'content': 'Answer in Vietnamese.', '_meta': 1},
            ]
        )
        self.assertEqual(
            messages,
            [
                {'role': 'system', 'content': 'Be brief.\n\nAnswer in Vietnamese.'},
                {'role': 'user', 'content': 'Count partners'},
                {
                    'role': 'assistant',
                    'content': 'Checking.',
                    'tool_calls': [
                        {'id': 'c1', 'type': 'function', 'function': {'name': 'search_count', 'arguments': '{"model": "res.partner"}'}},
                        {'id': 'c2', 'type': 'function', 'function': {'name': 'search_count', 'arguments': '{"model": "res.users"}'}},
                    ],
                },
                {'role': 'tool', 'tool_call_id': 'c1', 'content': '42'},
                {'role': 'tool', 'tool_call_id': 'c2', 'content': '{"count": 3}'},
            ],
        )

    def test_function_call_without_text_opens_assistant_turn(self):
        messages = self.openrouter._get_client()._inputs_to_messages(
            [
                {'role': 'user', 'content': 'hi'},
                {'type': 'function_call', 'call_id': 'c1', 'name': 'ping', 'arguments': '{}'},
            ]
        )
        self.assertEqual(messages[1]['role'], 'assistant')
        self.assertIsNone(messages[1]['content'])
        self.assertEqual(messages[1]['tool_calls'][0]['id'], 'c1')

    def test_attachments(self):
        block_image = {'type': 'muk_ai_attachment', 'strategy': 'image', 'mimetype': 'image/png', 'data_b64': 'AAA'}
        block_pdf = {'type': 'muk_ai_attachment', 'strategy': 'file', 'mimetype': 'application/pdf', 'filename': 'a.pdf', 'data_b64': 'BBB'}
        block_text = {'type': 'muk_ai_attachment', 'strategy': 'text', 'filename': 'n.txt', 'mimetype': 'text/plain', 'inline_text': 'note'}
        content = [{'type': 'input_text', 'text': 'Look'}, block_image, block_pdf, block_text, {'type': 'muk_ai_marker'}]
        parts = self.openrouter._get_client()._user_content(content)
        self.assertEqual(parts[0], {'type': 'text', 'text': 'Look\n\n--- File: n.txt (text/plain) ---\nnote'})
        self.assertEqual(parts[1], {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAA'}})
        self.assertEqual(parts[2]['type'], 'file')
        self.assertEqual(parts[2]['file']['file_data'], 'data:application/pdf;base64,BBB')
        ollama_parts = self.ollama._get_client()._user_content(content)
        self.assertEqual(len(ollama_parts), 2)
        self.assertIn('a.pdf', ollama_parts[0]['text'])

    # ----------------------------------------------------------
    # Request body
    # ----------------------------------------------------------

    def test_request_body(self):
        chunks = [{'choices': [{'delta': {'content': 'ok'}, 'finish_reason': 'stop'}]}]
        _result, sent, _deltas = self._run_stream(
            self.openrouter,
            chunks,
            inputs=[{'role': 'user', 'content': 'hi'}],
            tools_schema=[
                {'type': 'function', 'name': 'ping', 'description': 'Ping', 'parameters': {'type': 'object', 'properties': {}}, 'strict': False},
                {'type': 'function', 'name': 'ping'},
            ],
            model='qwen/qwen3.8-flash',
            enable_web_search=True,
            extra={'reasoning_effort': 'high'},
        )
        body = sent['body']
        self.assertEqual(sent['path'], '/chat/completions')
        self.assertEqual(body['model'], 'qwen/qwen3.8-flash')
        self.assertTrue(body['stream'])
        self.assertEqual(body['stream_options'], {'include_usage': True})
        self.assertEqual(body['reasoning'], {'effort': 'high'})
        self.assertEqual(body['max_tokens'], 4096)
        self.assertEqual(
            body['tools'],
            [{'type': 'function', 'function': {'name': 'ping', 'description': 'Ping', 'parameters': {'type': 'object', 'properties': {}}}}],
        )

    def test_ollama_reasoning_wire(self):
        chunks = [{'choices': [{'delta': {'content': 'ok'}, 'finish_reason': 'stop'}]}]
        _result, sent, _deltas = self._run_stream(
            self.ollama,
            chunks,
            inputs=[{'role': 'user', 'content': 'hi'}],
            extra={'reasoning_effort': 'low'},
        )
        self.assertEqual(sent['body']['reasoning_effort'], 'low')
        self.assertEqual(sent['body']['model'], 'llama3.1')

    # ----------------------------------------------------------
    # Streaming
    # ----------------------------------------------------------

    def test_stream_text_reasoning_and_tool_calls(self):
        chunks = [
            {'choices': [{'delta': {'reasoning': 'Need data. '}}]},
            {'choices': [{'delta': {'content': 'Let me '}}]},
            {'choices': [{'delta': {'content': 'check.'}}]},
            {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'call_a', 'function': {'name': 'search_read', 'arguments': ''}}]}}]},
            {'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': '{"model": '}}]}}]},
            {'choices': [{'delta': {'tool_calls': [{'index': 1, 'id': 'call_b', 'function': {'name': 'ping', 'arguments': '{}'}}]}}]},
            {'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': '"res.partner"}'}}]}}]},
            {'choices': [{'delta': {}, 'finish_reason': 'tool_calls'}]},
            {'choices': [], 'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'prompt_tokens_details': {'cached_tokens': 60}}},
        ]
        result, _sent, deltas = self._run_stream(
            self.openrouter, chunks, inputs=[{'role': 'user', 'content': 'hi'}]
        )
        self.assertEqual(result['text'], 'Let me check.')
        self.assertEqual(
            [(c['call_id'], c['name'], c['arguments']) for c in result['tool_calls']],
            [('call_a', 'search_read', {'model': 'res.partner'}), ('call_b', 'ping', {})],
        )
        self.assertEqual(
            result['carry_inputs'],
            [
                {'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'Let me check.'}]},
                {'type': 'function_call', 'call_id': 'call_a', 'name': 'search_read', 'arguments': '{"model": "res.partner"}'},
                {'type': 'function_call', 'call_id': 'call_b', 'name': 'ping', 'arguments': '{}'},
            ],
        )
        self.assertEqual(result['usage']['input_tokens'], 100)
        self.assertEqual(result['usage']['output_tokens'], 20)
        self.assertEqual(result['usage']['cache_read_tokens'], 60)
        kinds = [kind for kind, _data in deltas]
        self.assertEqual(kinds[0], 'reasoning')
        self.assertEqual(kinds.count('tool_start'), 2)
        self.assertIn(('tool_args', {'call_id': 'call_a', 'delta': '"res.partner"}'}), deltas)

    def test_carry_inputs_round_trip(self):
        chunks = [
            {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'call_a', 'function': {'name': 'ping', 'arguments': '{}'}}]}, 'finish_reason': 'tool_calls'}]},
        ]
        result, _sent, _deltas = self._run_stream(
            self.openrouter, chunks, inputs=[{'role': 'user', 'content': 'hi'}]
        )
        messages = self.openrouter._get_client()._inputs_to_messages(
            [{'role': 'user', 'content': 'hi'}]
            + result['carry_inputs']
            + [{'type': 'function_call_output', 'call_id': 'call_a', 'output': 'pong'}]
        )
        self.assertEqual([m['role'] for m in messages], ['user', 'assistant', 'tool'])

    def test_stream_length_truncates(self):
        chunks = [{'choices': [{'delta': {'content': 'partial'}, 'finish_reason': 'length'}]}]
        result, _sent, _deltas = self._run_stream(
            self.openrouter, chunks, inputs=[{'role': 'user', 'content': 'hi'}]
        )
        self.assertTrue(result['usage']['truncated'])
        self.assertIn('4096', result['text'])

    def test_stream_error_chunk_raises(self):
        chunks = [{'error': {'message': 'No endpoints found', 'code': 404}}]
        with self.assertRaisesRegex(Exception, 'No endpoints found'):
            self._run_stream(self.openrouter, chunks, inputs=[{'role': 'user', 'content': 'hi'}])

    def test_stream_cut_off_raises(self):
        chunks = [{'choices': [{'delta': {'content': 'half'}}]}]
        with self.assertRaisesRegex(Exception, 'closed before'):
            self._run_stream(self.openrouter, chunks, inputs=[{'role': 'user', 'content': 'hi'}])

    def test_reasoning_rejected_is_retried_without_it(self):
        bodies = []

        def fake_post(_self, path, body, timeout=None):
            bodies.append(json.loads(json.dumps(body)))
            if 'reasoning' in body:
                _self._raise('reasoning is not supported for this model')
            return {'choices': [{'message': {'content': 'fine'}, 'finish_reason': 'stop'}]}

        with patch.object(OpenRouterProvider, '_post_json', fake_post):
            result = self.openrouter._get_client().request(
                inputs=[{'role': 'user', 'content': 'hi'}],
                extra={'reasoning_effort': 'low'},
            )
        self.assertEqual(result['text'], 'fine')
        self.assertEqual(len(bodies), 2)
        self.assertNotIn('reasoning', bodies[1])

    # ----------------------------------------------------------
    # Buffered
    # ----------------------------------------------------------

    def test_buffered_response(self):
        payload = {
            'choices': [
                {
                    'message': {
                        'content': None,
                        'tool_calls': [{'id': 'x1', 'type': 'function', 'function': {'name': 'ping', 'arguments': '{"a": 1}'}}],
                    },
                    'finish_reason': 'tool_calls',
                }
            ],
            'usage': {'prompt_tokens': 5, 'completion_tokens': 7},
        }
        with patch.object(OpenRouterProvider, '_post_json', return_value=payload):
            result = self.openrouter._get_client().request(inputs=[{'role': 'user', 'content': 'hi'}])
        self.assertEqual(result['text'], '')
        self.assertEqual(result['tool_calls'][0]['arguments'], {'a': 1})
        self.assertEqual(result['usage']['output_tokens'], 7)
