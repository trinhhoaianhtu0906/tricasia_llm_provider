from __future__ import annotations

from odoo.addons.tricasia_llm_provider.providers.compat import (
    DEFAULT_EFFORTS,
    ChatCompletionsProvider,
)

PER_MILLION = 1_000_000


class OpenRouterProvider(ChatCompletionsProvider):
    """OpenRouter: hundreds of vendors' models behind one OpenAI-compatible key."""

    name = 'openrouter'
    label = 'OpenRouter'
    default_model = 'deepseek/deepseek-v4.1-flash'
    default_url = 'https://openrouter.ai/api/v1'

    # ----------------------------------------------------------
    # Contract
    # ----------------------------------------------------------

    def headers(self) -> dict:
        """Add the attribution headers OpenRouter asks every app to send."""
        base_url = self.env['ir.config_parameter'].sudo().get_param('web.base.url')
        return {
            **super().headers(),
            'HTTP-Referer': base_url or 'https://www.odoo.com',
            'X-Title': 'Odoo MuK AI',
        }

    def list_models(self) -> list[dict]:
        """Return OpenRouter's tool-calling chat models with their USD pricing."""
        payload = self._get_json(f'{self.api_url}/models')
        entries = (self._catalog_entry(model) for model in payload.get('data') or [])
        return [entry for entry in entries if entry]

    # ----------------------------------------------------------
    # Hooks
    # ----------------------------------------------------------

    def _apply_reasoning(self, body: dict, effort: str) -> None:
        """OpenRouter takes the tier in its unified ``reasoning`` object."""
        body['reasoning'] = {'effort': effort}

    def _file_part(self, block: dict) -> dict:
        """OpenRouter accepts files (PDFs above all) as base64 ``file`` parts."""
        mimetype = block.get('mimetype') or 'application/octet-stream'
        return {
            'type': 'file',
            'file': {
                'filename': block.get('filename') or 'attachment',
                'file_data': f'data:{mimetype};base64,{block.get("data_b64", "")}',
            },
        }

    # ----------------------------------------------------------
    # Catalogue
    # ----------------------------------------------------------

    @staticmethod
    def _rate(pricing: dict, key: str) -> float:
        """Convert a per-token USD price string to USD per million tokens.

        Router models quote ``-1`` for "depends on the routed model"; that
        and anything unparsable count as zero.
        """
        try:
            rate = float(pricing.get(key) or 0) * PER_MILLION
        except (TypeError, ValueError):
            return 0.0
        return round(rate, 6) if rate > 0 else 0.0

    @classmethod
    def _catalog_entry(cls, model: dict) -> dict | None:
        """Map one ``/models`` row to a catalogue entry, or ``None`` to skip it.

        An agent always declares its Odoo tools, so a model without tool
        calling cannot serve; ``:batch`` variants only answer the batch API.
        """
        technical_name = model.get('id') or ''
        parameters = model.get('supported_parameters') or []
        if (
            not technical_name
            or technical_name.endswith(':batch')
            or 'tools' not in parameters
            or not model.get('context_length')
        ):
            return None
        pricing = model.get('pricing') or {}
        reasoning = 'reasoning' in parameters or 'reasoning_effort' in parameters
        return {
            'technical_name': technical_name,
            'name': model.get('name') or technical_name,
            'context_window': int(model['context_length']),
            'input_rate': cls._rate(pricing, 'prompt'),
            'output_rate': cls._rate(pricing, 'completion'),
            'cache_read_rate': cls._rate(pricing, 'input_cache_read'),
            'cache_write_rate': cls._rate(pricing, 'input_cache_write'),
            'reasoning_efforts': list(DEFAULT_EFFORTS) if reasoning else False,
        }
