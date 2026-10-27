from __future__ import annotations

import requests

from odoo.addons.muk_ai.providers.region import CUSTOM, Region
from odoo.addons.tricasia_llm_provider.providers.compat import (
    DEFAULT_EFFORTS,
    ChatCompletionsProvider,
)

FALLBACK_CONTEXT = 8192


class OllamaProvider(ChatCompletionsProvider):
    """A self-hosted Ollama server through its OpenAI-compatible ``/v1`` routes.

    The default URL reaches the Docker host, where Ollama usually runs next
    to an Odoo container; a bare-metal Odoo picks the localhost region.
    """

    name = 'ollama'
    label = 'Ollama'
    default_model = 'llama3.1'
    default_url = 'http://host.docker.internal:11434/v1'
    regions = (
        Region('localhost', 'This Machine', 'http://localhost:11434/v1'),
        CUSTOM,
    )

    # ----------------------------------------------------------
    # Config
    # ----------------------------------------------------------

    @property
    def authenticated(self) -> bool:
        """A local Ollama needs no key; one behind a proxy may still send it."""
        return True

    @property
    def native_url(self) -> str:
        """Return the server root that serves Ollama's own ``/api`` routes."""
        return self.api_url.rstrip('/').removesuffix('/v1')

    # ----------------------------------------------------------
    # Contract
    # ----------------------------------------------------------

    def list_models(self) -> list[dict]:
        """Return the pulled models that can call tools, with their context size.

        ``/api/tags`` only names the models, so each one is asked for its
        capabilities and context length through ``/api/show``.
        """
        payload = self._get_json(f'{self.native_url}/api/tags')
        entries = []
        for model in payload.get('models') or []:
            if not (technical_name := model.get('name') or model.get('model')):
                continue
            if entry := self._catalog_entry(technical_name, self._show(technical_name)):
                entries.append(entry)
        return entries

    # ----------------------------------------------------------
    # Catalogue
    # ----------------------------------------------------------

    def _show(self, technical_name: str) -> dict:
        """Return the ``/api/show`` details of one pulled model.

        :raise UserError: on HTTP or transport errors
        """
        try:
            response = self._http_session().post(
                f'{self.native_url}/api/show',
                headers=self.headers(),
                json={'model': technical_name},
                timeout=self.request_timeout,
            )
            response.raise_for_status()
        except requests.HTTPError as error:
            self._raise(getattr(error.response, 'text', '') or str(error))
        except requests.RequestException as error:
            self._raise(error)
        return response.json()

    @staticmethod
    def _catalog_entry(technical_name: str, details: dict) -> dict | None:
        """Map a model and its ``/api/show`` details to an entry, or ``None`` to skip it.

        Local models cost nothing per token. The context length sits in
        ``model_info`` under an architecture prefix, e.g. ``llama.context_length``.
        """
        capabilities = details.get('capabilities') or []
        if 'tools' not in capabilities:
            return None
        context = next(
            (
                value
                for key, value in (details.get('model_info') or {}).items()
                if key.endswith('.context_length') and value
            ),
            FALLBACK_CONTEXT,
        )
        return {
            'technical_name': technical_name,
            'name': technical_name,
            'context_window': int(context),
            'input_rate': 0.0,
            'output_rate': 0.0,
            'cache_read_rate': 0.0,
            'cache_write_rate': 0.0,
            'reasoning_efforts': list(DEFAULT_EFFORTS)
            if 'thinking' in capabilities
            else False,
        }
