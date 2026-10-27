from __future__ import annotations

import requests

from odoo.addons.muk_ai.providers.base import ProviderBase


def mistral_models(client: ProviderBase) -> list[dict]:
    """Return Mistral's live chat models that call tools, from ``GET /models``.

    The muk_ai_mistral adapter stays untouched, so its catalogue is read here
    through its own session, URL and headers. Mistral publishes no prices on
    this route; synced models keep the rates they already have.

    :raise UserError: on HTTP or transport errors, or without an API key
    """
    try:
        response = client._http_session().get(
            f'{client.api_url}/models',
            headers=client.headers(),
            timeout=client.request_timeout,
        )
        response.raise_for_status()
    except requests.HTTPError as error:
        client._raise(getattr(error.response, 'text', '') or str(error))
    except requests.RequestException as error:
        client._raise(error)
    entries = (mistral_entry(model) for model in response.json().get('data') or [])
    return [entry for entry in entries if entry]


def mistral_entry(model: dict) -> dict | None:
    """Map one Mistral ``/models`` row to a catalogue entry, or ``None`` to skip it."""
    capabilities = model.get('capabilities') or {}
    if (
        not model.get('id')
        or model.get('deprecation')
        or model.get('archived')
        or not capabilities.get('completion_chat')
        or not capabilities.get('function_calling')
        or not model.get('max_context_length')
    ):
        return None
    return {
        'technical_name': model['id'],
        'name': model.get('name') or model['id'],
        'context_window': int(model['max_context_length']),
    }
