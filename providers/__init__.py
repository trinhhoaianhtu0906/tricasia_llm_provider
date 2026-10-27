from __future__ import annotations

from odoo.addons.muk_ai.providers import REGISTRY

from .compat import ChatCompletionsProvider
from .openrouter import OpenRouterProvider
from .ollama import OllamaProvider


REGISTRY[OpenRouterProvider.name] = OpenRouterProvider
REGISTRY[OllamaProvider.name] = OllamaProvider
