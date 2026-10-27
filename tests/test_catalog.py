from unittest.mock import patch

from odoo.exceptions import UserError
from odoo.tests.common import TransactionCase, tagged

from odoo.addons.tricasia_llm_provider.models import provider as provider_module
from odoo.addons.tricasia_llm_provider.providers.mistral_catalog import mistral_entry
from odoo.addons.tricasia_llm_provider.providers.ollama import OllamaProvider
from odoo.addons.tricasia_llm_provider.providers.openrouter import OpenRouterProvider


@tagged('post_install', '-at_install')
class TestModelCatalog(TransactionCase):
    """Reading vendor model lists and syncing them into muk_ai.model."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.openrouter = cls.env.ref('tricasia_llm_provider.provider_openrouter')
        cls.ollama = cls.env.ref('tricasia_llm_provider.provider_ollama')
        cls.mistral = cls.env.ref('muk_ai_mistral.provider_mistral')

    def _chat_models(self, provider, active=True):
        return self.env['muk_ai.model'].with_context(active_test=False).search(
            [('provider_id', '=', provider.id), ('modality', '=', 'chat'), ('active', '=', active)]
        )

    # ----------------------------------------------------------
    # Entries
    # ----------------------------------------------------------

    def test_openrouter_entry(self):
        entry = OpenRouterProvider._catalog_entry(
            {
                'id': 'anthropic/claude-haiku-5.5',
                'name': 'Anthropic: Claude Haiku 5.5',
                'context_length': 1000000,
                'pricing': {'prompt': '0.0000001', 'completion': '0.0000005', 'input_cache_read': '0.00000001'},
                'supported_parameters': ['tools', 'reasoning', 'max_tokens'],
            }
        )
        self.assertEqual(entry['technical_name'], 'anthropic/claude-haiku-5.5')
        self.assertEqual(entry['context_window'], 1000000)
        self.assertAlmostEqual(entry['input_rate'], 0.1)
        self.assertAlmostEqual(entry['output_rate'], 0.5)
        self.assertAlmostEqual(entry['cache_read_rate'], 0.01)
        self.assertEqual(entry['cache_write_rate'], 0.0)
        self.assertEqual(entry['reasoning_efforts'], ['low', 'medium', 'high'])

    def test_openrouter_entry_skips(self):
        base = {'id': 'x/y', 'context_length': 1000, 'pricing': {}, 'supported_parameters': ['tools']}
        self.assertIsNotNone(OpenRouterProvider._catalog_entry(base))
        self.assertIsNone(OpenRouterProvider._catalog_entry({**base, 'supported_parameters': ['max_tokens']}))
        self.assertIsNone(OpenRouterProvider._catalog_entry({**base, 'id': 'x/y:batch'}))
        self.assertIsNone(OpenRouterProvider._catalog_entry({**base, 'context_length': None}))
        router = OpenRouterProvider._catalog_entry({**base, 'pricing': {'prompt': '-1', 'completion': 'n/a'}})
        self.assertEqual((router['input_rate'], router['output_rate']), (0.0, 0.0))
        self.assertFalse(router['reasoning_efforts'])

    def test_ollama_entry(self):
        entry = OllamaProvider._catalog_entry(
            'qwen3:8b',
            {'capabilities': ['completion', 'tools', 'thinking'], 'model_info': {'qwen3.context_length': 40960}},
        )
        self.assertEqual(entry['context_window'], 40960)
        self.assertEqual(entry['input_rate'], 0.0)
        self.assertEqual(entry['reasoning_efforts'], ['low', 'medium', 'high'])
        self.assertIsNone(OllamaProvider._catalog_entry('gemma:2b', {'capabilities': ['completion']}))
        fallback = OllamaProvider._catalog_entry('x', {'capabilities': ['tools']})
        self.assertEqual(fallback['context_window'], 8192)
        self.assertFalse(fallback['reasoning_efforts'])

    def test_ollama_list_models(self):
        shows = {
            'qwen3:8b': {'capabilities': ['completion', 'tools'], 'model_info': {'qwen3.context_length': 40960}},
            'nomic-embed-text:latest': {'capabilities': ['embedding']},
        }
        client = self.ollama._get_client()
        with patch.object(OllamaProvider, '_get_json', return_value={'models': [{'name': name} for name in shows]}) as get, \
                patch.object(OllamaProvider, '_show', side_effect=lambda name: shows[name]):
            entries = client.list_models()
        get.assert_called_once_with('http://host.docker.internal:11434/api/tags')
        self.assertEqual([e['technical_name'] for e in entries], ['qwen3:8b'])

    def test_mistral_entry(self):
        row = {
            'id': 'mistral-small-2603',
            'name': 'mistral-small-2603',
            'max_context_length': 262144,
            'capabilities': {'completion_chat': True, 'function_calling': True},
        }
        self.assertEqual(mistral_entry(row), {'technical_name': 'mistral-small-2603', 'name': 'mistral-small-2603', 'context_window': 262144})
        self.assertIsNone(mistral_entry({**row, 'deprecation': '2026-12-01T00:00:00Z'}))
        self.assertIsNone(mistral_entry({**row, 'capabilities': {'completion_chat': True}}))
        self.assertIsNone(mistral_entry({**row, 'capabilities': {'function_calling': True}}))

    # ----------------------------------------------------------
    # Sync
    # ----------------------------------------------------------

    def _entry(self, name, **values):
        return {
            'technical_name': name,
            'name': name.upper(),
            'context_window': 1000,
            'input_rate': 1.0,
            'output_rate': 2.0,
            'cache_read_rate': 0.0,
            'cache_write_rate': 0.0,
            'reasoning_efforts': False,
            **values,
        }

    def test_sync_creates_updates_and_archives(self):
        counts = self.openrouter._sync_models([self._entry('a/one'), self._entry('a/two')])
        self.assertEqual(counts, {'created': 2, 'updated': 0, 'archived': 0})
        self.assertEqual(self.openrouter.default_chat_model_id.technical_name, 'a/one')

        counts = self.openrouter._sync_models([self._entry('a/two', input_rate=3.0), self._entry('a/three')])
        self.assertEqual(counts, {'created': 1, 'updated': 1, 'archived': 0})
        two = self._chat_models(self.openrouter).filtered(lambda m: m.technical_name == 'a/two')
        self.assertEqual(two.input_rate, 3.0)
        self.assertTrue(self._chat_models(self.openrouter).filtered(lambda m: m.technical_name == 'a/one'),
                        'the default chat model is never archived')

        self.openrouter.default_chat_model_id = two
        counts = self.openrouter._sync_models([self._entry('a/two')])
        self.assertEqual(counts['archived'], 2)
        self.assertEqual(
            sorted(self._chat_models(self.openrouter, active=False).mapped('technical_name')),
            ['a/one', 'a/three'],
        )

        counts = self.openrouter._sync_models([self._entry('a/two'), self._entry('a/one')])
        self.assertEqual(counts, {'created': 0, 'updated': 2, 'archived': 0})
        self.assertIn('a/one', self._chat_models(self.openrouter).mapped('technical_name'))

    def test_sync_prefers_class_default_model(self):
        self.openrouter._sync_models([self._entry('z/first'), self._entry(OpenRouterProvider.default_model)])
        self.assertEqual(self.openrouter.default_chat_model_id.technical_name, OpenRouterProvider.default_model)

    def test_sync_mistral_keeps_seeded_rates(self):
        large = self._chat_models(self.mistral).filtered(lambda m: m.technical_name == 'mistral-large-latest')
        rate = large.input_rate
        rows = {'data': [{'id': 'mistral-large-latest', 'max_context_length': 262144,
                          'capabilities': {'completion_chat': True, 'function_calling': True}}]}
        with patch.object(provider_module, 'CATALOGS', {'mistral': lambda client: [mistral_entry(r) for r in rows['data']]}):
            self.mistral.action_sync_models()
        self.assertEqual(large.input_rate, rate)
        self.assertTrue(large.active)
        self.assertEqual(self.mistral.default_chat_model_id.technical_name, 'mistral-medium-latest')

    def test_action_reports_counts(self):
        with patch.object(OpenRouterProvider, 'list_models', return_value=[self._entry('a/one')]):
            action = self.openrouter.action_sync_models()
        self.assertIn('1 added', action['params']['message'])

    def test_can_sync(self):
        self.assertTrue(self.openrouter.can_sync_models)
        self.assertTrue(self.ollama.can_sync_models)
        self.assertTrue(self.mistral.can_sync_models)
        openai = self.env.ref('muk_ai.provider_openai')
        self.assertFalse(openai.can_sync_models)
        with self.assertRaises(UserError):
            openai._fetch_model_catalog()
