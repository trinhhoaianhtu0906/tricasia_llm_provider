{
    'name': 'Tricasia LLM Providers',
    'summary': 'OpenRouter, Ollama and Mistral providers for MuK AI, with model sync',
    'description': """
        Adds OpenRouter and Ollama as MuK AI providers over the
        OpenAI-compatible Chat Completions API, and brings in Mistral
        through muk_ai_mistral. Every one of them gets a Sync Models
        button that reads the vendor's model list and keeps the
        muk_ai.model catalogue in step with it.
    """,
    'version': '19.0.1.0.0',
    'category': 'Productivity',
    'license': 'LGPL-3',
    'author': 'Tricasia',
    'depends': [
        'muk_ai',
        'muk_ai_mistral',
    ],
    'data': [
        'data/provider.xml',
        'views/provider.xml',
    ],
    'installable': True,
    'application': False,
    'auto_install': False,
}
