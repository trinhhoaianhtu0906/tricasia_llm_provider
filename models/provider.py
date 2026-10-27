from __future__ import annotations

from odoo import _, api, fields, models
from odoo.exceptions import UserError

from odoo.addons.muk_ai.providers import REGISTRY
from odoo.addons.tricasia_llm_provider.providers.mistral_catalog import mistral_models

CATALOGS = {
    'mistral': mistral_models,
}


class AIProvider(models.Model):
    """Add a model sync that reads the vendor's own model list."""

    _inherit = 'muk_ai.provider'

    # ----------------------------------------------------------
    # Fields
    # ----------------------------------------------------------

    can_sync_models = fields.Boolean(
        compute='_compute_can_sync_models',
        string='Can Sync Models',
    )

    # ----------------------------------------------------------
    # Helper
    # ----------------------------------------------------------

    def _fetch_model_catalog(self) -> list[dict]:
        """Return the vendor's chat models as normalized catalogue entries.

        :raise UserError: when the provider has no catalogue to read
        """
        client = self._get_client()
        if fetch := CATALOGS.get(self.name):
            return fetch(client)
        if hasattr(client, 'list_models'):
            return client.list_models()
        raise UserError(
            _('Provider %(provider)s cannot list its models.', provider=self.display_name)
        )

    def _sync_models(self, entries: list[dict]) -> dict:
        """Create, update and archive chat models so they match ``entries``.

        A model the vendor no longer lists is archived rather than deleted,
        so agents and past sessions that point at it keep their reference;
        the provider's default chat model is never archived.

        :return: the counts of created, updated and archived models
        """
        self.ensure_one()
        Model = self.env['muk_ai.model'].with_context(active_test=False)
        existing = {
            record.technical_name: record
            for record in Model.search(
                [('provider_id', '=', self.id), ('modality', '=', 'chat')]
            )
        }
        created = updated = 0
        for entry in entries:
            values = {**entry, 'active': True}
            if record := existing.pop(entry['technical_name'], None):
                record.write(values)
                updated += 1
            else:
                Model.create(
                    {**values, 'provider_id': self.id, 'modality': 'chat', 'currency': 'USD'}
                )
                created += 1
        stale = Model.browse(
            record.id
            for record in existing.values()
            if record.active and record != self.default_chat_model_id
        )
        stale.write({'active': False})
        if not self.default_chat_model_id.active:
            self.default_chat_model_id = self._pick_default_model(
                self._get_client().default_model
            )
        return {'created': created, 'updated': updated, 'archived': len(stale)}

    def _pick_default_model(self, preferred: str) -> models.BaseModel:
        """Return the active chat model named ``preferred``, else the first active one."""
        domain = [
            ('provider_id', '=', self.id),
            ('modality', '=', 'chat'),
            ('active', '=', True),
        ]
        Model = self.env['muk_ai.model']
        return Model.search(domain + [('technical_name', '=', preferred)], limit=1) or (
            Model.search(domain, limit=1)
        )

    # ----------------------------------------------------------
    # Actions
    # ----------------------------------------------------------

    def action_sync_models(self) -> dict:
        """Pull the vendor's model list into the catalogue and report what changed."""
        self.ensure_one()
        counts = self._sync_models(self._fetch_model_catalog())
        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'type': 'success',
                'title': _('Models synced'),
                'message': _(
                    '%(provider)s: %(created)s added, %(updated)s updated, '
                    '%(archived)s archived.',
                    provider=self.display_name,
                    **counts,
                ),
                'sticky': False,
                'next': {'type': 'ir.actions.client', 'tag': 'soft_reload'},
            },
        }

    # ----------------------------------------------------------
    # Compute
    # ----------------------------------------------------------

    @api.depends('name')
    def _compute_can_sync_models(self) -> None:
        """Offer the sync wherever a catalogue reader exists for the provider."""
        for record in self:
            impl = REGISTRY.get(record.name)
            record.can_sync_models = record.name in CATALOGS or bool(
                impl and hasattr(impl, 'list_models')
            )
