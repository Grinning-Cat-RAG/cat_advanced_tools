"""The settings as the core stores and loads them: normalized by the settings model (a value is the one the validators
return, an enum is its value).

Run from the core root: ``python -m pytest cat/plugins/cat_advanced_tools/tests``.

The Cat imports every ``.py`` file of the plugin, tests included: at import time this module needs only the stdlib,
the Cat and the plugin are loaded in ``setUpModule``.
"""
import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock

PLUGIN_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Plugin = RecallSettings = ValidationError = None


def setUpModule():
    global Plugin, RecallSettings, ValidationError
    from pydantic import ValidationError
    from cat import RecallSettings
    from cat.looking_glass.mad_hatter.plugin import Plugin
LANGUAGES = ["English", "French", "German", "Italian", "Spanish", "Russian", "Chinese", "Japanese", "Korean", "Human"]


def _plugin() -> Plugin:
    plugin = Plugin(PLUGIN_PATH)
    plugin._load_decorated_functions()
    return plugin


def _cat(settings):
    cat = MagicMock()
    cat.mad_hatter.get_plugin.return_value.load_settings = AsyncMock(return_value=settings)
    return cat


class TestSettings(unittest.TestCase):
    def setUp(self):
        self.plugin = _plugin()
        self.hooks = {h.name: h.function for h in self.plugin.hooks}

    def _saved(self, payload):
        """The settings as PUT /plugins/settings/{id} stores them, and the default load_settings returns them."""
        return self.plugin.normalized_settings(payload)

    def test_saved_values_are_kept(self):
        payload = {"prompt_prefix": "p", "k": 2, "threshold": 0.5, "latest_n_history": 3, "language": "Italian"}
        self.assertEqual(self._saved(payload), payload)

    def test_default_values_are_kept(self):
        saved = self._saved(self.plugin.settings_model()().model_dump())
        self.assertEqual(saved["threshold"], 0.7)
        self.assertEqual(saved["latest_n_history"], 5)
        self.assertEqual(saved["language"], "English")

    def test_invalid_values_are_rejected(self):
        for payload in ({"threshold": 0}, {"threshold": -1.0}, {"latest_n_history": 0}):
            with self.assertRaises(ValidationError):
                self._saved(payload)

    def test_recall_settings(self):
        saved = self._saved({"k": 4, "threshold": 0.3, "latest_n_history": 2})
        config = asyncio.run(self.hooks["before_cat_recalls_memories"](RecallSettings(embedding=[0.0]), cat=_cat(saved)))
        self.assertEqual((config.k, config.threshold, config.latest_n_history), (4, 0.3, 2))

    def test_language_of_the_answer(self):
        for language in LANGUAGES:
            suffix = asyncio.run(self.hooks["agent_prompt_suffix"]("", cat=_cat(self._saved({"language": language}))))
            expected = "the user's language" if language == "Human" else language.lower()
            self.assertIn(f"ALWAYS answer in {expected}", suffix)

    def test_language_as_enum(self):
        language = self.plugin.settings_model()().language  # the model default, an enum member
        suffix = asyncio.run(self.hooks["agent_prompt_suffix"]("", cat=_cat({"language": language})))
        self.assertIn("ALWAYS answer in english", suffix)

    def test_prompt_prefix(self):
        prefix = asyncio.run(self.hooks["agent_prompt_prefix"]("", cat=_cat({"prompt_prefix": "use {json}"})))
        self.assertEqual(prefix, "use {{json}}")

    def test_any_settings(self):
        """Invariant: a valid payload is stored and read back with the same values (and drives the recall and the
        language); an invalid one is rejected."""
        from hypothesis import given, settings, strategies as st

        payloads = st.fixed_dictionaries({}, optional={
            "k": st.integers(min_value=0, max_value=100),
            "threshold": st.floats(min_value=-2, max_value=2, allow_nan=False),
            "latest_n_history": st.integers(min_value=-3, max_value=50),
            "language": st.sampled_from(LANGUAGES),
        })

        @settings(max_examples=200, deadline=None)
        @given(payloads)
        def check(payload):
            valid = payload.get("threshold", 1) > 0 and payload.get("latest_n_history", 1) >= 1
            if not valid:
                with self.assertRaises(ValidationError):
                    self._saved(payload)
                return
            saved = self._saved(payload)
            self.assertEqual(saved, payload)
            full = {**self.plugin.settings_model()().model_dump(mode="json"), **saved}
            config = asyncio.run(
                self.hooks["before_cat_recalls_memories"](RecallSettings(embedding=[0.0]), cat=_cat(full))
            )
            self.assertEqual(
                (config.k, config.threshold, config.latest_n_history),
                (full["k"], full["threshold"], full["latest_n_history"]),
            )
            suffix = asyncio.run(self.hooks["agent_prompt_suffix"]("", cat=_cat(full)))
            self.assertIn("ALWAYS answer in", suffix)

        check()


if __name__ == "__main__":
    unittest.main()
