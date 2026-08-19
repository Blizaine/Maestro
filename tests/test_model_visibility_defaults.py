import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STORE_PATH = ROOT / "ui" / "src" / "stores" / "useStore.ts"


def _store_source() -> str:
    return STORE_PATH.read_text(encoding="utf-8")


def _typescript_block(source: str, declaration: str) -> str:
    return source.split(declaration, 1)[1].split("])\n", 1)[0]


class ModelVisibilityDefaultsTests(unittest.TestCase):
    def test_fresh_defaults_exclude_unreachable_and_mature_only_models(self):
        source = _store_source()
        defaults = _typescript_block(
            source,
            "const DEFAULT_ENABLED_MODELS = new Set([",
        )
        self.assertNotIn("'animate'", defaults)
        self.assertNotIn("'mmaudio_nsfw'", defaults)
        self.assertIn("'mmaudio_v2'", defaults)

    def test_mmaudio_nsfw_is_gated_by_mature_mode(self):
        source = _store_source()
        match = re.search(
            r"\{ model_type: 'mmaudio_nsfw'.*?\}",
            source,
        )
        self.assertIsNotNone(match)
        self.assertIn("nsfw_only: true", match.group(0))

    def test_edit_mode_prefers_the_enabled_ltx23_model(self):
        source = _store_source()
        defaults = source.split(
            "const modeDefaultModel: Record<GenerationMode, string> = {",
            1,
        )[1].split("}\n", 1)[0]
        self.assertIn("avatar: 'ltx2_22B_distilled_1_1'", defaults)
        self.assertIn(
            "enabledModels?: ReadonlySet<string>",
            source,
        )
        self.assertIn(
            "enabledModels.has(savedModel)",
            source,
        )


if __name__ == "__main__":
    unittest.main()
