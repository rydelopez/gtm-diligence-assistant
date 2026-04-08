from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from gtm_diligence_assistant.config import load_local_env


class ConfigTests(unittest.TestCase):
    def test_load_local_env_parses_inline_comments_without_overwriting_existing_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "MODEL=openai # local default",
                        "OPENAI_MODEL=gpt-5",
                        'LANGSMITH_PROJECT="gtm diligence"',
                    ]
                ),
                encoding="utf-8",
            )

            original_model = os.environ.get("MODEL")
            os.environ["MODEL"] = "anthropic"
            try:
                loaded = load_local_env(env_path)
                self.assertEqual(os.environ["MODEL"], "anthropic")
                self.assertEqual(os.environ["OPENAI_MODEL"], "gpt-5")
                self.assertEqual(os.environ["LANGSMITH_PROJECT"], "gtm diligence")
                self.assertIn("OPENAI_MODEL", loaded)
            finally:
                if original_model is None:
                    os.environ.pop("MODEL", None)
                else:
                    os.environ["MODEL"] = original_model
                os.environ.pop("OPENAI_MODEL", None)
                os.environ.pop("LANGSMITH_PROJECT", None)


if __name__ == "__main__":
    unittest.main()
