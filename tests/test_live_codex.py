from __future__ import annotations

import json
import os
import unittest

from chemeagle_llm import LLMRequest, create_backend


@unittest.skipUnless(
    os.getenv("CHEMEAGLE_RUN_LIVE_CODEX") == "1",
    "set CHEMEAGLE_RUN_LIVE_CODEX=1 to consume a live Codex subscription request",
)
class LiveCodexSmokeTest(unittest.TestCase):
    def test_subscription_json_turn(self):
        backend = create_backend("codex")
        try:
            response = backend.generate(
                LLMRequest(
                    messages=[
                        {
                            "role": "user",
                            "content": "Return a JSON object with exactly {\"ok\": true}.",
                        }
                    ],
                    output_schema={
                        "type": "object",
                        "properties": {"ok": {"const": True}},
                        "required": ["ok"],
                        "additionalProperties": False,
                    },
                    timeout=120,
                )
            )
            self.assertEqual(json.loads(response.content), {"ok": True})
        finally:
            backend.close()


if __name__ == "__main__":
    unittest.main()
