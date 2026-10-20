"""
Persistent semantic cache for generated robot explanations.

Responsibility:
    Prevent repeated LLM calls for navigation situations that have already
    received a valid natural-language explanation.

Flow:
    ExplanationContext
            ↓
    SEMANTIC SIGNATURE
            ↓
    SHA-256 CACHE KEY
            ↓
        ┌── HIT  → cached explanation
        └── MISS → LLM provider
                     ↓
                  cache store

Important:
    Dynamic values such as timestamps, person IDs, positions and exact
    measurements are intentionally excluded from the semantic key.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional

from g1_social_msgs.msg import ExplanationContext


class ExplanationCache:
    """Persistent semantic explanation cache."""

    def __init__(
        self,
        cache_file: str,
    ) -> None:
        """
        Initialize persistent cache.

        Inputs:
            cache_file:
                JSON file used to persist explanations between executions.

        Side effects:
            Reads existing cache file when available.
        """

        self.cache_file = Path(
            os.path.expanduser(cache_file)
        )

        self.entries: Dict[str, dict] = {}

        self._load()

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get(
        self,
        context: ExplanationContext,
    ) -> Optional[dict]:
        """
        Return a cached explanation for an equivalent semantic situation.

        Output:
            Cached entry or None.
        """

        key = self.build_key(
            context
        )

        return self.entries.get(
            key
        )

    def put(
        self,
        context: ExplanationContext,
        text: str,
        provider: str,
        model: str,
    ) -> None:
        """
        Persist a successful LLM explanation.

        Inputs:
            context:
                Source deterministic explanation context.

            text:
                Generated natural-language explanation.

            provider:
                Provider responsible for the original generation.

            model:
                Model responsible for the original generation.

        Side effects:
            Updates cache memory and JSON persistence.
        """

        key = self.build_key(
            context
        )

        self.entries[key] = {
            "signature": self.build_signature(
                context
            ),
            "text": text,
            "provider": provider,
            "model": model,
        }

        self._save()

    # =========================================================================
    # SEMANTIC SIGNATURE
    # =========================================================================

    @staticmethod
    def build_signature(
        context: ExplanationContext,
    ) -> dict:
        """
        Identify the complete reusable semantic explanation context.

        Exclude transaction IDs, person IDs, timestamps and geometry.
        Include semantic wording because it carries navigation history.
        """
        return {
            "language": str(
                context.language or ""
            ).strip().lower(),

            "action": str(
                context.action or ""
            ).strip().upper(),

            "reason_code": str(
                context.reason_code or ""
            ).strip().upper(),

            "zone_type": str(
                context.zone_type or ""
            ).strip().upper(),

            "tracking_state": str(
                context.tracking_state or ""
            ).strip().upper(),

            "navigation_state": str(
                context.navigation_state or ""
            ).strip().upper(),

            "has_navigation_goal": bool(
                str(context.navigation_goal_id or "").strip()
            ),

            "navigation_active": bool(
                context.navigation_active
            ),

            "social_conflict_active": bool(
                context.social_conflict_active
            ),

            "requires_replan": bool(
                context.requires_replan
            ),

            "situation_summary": str(
                context.situation_summary or ""
            ).strip(),

            "robot_intention": str(
                context.robot_intention or ""
            ).strip(),

            "explanation_goal": str(
                context.explanation_goal or ""
            ).strip(),

            "speech_style": (
                "robot_first_person_human_second_person_v2"
            ),

            "cache_schema": 3,
        }
    
    @classmethod
    def build_key(
        cls,
        context: ExplanationContext,
    ) -> str:
        """
        Generate deterministic SHA-256 key from semantic context.
        """

        signature = cls.build_signature(
            context
        )

        serialized = json.dumps(
            signature,
            sort_keys=True,
            separators=(",", ":"),
        )

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def _load(
        self,
    ) -> None:
        """Load existing persistent cache."""

        if not self.cache_file.is_file():
            return

        try:
            data = json.loads(
                self.cache_file.read_text(
                    encoding="utf-8"
                )
            )

            if isinstance(data, dict):
                self.entries = data

        except Exception:
            # A damaged cache must never prevent robot explainability.
            self.entries = {}

    def _save(
        self,
    ) -> None:
        """
        Persist cache atomically.

        A temporary file is written first and then renamed.
        """

        self.cache_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_file = (
            self.cache_file.with_suffix(
                ".tmp"
            )
        )

        temporary_file.write_text(
            json.dumps(
                self.entries,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        temporary_file.replace(
            self.cache_file
        )
