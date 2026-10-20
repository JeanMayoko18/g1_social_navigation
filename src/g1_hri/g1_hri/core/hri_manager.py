from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class HRIIntent:
    """
    Internal representation of a human intention.

    This class is independent from ROS messages.
    """

    interaction_id: str = ""
    person_id: str = ""

    intent: str = "UNKNOWN"
    direction: str = "NONE"
    target: str = ""
    constraint: str = ""

    confidence: float = 0.0
    utterance: str = ""

    source: str = "UNKNOWN"
    modality: str = "UNKNOWN"


class HRIManager:
    """
    Core HRI manager.

    This class contains HRI interpretation logic and must remain
    independent from ROS2.
    """

    def __init__(self, min_confidence: float = 0.0):
        self.min_confidence = min_confidence

    def process_intent(
        self,
        intent: str,
        data: Optional[Dict[str, Any]] = None,
        confidence: float = 0.0,
        source: str = "UNKNOWN",
        modality: str = "UNKNOWN",
        interaction_id: str = "",
        person_id: str = "",
        utterance: str = "",
    ) -> HRIIntent:
        """
        Convert a normalized intent into the internal HRI representation.
        """

        data = data or {}

        confidence = max(0.0, min(1.0, confidence))

        if confidence < self.min_confidence:
            return HRIIntent(
                interaction_id=interaction_id,
                intent="UNKNOWN",
                confidence=confidence,
                utterance=utterance,
                source=source,
                modality=modality,
            )

        normalized_intent = self._normalize_intent(intent)

        target = self._extract_target(data)
        direction = self._extract_direction(data)
        constraint = self._extract_constraint(data)

        return HRIIntent(
            interaction_id=interaction_id,
            person_id=person_id,

            intent=normalized_intent,
            direction=direction,
            target=target,
            constraint=constraint,

            confidence=confidence,
            utterance=utterance,

            source=source,
            modality=modality,
        )

    def _normalize_intent(self, intent: str) -> str:
        """
        Normalize ROS4HRI/custom intent names to the project's
        internal intent vocabulary.
        """

        if not intent:
            return "UNKNOWN"

        value = intent.strip().upper()

        mapping = {
            "MOVE_TO": "MOVE_TO",
            "GUIDE": "GUIDE",
            "ENGAGE_WITH": "ENGAGE_WITH",
            "GREET": "GREET",
            "SAY": "SAY",
            "STOP_ACTIVITY": "STOP_ACTIVITY",
            "START_ACTIVITY": "START_ACTIVITY",
            "SUSPEND": "SUSPEND",
            "WAKEUP": "WAKEUP",
        }

        return mapping.get(value, value)

    def _extract_target(self, data: Dict[str, Any]) -> str:
        """
        Extract the semantic target from ROS4HRI intent data.
        """

        for key in ("goal", "object", "recipient", "target"):
            value = data.get(key)

            if value is not None and str(value).strip():
                return str(value).strip()

        return ""

    def _extract_direction(self, data: Dict[str, Any]) -> str:
        """
        Extract an optional spatial direction.
        """

        value = data.get("direction")

        if value is None:
            return "NONE"

        value = str(value).strip().upper()

        allowed = {
            "LEFT",
            "RIGHT",
            "FRONT",
            "BACK",
            "NONE",
        }

        return value if value in allowed else "NONE"

    def _extract_constraint(self, data: Dict[str, Any]) -> str:
        """
        Extract an optional human constraint.
        """

        value = data.get("constraint")

        if value is None:
            return ""

        return str(value).strip()