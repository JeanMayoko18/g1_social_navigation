
"""Direction and perspective semantics for negotiated robot passage."""

import re
from dataclasses import dataclass
from typing import Optional


VALID_DIRECTIONS = {"LEFT", "RIGHT"}
VALID_REFERENCES = {"PERSON", "ROBOT", "AMBIGUOUS"}

FACE_TO_FACE_ASSUMED = "FACE_TO_FACE_ASSUMED"
ORIENTATION_UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class PassageDirection:
    """Preserve the requested direction and its reference frame."""

    direction: str
    reference: str
    explicit_reference: bool


SIDE_WORDS = r"(?:left|right|gauche|droite|izquierda|derecha)"

# The human is speaking to the robot.
# First-person expressions refer to the human.
PERSON_PATTERNS = (
    # English, French, Spanish
    rf"\b(?:my|mon|ma|mes|mi|mis)\s+{SIDE_WORDS}\b",

    # Short answers: "mine", "la mienne", "la mía"
    r"\bmine\b",
    r"\b(?:la|le|de la|de le)\s+mienne\b",
    r"\b(?:la|le|de la|de le)\s+mien\b",
    r"\b(?:la|el|de la|del)\s+m[ií]a\b",
    r"\b(?:el|lo|del)\s+m[ií]o\b",
)

# Second-person expressions refer to the robot.
ROBOT_PATTERNS = (
    # English, French, Spanish (informal and formal address)
    rf"\b(?:your|ton|ta|tes|votre|vos|tu|tus|su|sus)\s+{SIDE_WORDS}\b",

    # Short answers: "yours", "la vôtre", "la suya"
    r"\byours\b",
    r"\b(?:la|le|de la|de le)\s+v[oô]tre\b",
    r"\b(?:les|des)\s+v[oô]tres\b",
    r"\b(?:la|el|de la|del)\s+suya\b",
    r"\b(?:el|lo|del)\s+suyo\b",
)


def _matches_any(patterns, utterance):
    return any(
        re.search(pattern, utterance, flags=re.IGNORECASE)
        for pattern in patterns
    )


def interpret_passage_direction(
    direction,
    utterance,
) -> Optional[PassageDirection]:
    """
    Interpret the direction and the human/robot perspective.

    Supported languages: French, English, Spanish.

    First person -> PERSON.
    Second person -> ROBOT.
    No reference or conflicting references -> AMBIGUOUS.

    Short possessive answers retain the direction supplied
    by the calling dialogue context.

    This function never grants navigation authorization.
    """
    direction = str(direction or "").strip().upper()
    utterance = str(utterance or "").strip().lower()

    if direction not in VALID_DIRECTIONS:
        return None

    person_match = _matches_any(
        PERSON_PATTERNS,
        utterance,
    )

    robot_match = _matches_any(
        ROBOT_PATTERNS,
        utterance,
    )

    if person_match and robot_match:
        reference = "AMBIGUOUS"

    elif person_match:
        reference = "PERSON"

    elif robot_match:
        reference = "ROBOT"

    else:
        reference = "AMBIGUOUS"

    return PassageDirection(
        direction=direction,
        reference=reference,
        explicit_reference=(
            reference != "AMBIGUOUS"
        ),
    )


def resolve_robot_direction(
    passage: PassageDirection,
    *,
    spatial_relation: str,
) -> Optional[str]:
    """
    Resolve a confirmed direction into the robot frame.

    PERSON is inverted only when FACE_TO_FACE_ASSUMED
    has been explicitly established.

    This function does not authorize navigation.
    """
    if passage.direction not in VALID_DIRECTIONS:
        return None

    if passage.reference == "ROBOT":
        return passage.direction

    if passage.reference != "PERSON":
        return None

    if spatial_relation != FACE_TO_FACE_ASSUMED:
        return None

    return (
        "RIGHT"
        if passage.direction == "LEFT"
        else "LEFT"
    )
