DIALOGUE_TEXTS = {
    "en": {
        "passage": "May I pass?",
        "confirm_allow": (
            "I understood that I may pass. "
            "Is that correct?"
        ),
        "confirm_allow_left": (
            "I understood that I may pass on your left. "
            "Is that correct?"
        ),
        "confirm_allow_right": (
            "I understood that I may pass on your right. "
            "Is that correct?"
        ),
        "confirm_deny": (
            "I understood that you do not want me to pass. "
            "Is that correct?"
        ),
        "alternative_left": (
            "The previous passage is difficult. "
            "May I pass on your left instead?"
        ),
        "alternative_right": (
            "The previous passage is difficult. "
            "May I pass on your right instead?"
        ),
    },
    "fr": {
        "passage": "Puis-je passer ?",
        "confirm_allow": (
            "J'ai compris que je peux passer. "
            "Est-ce bien cela ?"
        ),
        "confirm_allow_left": (
            "J'ai compris que je peux passer par votre gauche. "
            "Est-ce bien cela ?"
        ),
        "confirm_allow_right": (
            "J'ai compris que je peux passer par votre droite. "
            "Est-ce bien cela ?"
        ),
        "confirm_deny": (
            "J'ai compris que vous ne souhaitez pas que je passe. "
            "Est-ce bien cela ?"
        ),
        "alternative_left": (
            "Le passage précédent est difficile. "
            "Puis-je passer par votre gauche à la place ?"
        ),
        "alternative_right": (
            "Le passage précédent est difficile. "
            "Puis-je passer par votre droite à la place ?"
        ),
    },
    "es": {
        "passage": "¿Puedo pasar?",
        "confirm_allow": (
            "He entendido que puedo pasar. "
            "¿Es correcto?"
        ),
        "confirm_allow_left": (
            "He entendido que puedo pasar por su izquierda. "
            "¿Es correcto?"
        ),
        "confirm_allow_right": (
            "He entendido que puedo pasar por su derecha. "
            "¿Es correcto?"
        ),
        "confirm_deny": (
            "He entendido que no desea que pase. "
            "¿Es correcto?"
        ),
        "alternative_left": (
            "El paso anterior es difícil. "
            "¿Puedo pasar por su izquierda en su lugar?"
        ),
        "alternative_right": (
            "El paso anterior es difícil. "
            "¿Puedo pasar por su derecha en su lugar?"
        ),
    },
}


DEFAULT_DIALOGUE_LANGUAGE = "en"


def normalize_dialogue_language(language):
    normalized = str(language or "").strip().lower()

    if normalized in DIALOGUE_TEXTS:
        return normalized

    return DEFAULT_DIALOGUE_LANGUAGE


def get_dialogue_text(key, language="en"):
    language = normalize_dialogue_language(language)

    language_texts = DIALOGUE_TEXTS.get(
        language,
        DIALOGUE_TEXTS[DEFAULT_DIALOGUE_LANGUAGE],
    )

    text = language_texts.get(key)

    if text:
        return text

    fallback_text = DIALOGUE_TEXTS[
        DEFAULT_DIALOGUE_LANGUAGE
    ].get(key)

    if fallback_text:
        return fallback_text

    raise KeyError(
        f"Unknown dialogue text key: {key!r}"
    )