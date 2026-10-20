"""
Shared base class for generative language-model explanation providers.

Responsibility:
    Centralize provider-independent prompt construction for all generative
    language-model backends.

Flow:
    ExplanationContext
            ↓
    SEMANTIC PROMPT CONSTRUCTION
            ↓
    provider-specific inference
        OpenAI / Gemini / Local
            ↓
    LLMExplanationResult

Important:
    Dynamic numerical measurements and identifiers are intentionally excluded
    from the reusable semantic prompt.

    This allows equivalent navigation situations to share the same cached
    explanation without producing stale numerical statements.
"""

from g1_social_msgs.msg import ExplanationContext

from g1_robot_explainability.llm_provider import (
    LLMProvider,
)

import re


FORBIDDEN_ROBOT_REFERENCES = (
    r"\bthe robot\b",
    r"\bthis robot\b",
    r"\bthe service robot\b",
    r"\bthe system\b",
    r"\brobot is\b",
    r"\brobot has\b",
    r"\brobot will\b",
    r"\bit is slowing\b",
    r"\bit is stopping\b",
    r"\bit is replanning\b",
    r"\bel robot\b",
    r"\beste robot\b",
    r"\bel sistema\b",
)

DIRECT_HUMAN_REFERENCES = (
    r"\byou\b",
    r"\byour\b",
    r"\byou're\b",
    r"\byou are\b",
    r"\busted\b",
    r"\btu\b",
    r"\btú\b",
    r"\bte encuentras\b",
)

THIRD_PERSON_HUMAN_REFERENCES = (
    r"\ba person\b",
    r"\bthe person\b",
    r"\bsomeone\b",
    r"\bpeople\b",
    r"\buna persona\b",
    r"\bla persona\b",
    r"\balguien\b",
)


def has_invalid_speech_perspective(text: str) -> bool:
    """
    Reject explanations that describe the robot externally or describe the
    detected interaction partner as an uninvolved third person.
    """

    normalized = " ".join(
        text.strip().lower().split()
    )

    patterns = (
        FORBIDDEN_ROBOT_REFERENCES
        + THIRD_PERSON_HUMAN_REFERENCES
    )

    return any(
        re.search(pattern, normalized)
        for pattern in patterns
    )



class GenerativeLLMProvider(LLMProvider):
    """
    Base class for providers backed by a generative language model.

    Concrete providers remain responsible only for:
        - client initialization,
        - availability checks,
        - API/local inference,
        - conversion to LLMExplanationResult.

    Prompt semantics remain centralized here.
    """

    # =========================================================================
    # PROMPT CONSTRUCTION
    # =========================================================================

    @staticmethod
    def build_prompt(
        context: ExplanationContext,
    ) -> str:
        """Build speech instructions with an explicit assistance exception."""

        language = {
            "en": "English",
            "fr": "French",
            "es": "Spanish",
        }.get(
            str(context.language or "").strip().lower(),
            "Spanish",
        )

        reason = str(context.reason_code or "").strip().upper()
        assistance = reason == "PASSAGE_ASSISTANCE"

        if assistance and not (
            str(context.action or "").strip().upper() == "WAIT"
            and str(context.navigation_goal_id or "").strip()
            and str(context.person_id or "").strip()
            and not context.navigation_active
            and context.social_conflict_active
            and not context.requires_replan
        ):
            raise ValueError(
                "Invalid PASSAGE_ASSISTANCE context."
            )

        if assistance:
            instructions = (
                "The checked candidate trajectories did not provide "
                "an admissible passage. I am stopped. "
                "Politely ask whether you could move slightly, "
                "if convenient, to provide more room. "
                "Make this optional, never an order. "
                "Say that I will remain stopped. "
                "Do not ask for passage permission again. "
                "Do not claim that all possible routes are blocked "
                "or that you caused the failure. "
                "Do not promise to move after you comply; "
                "new observation and trajectory validation "
                "are still required."
            )

        else:
            instructions = (
                "Explain only the supplied decision and "
                "immediate intention. "
                "For WAIT, describe remaining stopped and "
                "preserving personal space. "
                "For REPLAN, describe looking for another path. "
                "For SLOW_DOWN, describe slowing down. "
                "For OBSERVE, acknowledge your presence without "
                "claiming navigation was interrupted. "
                "Do not ask you to move or invent a request "
                "for assistance."
            )

        return (
            "You produce a short spoken explanation "
            "for a service robot.\n"
            f"Language: {language}\n"
            f"Action: {context.action}\n"
            f"Reason: {context.reason_code}\n"
            f"Social zone type: {context.zone_type}\n"
            f"Tracking state: {context.tracking_state}\n"
            f"Robot intention: {context.robot_intention}\n"
            f"Explanation objective: {context.explanation_goal}\n\n"
            f"Situation-specific instructions: {instructions}\n\n"
            "Speak in the first person as the robot and address "
            "the person directly and respectfully, using the "
            "selected language. "
            "Never refer to yourself as 'the robot' or 'the system', "
            "or to the person as an uninvolved third party. "
            "Do not choose or modify navigation policy. "
            "Human agreement does not authorize movement. "
            "Missing observation is not clearance. "
            "Do not invent sensor facts, collision risk, "
            "or human intentions. "
            "Do not blame the person or describe them as an obstacle. "
            "Do not include numerical measurements, coordinates "
            "or identifiers. "
            "Return only speech text, at most two short sentences "
            "and 30 words."
        )
    
    @staticmethod
    def build_prompt_old(
        context: ExplanationContext,
    ) -> str:
        """
        Build a reusable semantic explanation prompt.

        Inputs:
            context:
                Deterministic ExplanationContext produced upstream.

        Processing:
            Preserves semantic navigation facts while deliberately excluding
            continuously changing values such as:
                - decision_id,
                - person_id,
                - exact position,
                - exact distance,
                - exact speed,
                - occupancy ratio.

        Output:
            Provider-independent natural-language prompt.

        Called by:
            OpenAIProvider.generate().
            GeminiProvider.generate().
            LocalProvider.generate().

        Side effects:
            None.
        """

        language_map = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
        }

        language_name = language_map.get(
            str(context.language or "").strip().lower(),
            "Spanish",
        )

        return (
            "You are the explanation component of a service robot.\n\n"

            "Your task is to explain an already determined robot "
            "navigation decision.\n"

            "You must not choose, modify, question, or replace the "
            "navigation action.\n"

            "You must not invent sensor information, environmental facts, "
            "or robot intentions.\n\n"

            "Deterministic navigation context:\n"

            f"- Language: {language_name}\n"
            f"- Action: {context.action}\n"
            f"- Reason: {context.reason_code}\n"
            f"- Social zone type: {context.zone_type}\n"
            f"- Tracking state: {context.tracking_state}\n"
            f"- Robot intention: {context.robot_intention}\n\n"

            "Explanation objective:\n"
            f"{context.explanation_goal}\n\n"

            "Requirements:\n"
            "- Speak as the robot using the first-person singular.\n"
            "- Refer to the robot only with 'I', 'me', or 'my'.\n"
            "- Never call the robot 'the robot', 'this robot', "
            "'the system', 'it', 'he', or 'she'.\n"

            "- When explaining a social interaction caused by a detected person, "
            "address that person directly and respectfully using 'you' or 'your'.\n"
            "- Speak to the person involved in the interaction, not to an external "
            "observer describing that person.\n"
            "- Prefer natural phrases such as 'I can see you' or "
            "'I noticed you nearby' instead of 'I detect a person'.\n"
            "- Do not blame, command, criticize, or make assumptions about the "
            "person's intentions.\n"
            "- Do not claim that the person is blocking the robot unless that fact "
            "is explicitly present in the deterministic context.\n"

            "- Describe only the deterministic action and social situation given "
            "in the context.\n"

            "- A social navigation conflict does not necessarily mean that the person "
            "is physically blocking the entire visible passage.\n"
            "- The available navigation clearance can be insufficient even when the "
            "passage appears large enough to a person.\n"
            
            "- Use a human-first perspective for social navigation conflicts.\n"
            "- Prioritize the person's safety, comfort, personal space, and freedom "
            "of movement in the explanation.\n"
            "- Explain how I adapt my behavior for the person rather than presenting "
            "the person as a navigation obstacle.\n"
            "- When the action is WAIT, explain that I stop or wait to give the person "
            "enough space and avoid getting too close or making contact.\n"
            "- When the action is REPLAN, explain that I preserve the person's space "
            "and look for another path instead of expecting the person to move.\n"
            "- A passage that appears visually open may still provide insufficient "
            "clearance for safe navigation because my navigation must account for my "
            "physical footprint and safety margins.\n"
            "- Express this through my protective behavior toward the person, not as "
            "a complaint that I need more space.\n"
            "- Never describe the person as an obstacle, inconvenience, or problem.\n"
            "- Never imply that the person should move so that I can continue.\n"
            "- Never blame the person for insufficient navigation clearance.\n"

            "- If the action is REPLAN, describe changing or requesting a path; "
            "do not claim that the robot is slowing down.\n"
            "- If the action is SLOW_DOWN, say that I am slowing down.\n"
            "- Do not mention exact distances, coordinates, speeds, "
            "occupancy ratios, person identifiers, or decision identifiers.\n"
            "- Keep the spoken explanation very short.\n"
            "- Use one or two short sentences and never exceed 30 words.\n"
            "- State only the immediate reason and what I am doing or will do.\n"
            "- Do not repeat the same information in different words.\n"
            "- Do not add background explanation, commentary, or a conclusion.\n"
            "- Produce concise, speech-friendly language.\n"
            "- Return only the explanation text.\n\n"

            "- Explain the safety purpose of the action when appropriate.\n"
            "- Natural safety explanations include giving the person enough space, "
            "avoiding getting too close, or avoiding accidental contact.\n"
            "- Do not exaggerate the risk: do not claim that a collision would have "
            "occurred unless that fact is explicitly supplied by the deterministic context.\n"

            "Correct examples:\n"
            "- I can see you nearby. I have stopped to give you enough space and avoid getting too close.\n"
            "- I will give you space and look for another path.\n"
            "- I can see you nearby. I will keep a safe distance if I start moving.\n\n"

            "Incorrect examples:\n"
            "- The robot is slowing down because the person is close.\n"
            "- I detect a person close to my path.\n"
            "- I need more space, so I have stopped.\n"
            "- You are blocking my path.\n"
            "- You are too close, so I cannot pass.\n"
            "- You need to move out of my way.\n"

        )
