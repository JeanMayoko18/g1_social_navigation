"""
Deterministic multilingual explanations for social collaboration events.

Responsibilities:
    - Acknowledge a confirmed person/social-zone intrusion when no
      navigation goal is executing.
    - Explain that navigation is being held because of a confirmed
      social intrusion.
    - Explain a replanning request during active navigation.
    - Explain conservative replanning after repeated human silence.
    - Reject unsupported or inconsistent speech contexts.

This provider generates semantic speech text only.

Dynamic measurements remain available in ExplanationContext for
diagnostics and analysis and are not embedded in the reusable
deterministic semantic templates.

This module evaluates only the supplied ExplanationContext snapshot.
Live navigation-state and observation-freshness checks belong to the
ExplanationNode caller.
"""

import math

from g1_social_msgs.msg import ExplanationContext

from g1_robot_explainability.llm_provider import (
    LLMExplanationResult,
    LLMProvider,
)


# =============================================================================
# SUPPORTED DECISION SEMANTICS
# =============================================================================

ACTION_OBSERVE = "OBSERVE"
ACTION_WAIT = "WAIT"
ACTION_REPLAN = "REPLAN"

REASON_SOCIAL_ZONE_INTRUSION = "SOCIAL_ZONE_INTRUSION"
REASON_NO_HUMAN_RESPONSE = "NO_HUMAN_RESPONSE"
REASON_DIALOGUE_SPEECH_UNAVAILABLE = "DIALOGUE_SPEECH_UNAVAILABLE"
REASON_SOCIAL_LOSS_TIMEOUT = "SOCIAL_LOSS_TIMEOUT"


# =============================================================================
# DETERMINISTIC PROVIDER
# =============================================================================

class DeterministicFallbackProvider(LLMProvider):
    """
    Generate deterministic English, French or Spanish social explanations.

    The provider does not choose navigation policy. It only verbalizes the
    decision already represented by ExplanationContext.
    """

    provider_name = "deterministic"

    def __init__(self) -> None:
        super().__init__(
            model_name="template_v2_navigation_context"
        )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def generate(
        self,
        context: ExplanationContext,
    ) -> LLMExplanationResult:
        """Generate speech for a validated deterministic context."""

        language = self._resolve_language(context.language)
        reason = str(context.reason_code or "").strip().upper()

        if reason == "PASSAGE_ASSISTANCE":
            texts = {
                "en": (
                    "I haven't found a safe passage. "
                    "If convenient, could you move slightly "
                    "to give me more room? I'll stay stopped."
                ),
                "fr": (
                    "Je n'ai pas trouvé de passage sûr. "
                    "Si vous le pouvez, pourriez-vous vous décaler "
                    "légèrement ? Je reste arrêté."
                ),
                "es": (
                    "No he encontrado un paso seguro. "
                    "Si puedes, ¿podrías apartarte un poco "
                    "para dejarme más espacio? Seguiré detenido."
                ),
            }

            text = (
                texts[language]
                if self.is_supported_context(context)
                else ""
            )

        else:
            text = self._build_text(
                context=context,
                language=language,
            )

        return LLMExplanationResult(
            success=bool(text),
            text=text,
            provider=self.provider_name,
            model=self.model_name,
            language=language,
            error=(
                None
                if text
                else (
                    "Unsupported or inconsistent "
                    "social speech context."
                )
            ),
            used_fallback=True,
        )
    
    # =========================================================================
    # CONTEXT CLASSIFICATION
    # =========================================================================

    def _build_text(
        self,
        context: ExplanationContext,
        language: str,
    ) -> str:
        """Select speech for the exact supported semantic context."""

        if not self.is_supported_context(context):
            return ""

        language = self._resolve_language(language)

        action = str(context.action or "").strip().upper()
        reason = str(context.reason_code or "").strip().upper()

        if reason == REASON_NO_HUMAN_RESPONSE:
            return self._build_no_human_response_explanation(
                language
            )

        if reason == REASON_DIALOGUE_SPEECH_UNAVAILABLE:
            return self._build_speech_unavailable_explanation(
                language
            )

        if reason == "SOCIAL_CLEAR_BEFORE_ANNOUNCEMENT":
            if action == "WAIT":
                texts = {
                    "en": (
                        "I am staying still after detecting you nearby. "
                        "There is now more space; I will check a new "
                        "route before moving."
                    ),
                    "fr": (
                        "Je reste arrêté après vous avoir détecté à proximité. "
                        "Il y a maintenant plus d'espace ; je vérifierai "
                        "un nouveau trajet avant de bouger."
                    ),
                    "es": (
                        "Sigo detenido tras detectarte cerca. "
                        "Ahora hay más espacio; comprobaré una nueva "
                        "ruta antes de moverme."
                    ),
                }
            else:
                texts = {
                    "en": (
                        "I noticed you nearby. "
                        "There is now more space between us; "
                        "I am remaining idle."
                    ),
                    "fr": (
                        "Je vous avais détecté à proximité. "
                        "Il y a maintenant plus d'espace entre nous ; "
                        "je reste immobile."
                    ),
                    "es": (
                        "Te había detectado cerca. "
                        "Ahora hay más espacio entre nosotros; "
                        "permanezco inmóvil."
                    ),
                }

            return texts[language]
        
        if reason == REASON_SOCIAL_LOSS_TIMEOUT:
            texts = {
                "fr": (
                    "Je ne vous vois plus pour le moment. "
                    "Je vais vérifier si je peux continuer ma route "
                    "avec un passage sûr."
                ),
                "en": (
                    "I can no longer see you for the moment. "
                    "I will check whether I can continue my route "
                    "using a safe passage."
                ),
                "es": (
                    "Por el momento ya no puedo verte. "
                    "Voy a comprobar si puedo continuar mi recorrido "
                    "por un paso seguro."
                ),
            }

            return texts[language]

        if reason == "SOCIAL_OBSERVATION_LOST":
            texts = {
                "en": (
                    "I noticed you nearby, but I can no longer "
                    "confirm your position. I am staying still."
                ),
                "fr": (
                    "Je vous avais détecté à proximité, mais je ne peux "
                    "plus confirmer votre position. Je reste immobile."
                ),
                "es": (
                    "Te había detectado cerca, pero ya no puedo "
                    "confirmar tu posición. Permanezco inmóvil."
                ),
            }

            return texts[language]

        if action == "OBSERVE":
            texts = {
                "en": (
                    "I noticed you nearby. "
                    "I am remaining idle and will take your "
                    "presence into account when navigating."
                ),
                "fr": (
                    "Je vous ai détecté à proximité. "
                    "Je reste immobile et prendrai votre présence "
                    "en compte lors de mes déplacements."
                ),
                "es": (
                    "Te he detectado cerca. "
                    "Permanezco inmóvil y tendré en cuenta "
                    "tu presencia al desplazarme."
                ),
            }

            return texts[language]

        if action == "WAIT":
            texts = {
                "en": (
                    "I noticed you nearby. "
                    "I am staying still to give you space "
                    "and avoid getting too close."
                ),
                "fr": (
                    "Je vous ai détecté à proximité. "
                    "Je reste arrêté pour vous laisser de l'espace "
                    "et éviter de trop m'approcher."
                ),
                "es": (
                    "Te he detectado cerca. "
                    "Sigo detenido para dejarte espacio "
                    "y evitar acercarme demasiado."
                ),
            }

            return texts[language]

        return self._build_replan_explanation(language)
    
    @staticmethod
    def _has_confirmed_intrusion(
        context: ExplanationContext,
    ) -> bool:
        """
        Validate a confirmed social intrusion produced upstream.

        The social-navigation decision node remains authoritative for
        conflict detection and hysteresis.

        This provider validates only the factual context fields required
        to verbalize that already-confirmed event.
        """

        reason = str(
            context.reason_code or ""
        ).strip().upper()

        if (
            not context.social_conflict_active
            or reason != REASON_SOCIAL_ZONE_INTRUSION
        ):
            return False

        try:
            distance = float(
                context.person_distance
            )

            social_radius = float(
                context.social_radius
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            return False

        return (
            math.isfinite(distance)
            and math.isfinite(social_radius)
            and distance >= 0.0
            and social_radius > 0.0
        )

    # =========================================================================
    # SPEECH TEMPLATES
    # =========================================================================

    @staticmethod
    def _build_presence_explanation(
        language: str,
    ) -> str:
        """
        Acknowledge a nearby person without claiming that navigation stopped.
        """

        if language == "en":
            return (
                "I can see you nearby. "
                "I will keep a comfortable distance."
            )

        if language == "fr":
            return (
                "Je vous vois à proximité. "
                "Je vais garder une distance confortable."
            )

        return (
            "Puedo verte cerca. "
            "Mantendré una distancia cómoda."
        )

    @staticmethod
    def _build_wait_explanation(
        language: str,
    ) -> str:
        """
        Explain that navigation is being held to preserve personal space.
        """

        if language == "en":
            return (
                "I can see you nearby. "
                "I have stopped to give you enough space "
                "and avoid getting too close."
            )

        if language == "fr":
            return (
                "Je vous vois à proximité. "
                "Je me suis arrêté pour vous laisser suffisamment "
                "d'espace et éviter de trop m'approcher."
            )

        return (
            "Puedo verte cerca. "
            "Me he detenido para dejarte suficiente espacio "
            "y evitar acercarme demasiado."
        )

    @staticmethod
    def _build_replan_explanation(
        language: str,
    ) -> str:
        """
        Explain that another socially compliant path is being requested.
        """

        if language == "en":
            return (
                "I will give you space "
                "and look for another path."
            )

        if language == "fr":
            return (
                "Je vais vous laisser de l'espace "
                "et chercher un autre chemin."
            )

        return (
            "Te dejaré espacio "
            "y buscaré otra ruta."
        )

    @staticmethod
    def is_supported_context(
        context: ExplanationContext,
    ) -> bool:
        """Validate the semantic contract supplied by collaboration."""

        action = str(context.action or "").strip().upper()
        reason = str(context.reason_code or "").strip().upper()

        has_goal = bool(
            str(context.navigation_goal_id or "").strip()
        )
        active = bool(context.navigation_active)
        conflict = bool(context.social_conflict_active)
        replan = bool(context.requires_replan)

        if action == "WAIT":
            if not has_goal or active:
                return False

        elif action == "OBSERVE":
            if has_goal or active or replan:
                return False

        elif action == "REPLAN":
            if not has_goal or not replan:
                return False

        else:
            return False

        if reason == "PASSAGE_ASSISTANCE":
            return bool(
                action == "WAIT"
                and conflict
                and not replan
                and str(context.person_id or "").strip()
            )

        if reason == "SOCIAL_CLEAR_BEFORE_ANNOUNCEMENT":
            return (
                not conflict
                and (
                    action == "OBSERVE"
                    or (action == "WAIT" and replan)
                )
            )

        if reason == REASON_SOCIAL_LOSS_TIMEOUT:
            return (
                action == "WAIT"
                and has_goal
                and not active
                and conflict
                and replan
            )

        if reason == "SOCIAL_OBSERVATION_LOST":
            return (
                conflict
                and action in {"OBSERVE", "WAIT"}
                and not replan
            )

        if reason == REASON_SOCIAL_ZONE_INTRUSION:
            if not conflict:
                return False

            if action in {"OBSERVE", "WAIT"}:
                return not replan

            return action == "REPLAN" and replan

        if reason in {
            REASON_NO_HUMAN_RESPONSE,
            REASON_DIALOGUE_SPEECH_UNAVAILABLE,
        }:
            return (
                action == "WAIT"
                and conflict
                and replan
            )

        return False
    
    @staticmethod
    def _build_speech_unavailable_explanation(
        language: str,
    ) -> str:
        """
        Explain conservative adaptation when the robot cannot continue
        the spoken dialogue.

        This sentence must not claim that the human remained silent.
        """

        if language == "en":
            return (
                "I cannot continue the spoken exchange. "
                "I will look for another safe way to continue."
            )

        if language == "fr":
            return (
                "Je ne peux pas poursuivre l'échange vocal. "
                "Je vais chercher une autre manière sûre de continuer."
            )

        return (
            "No puedo continuar el intercambio de voz. "
            "Buscaré otra forma segura de continuar."
        )

    @staticmethod
    def _build_no_human_response_explanation(
        language: str,
    ) -> str:
        """
        Explain conservative replanning after repeated human silence.

        The robot is still stopped while this sentence is spoken.
        Replanning is the selected NEXT action.
        """

        if language == "en":
            return (
                "I did not receive a response. "
                "I will look for another safe way to continue."
            )

        if language == "fr":
            return (
                "Je n'ai pas reçu de réponse. "
                "Je vais chercher une autre manière sûre de continuer."
            )

        return (
            "No he recibido una respuesta. "
            "Buscaré otra forma segura de continuar."
        )

    # =========================================================================
    # LANGUAGE RESOLUTION
    # =========================================================================

    @staticmethod
    def _resolve_language(
        language: str,
    ) -> str:
        """
        Resolve supported languages, preserving Spanish as fallback.
        """

        normalized = str(
            language or ""
        ).strip().lower()

        if normalized in {
            "en",
            "es",
            "fr",
        }:
            return normalized

        return "es"