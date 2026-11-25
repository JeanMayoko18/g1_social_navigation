
"""Mode C/D interaction: bounded HRI dialogue and autonomous route recovery."""

import json
import math
import time
import uuid
from .direction_semantics import interpret_passage_direction
from std_msgs.msg import String
from g1_social_msgs.msg import (
    DirectionalConstraintRequest,
)


from g1_social_common.collaboration_states import (
    STATE_NAVIGATING,
    STATE_REPLANNING,
    STATE_WAITING,
    STATE_WAITING_FOR_CONFIRMATION,
    STATE_WAITING_FOR_HUMAN,
)
from g1_social_common.dialogue_texts import get_dialogue_text


class InteractionMixin:
    """Coordinate social interaction and autonomous Nav2 handoff."""

    # --------------------------------------------------
    # SOCIAL RESOLUTION
    # --------------------------------------------------
    def _restore_retained_same_person_consent(self):
        """
        Restore an already confirmed human permission.

        Reuse requires the exact same public navigation goal,
        person ID and tracking session.

        A new social episode may have a different identity,
        but no new human permission is inferred.

        This method does not authorize physical motion.
        """
        import copy

        retained = getattr(
            self,
            "_retained_same_person_consent",
            None,
        )

        if not isinstance(retained, dict):
            return False

        scope = self._social_scope_key()

        if (
            not scope
            or len(scope) != 5
            or not all(scope)
            or self._scope != scope
            or retained.get("scope") != scope
            or scope[0] != self.active_navigation_goal_id
            or self.interaction != "dialogue"
            or self._denied_scope
            or self.pending_terminal_stop is not None
        ):
            return False

        requested = self._normalize_direction(
            retained.get("direction")
        )

        if requested not in {"NONE", "LEFT", "RIGHT"}:
            return False

        finalization = retained.get("finalization")
        restored_finalization = None

        if requested in {"LEFT", "RIGHT"}:
            if not isinstance(finalization, dict):
                return False

            candidate = finalization.get("candidate")

            if not isinstance(candidate, dict):
                return False

            reference = str(
                candidate.get("reference") or ""
            ).strip().upper()

            resolved = str(
                candidate.get("resolved_robot_direction") or ""
            ).strip().upper()

            original_direction = str(
                candidate.get("direction") or ""
            ).strip().upper()

            interaction_id = str(
                retained.get("interaction_id") or ""
            ).strip()

            if reference == "PERSON":
                expected = (
                    "RIGHT" if requested == "LEFT"
                    else "LEFT"
                )
                expected_relation = "FACE_TO_FACE_ASSUMED"

            elif reference == "ROBOT":
                expected = requested
                expected_relation = "ROBOT_REFERENCE"

            else:
                return False

            if not (
                interaction_id
                and finalization.get("phase") == "ANNOUNCED"
                and candidate.get(
                    "final_confirmation_received"
                ) is True
                and original_direction == requested
                and resolved == expected
                and candidate.get("spatial_relation")
                    == expected_relation
                and candidate.get("interaction_id")
                    == interaction_id
                and finalization.get("interaction_id")
                    == interaction_id
                and candidate.get(
                    "final_confirmation_request_id"
                ) == finalization.get("request_id")
            ):
                return False

            restored_finalization = copy.deepcopy(
                finalization
            )

            # The human decision is retained, but its new
            # social episode receives a new canonical scope.
            restored_finalization["scope"] = scope
            restored_finalization["candidate"]["scope"] = scope

        # The original human permission is restored only
        # after every semantic and identity check succeeds.
        self.confirmed_human_intent = "ALLOW_PASSAGE"
        self.confirmed_human_direction = requested
        self.dialogue_outcome = "ALLOW_CONFIRMED"
        self.authorized_episode_key = scope

        if restored_finalization is not None:
            self._directional_finalization = (
                restored_finalization
            )
            self._directional_announcement_complete = True
            self._direction_semantic_ready = True
            self._directional_confirmed_scope = scope
            self._directional_confirmed_interaction_id = (
                str(retained["interaction_id"])
            )

        else:
            # A simple YES grants passage without
            # imposing a directional preference.
            self._directional_finalization = None
            self._directional_announcement_complete = False
            self._direction_semantic_ready = False
            self._directional_confirmed_scope = None
            self._directional_confirmed_interaction_id = ""

        self._retained_same_person_consent = None

        self.get_logger().info(
            "[SOCIAL CONSENT] Previous permission restored | "
            f"goal={scope[0]} | "
            f"person={scope[4]} | "
            f"tracking_session={scope[3]} | "
            f"human_direction={requested} | "
            "dialogue_repeated=False | "
            "motion_authorized=False"
        )

        return True

    def _enter_announcement(self):
        """
        Avoid repeating the announcement and dialogue
        when a valid permission from the same person
        has been retained across a social episode.
        """
        self._restore_retained_same_person_consent()

        reuse_permission = (
            self._permission_applies_to_current_episode()
        )

        reuse_autonomous = bool(
            self.interaction == "none"
            and self._scope
            and getattr(
                self,
                "_reuse_autonomous_scope",
                None,
            ) == self._scope
        )

        if (
            self._scope
            and not self._released
            and not self._denied_scope
            and self.pending_terminal_stop is None
            and (reuse_permission or reuse_autonomous)
        ):
            self.announcement_context_key = None
            self.current_announcement_action = ""

            self._transition_to(
                STATE_WAITING,
                reason="EXISTING_SOCIAL_RESOLUTION_REUSED",
            )

            direction = (
                self.confirmed_human_direction
                if reuse_permission
                else "NONE"
            )

            self._release_navigation(direction)
            return

        return super()._enter_announcement()
    
    def _revoke_optional_directional_mask(self, reason):
        """
        Revoke the exact directional mask.

        REVOKE is complete when:
        - no mask was ever published for this request, or
        - neutralization was observed in both Nav2 costmaps.

        Do not repeat REVOKE after a matching terminal status.
        """
        import copy

        request = getattr(
            self,
            "_directional_request",
            None,
        )

        if request is None:
            self._directional_status = None
            self._directional_revoke_pending = False
            return True

        status = getattr(
            self,
            "_directional_status",
            None,
        )

        identity_fields = (
            "request_id",
            "navigation_goal_id",
            "interaction_id",
            "person_id",
            "tracking_session_id",
            "identity_generation",
            "directional_constraint",
        )

        exact_status = bool(
            status is not None
            and all(
                getattr(status, field)
                == getattr(request, field)
                for field in identity_fields
            )
        )

        revoked = bool(
            exact_status
            and str(status.state).strip().upper()
                == "REVOKED"
            and (
                str(status.reason).strip().upper()
                    == "NO_DIRECTIONAL_MASK_WAS_PUBLISHED"
                or (
                    bool(status.global_costmap_confirmed)
                    and bool(status.local_costmap_confirmed)
                )
            )
        )

        # IMPORTANT: consume the confirmed result before
        # attempting another publication.
        #
        # This also handles the case where a matching
        # REVOKED status arrived before this method ran.
        if revoked:
            self.get_logger().info(
                "[DIRECTIONAL MASK] Revocation completed | "
                f"goal={request.navigation_goal_id} | "
                f"request={request.request_id} | "
                f"reason={status.reason}"
            )

            self._directional_request = None
            self._directional_status = None
            self._directional_revoke_pending = False
            self._directional_watch_request_id = ""
            self._directional_last_renewal_at = None

            return True

        publisher = getattr(
            self,
            "directional_constraint_pub",
            None,
        )

        if publisher is None:
            self.get_logger().error(
                "[DIRECTIONAL MASK] REVOKE unavailable | "
                "publisher_missing=True"
            )
            return False

        now = time.monotonic()

        if (
            getattr(
                self,
                "_directional_revoke_pending",
                False,
            )
            and now - getattr(
                self,
                "_directional_revoke_last_sent",
                0.0,
            ) < 1.0
        ):
            return False

        try:
            revoke = copy.deepcopy(request)
            revoke.action = "REVOKE"
            revoke.header.stamp = (
                self.get_clock().now().to_msg()
            )

            publisher.publish(revoke)

        except Exception as exc:
            self.get_logger().error(
                "[DIRECTIONAL MASK] REVOKE failed | "
                f"reason={reason} | error={exc}"
            )
            return False

        self._directional_revoke_pending = True
        self._directional_revoke_last_sent = now

        return False
    
    def _use_unconstrained_nav2(self, reason):
        """Select autonomous routing while retaining confirmed preference."""
        self._revoke_optional_directional_mask(reason)

        # Effective routing and remembered human preference are distinct.
        self._direction = "NONE"
        self._episode_mode = "C"
        self._directional_retry_pending = False

        if self._nav_phase == "DIRECT_NAV2_WAIT_MASK":
            self._nav_phase = "DIRECT_NAV2_WAIT_STOP"

        self.get_logger().warning(
            "[DIRECT NAV2] Optional direction unavailable | "
            f"reason={reason} | "
            f"remembered_direction={self.confirmed_human_direction} | "
            "effective_direction=NONE | "
            "route_selection=NAV2"
        )

        return True
        
    def _enter_social_resolution(
        self,
        *,
        speech_completed=False,
    ):
        """
        Reuse confirmed permission or start a new HRI turn.

        A different episode ID does not force a repeated
        dialogue for the same authorized person and
        tracking session.
        """
        if (
            not self._scope
            or self._released
            or self._denied_scope
            or self.pending_terminal_stop is not None
        ):
            return

        self.announcement_context_key = None
        self.current_announcement_action = ""

        self._restore_retained_same_person_consent()

        if self._permission_applies_to_current_episode():
            self._release_navigation(
                self.confirmed_human_direction
            )
            return

        if self._episode_mode == "C":
            self._release_navigation("NONE")
            return

        if not speech_completed:
            self._begin_autonomous_readaptation_announcement(
                "INITIAL_SPEECH_UNAVAILABLE"
            )
            return

        self.dialogue_response_timeout_sec = min(
            15.0,
            float(self.dialogue_response_timeout_sec),
        )

        self.dialogue_retry_count = 0
        self.dialogue_speech_retry_count = 0

        self._hri_retry_after_close = None
        self._hri_retry_notice = "MISSED"
        self._hri_nav_fallback_active = False

        self._start_human_initiative_window()

    def _start_human_initiative_window(self):
        """Offer the person an initial turn before clarification."""
        if (
            not self._scope
            or self._released
            or self._denied_scope
            or self.pending_terminal_stop is not None
        ):
            return

        if (
            self.pending_hri_close is not None
            or self.pending_dialogue_request_id
            or self.dialogue_audio_recovery_pending
            or getattr(self, "_hri_detached_close", None) is not None
        ):
            self._begin_autonomous_readaptation_announcement(
                "HRI_TURN_UNAVAILABLE"
            )
            return
        
        if getattr(
            self,
            "demo_robot_only_single_yes",
            False,
        ):
            self.get_logger().info(
                "[HRI DEMO] Skipping human initiative window | "
                "next=ROBOT_PASSAGE_QUESTION"
            )

            return self._request_passage_clarification()
        
        request = self._new_dialogue_request()

        if request is None:
            self._begin_autonomous_readaptation_announcement(
                "HRI_TURN_UNAVAILABLE"
            )
            return


        request.request_type = "HUMAN_INITIATIVE_WINDOW"
        request.context = "SOCIAL_CONFLICT"
        request.question = ""
        request.intent_to_confirm = ""
        request.direction_to_confirm = "NONE"
        request.robot_initiated = False
        request.response_timeout = float(
            self.human_initiative_timeout_sec
        )

        self.active_dialogue_request_type = request.request_type
        self.waiting_for_human_initiative = True
        self.dialogue_outcome = "PENDING"
        self.dialogue_started_at = time.monotonic()

        self._transition_to(
            STATE_WAITING_FOR_HUMAN,
            reason="HUMAN_INITIATIVE_WINDOW",
        )

        self.dialogue_request_pub.publish(request)
    
    def _directional_identity_callback(self, msg):
        """Retry directional release when a fresh identity arrives."""
        super()._directional_identity_callback(msg)

        if (
            getattr(self, "_directional_retry_pending", False)
            and self.fsm_state == STATE_WAITING
            and self.confirmed_human_intent == "ALLOW_PASSAGE"
            and self.dialogue_outcome == "ALLOW_CONFIRMED"
        ):
            self._release_navigation(
                self.confirmed_human_direction
            )
            
    def _request_passage_clarification(self):
        """Ask for passage clarification using the existing HRI policy."""
        if (
            not self._scope
            or self._released
            or self._denied_scope
        ):
            return

        if (
            self.pending_hri_close is not None
            or self.pending_dialogue_request_id
            or getattr(self, "_hri_detached_close", None) is not None
        ):
            self._begin_autonomous_readaptation_announcement(
                "HRI_TURN_UNAVAILABLE"
            )
            return

        request = self._new_dialogue_request()

        if request is None:
            self._begin_autonomous_readaptation_announcement(
                "HRI_TURN_UNAVAILABLE"
            )
            return

        request.request_type = "PASSAGE_CLARIFICATION"
        request.context = "SOCIAL_CONFLICT"
        request.question = get_dialogue_text(
            "passage",
            self.language,
        )

        request.intent_to_confirm = ""
        request.direction_to_confirm = "NONE"
        request.robot_initiated = True
        request.response_timeout = min(
            15.0,
            float(self.dialogue_response_timeout_sec),
        )

        self.active_dialogue_request_type = request.request_type
        self.waiting_for_human_initiative = False
        self.dialogue_outcome = "PENDING"
        self.dialogue_started_at = time.monotonic()

        self._transition_to(
            STATE_WAITING_FOR_HUMAN,
            reason="PASSAGE_CLARIFICATION_REQUESTED",
        )

        self.dialogue_request_pub.publish(request)

    # --------------------------------------------------
    # HUMAN INTENT PROCESSING
    # --------------------------------------------------

    def _human_intent_callback(self, msg):
        """
        Route the exact current HRI response.

        Human silence -> announce, then Mode C.
        Audio failure -> technical announcement, then Mode C.
        Explicit NO -> existing 180-second refusal procedure.
        Direction correction -> existing directional finalization.

        No direct Nav2 START is issued here.
        """
        import re

        if not self._scope or self._released:
            return

        source = str(
            msg.source or ""
        ).strip().upper()

        intent = str(
            msg.intent or ""
        ).strip().upper()

        interaction_id = str(
            msg.interaction_id or ""
        ).strip()

        request_id = str(
            msg.request_id or ""
        ).strip()

        current_turn = bool(
            self.active_interaction_id
            and self.pending_dialogue_request_id
            and interaction_id == self.active_interaction_id
            and request_id == self.pending_dialogue_request_id
        )

        if not current_turn:
            return

        if self.pending_hri_close is not None:
            return

        technical_sources = {
            "ROBOT_SPEECH_FAILED",
            "ROBOT_SPEECH_TIMEOUT",
            "DIALOGUE_REQUEST_REJECTED",
            "AUDIO_TERMINATION_UNCONFIRMED",
        }

        # A real human-response timeout is not a refusal
        # and does not require another question.
        if source in {
            "HUMAN_RESPONSE_TIMEOUT",
            "TIMEOUT",
        }:
            self._hri_retry_notice = "MISSED"
            self._begin_autonomous_readaptation_announcement(
                "NO_HUMAN_RESPONSE"
            )
            return

        if source in technical_sources:
            self._begin_autonomous_readaptation_announcement(
                "HRI_TECHNICAL_FAILURE"
            )
            return

        directional_stages = {
            "DIRECTION_REFERENCE_CLARIFICATION",
            "DIRECTION_FINAL_CONFIRMATION",
        }

        stage = str(
            self.active_dialogue_request_type or ""
        ).strip().upper()

        # In demo mode, an explicit correction received during
        # final confirmation must not become REJECT.
        #
        # The existing directional handler already handles
        # corrections during reference clarification.
        if (
            getattr(
                self,
                "demo_robot_only_single_yes",
                False,
            )
            and stage == "DIRECTION_FINAL_CONFIRMATION"
        ):
            spoken = re.sub(
                r"[,.!?;:]+",
                " ",
                str(msg.utterance or "").lower(),
            )

            spoken = " ".join(spoken.split())

            match = re.fullmatch(
                r"no\s+(?:on\s+)?your\s+(left|right)",
                spoken,
            )

            if match is not None:
                candidate = getattr(
                    self,
                    "_direction_semantic_candidate",
                    None,
                )

                corrected_direction = (
                    match.group(1).upper()
                )

                passage = interpret_passage_direction(
                    corrected_direction,
                    msg.utterance,
                )

                if (
                    isinstance(candidate, dict)
                    and candidate.get("scope") == self._scope
                    and candidate.get("interaction_id")
                        == interaction_id
                    and passage is not None
                    and passage.reference == "ROBOT"
                ):
                    candidate["direction"] = (
                        corrected_direction
                    )
                    candidate["reference"] = "ROBOT"
                    candidate["explicit_reference"] = True

                    candidate[
                        "resolved_robot_direction"
                    ] = "NONE"

                    candidate[
                        "final_confirmation_received"
                    ] = False

                    self.pending_human_direction = (
                        corrected_direction
                    )

                    self._direction_semantic_ready = False
                    self.dialogue_retry_count = 0

                    if self._finalize_demo_clarified_direction(
                        candidate
                    ):
                        return

                    if self.pending_hri_close is not None:
                        return

                    self._publish_wait(
                        "DIRECTION_CORRECTION_NOT_FINALIZED"
                    )
                    return

        if stage in directional_stages:
            return self._handle_directional_dialogue_response(
                msg
            )

        if self._spatial_response_is_ambiguous(msg):
            self.get_logger().warning(
                "[HRI SPATIAL AMBIGUITY] "
                f"utterance={msg.utterance!r} | "
                f"intent={intent} | "
                f"direction={msg.direction} | "
                "authorization=False | "
                "action=REQUEST_CLARIFICATION"
            )

            self._direction_semantic_candidate = None
            self._direction_semantic_ready = False

            self._hri_retry_notice = "UNCLEAR"
            self._handle_rejected_human_response()
            return

        if self._handle_explicit_human_refusal(msg):
            return

        if intent == "REJECT":
            self._handle_rejected_human_response()
            return

        # Preserve the perspective of the initial answer.
        if self.fsm_state == STATE_WAITING_FOR_HUMAN:
            direction = self._normalize_direction(
                msg.direction
            )

            expected = {
                "REQUEST_LEFT": "LEFT",
                "REQUEST_RIGHT": "RIGHT",
            }.get(intent)

            if (
                expected is not None
                and direction == "NONE"
            ):
                direction = expected

            if (
                intent in {
                    "REQUEST_LEFT",
                    "REQUEST_RIGHT",
                    "ALLOW_PASSAGE",
                    "CONFIRM",
                }
                and direction in {"LEFT", "RIGHT"}
                and (
                    expected is None
                    or expected == direction
                )
            ):
                passage = interpret_passage_direction(
                    direction,
                    msg.utterance,
                )

                if passage is not None:
                    self._direction_semantic_candidate = {
                        "scope": self._scope,
                        "interaction_id": interaction_id,
                        "request_id": request_id,
                        "direction": passage.direction,
                        "reference": passage.reference,
                        "explicit_reference": (
                            passage.explicit_reference
                        ),
                        "resolved_robot_direction": "NONE",
                        "final_confirmation_received": False,
                    }

                    self._direction_semantic_ready = False

                    self.get_logger().info(
                        "[HRI DIRECTION] Candidate captured | "
                        f"direction={passage.direction} | "
                        f"reference={passage.reference} | "
                        f"interaction={interaction_id} | "
                        "authorization=False"
                    )

        return super()._human_intent_callback(msg)
    
    def _request_intent_confirmation(self):
        """
        Route passage permission through perspective-aware HRI turns.

        Demo:
          - YES without direction: retain existing direct C handling.
          - Explicit PERSON/ROBOT direction: finalize D without
            another confirmation question.
          - Ambiguous LEFT/RIGHT: ask whose left/right.

        Normal:
          - Preserve the existing directional confirmation protocol.

        Never resolve an ambiguous reference automatically.
        """

        demo = bool(
            getattr(
                self,
                "demo_robot_only_single_yes",
                False,
            )
        )

        if (
            demo
            and self.active_dialogue_request_type
                == "PASSAGE_CLARIFICATION"
            and self.pending_human_intent == "ALLOW_PASSAGE"
            and self.pending_hri_close is None
            and self.pending_dialogue_scope
                == self._social_scope_key()
        ):
            direction = self._normalize_direction(
                self.pending_human_direction
            )

            # YES without a directional preference.
            if direction == "NONE":
                self._handle_confirmation_response(
                    "CONFIRM"
                )
                return True

            if direction in {"LEFT", "RIGHT"}:
                candidate = getattr(
                    self,
                    "_direction_semantic_candidate",
                    None,
                )

                if (
                    isinstance(candidate, dict)
                    and candidate.get("scope") == self._scope
                    and candidate.get("interaction_id")
                        == self.active_interaction_id
                    and candidate.get("direction") == direction
                ):
                    reference = str(
                        candidate.get("reference") or ""
                    ).strip().upper()

                    # Only an EXPLICIT perspective can bypass
                    # the final confirmation in demo mode.
                    #
                    # AMBIGUOUS deliberately falls through
                    # to the directional clarification below.
                    if reference in {"PERSON", "ROBOT"}:

                        resolved = (
                            "RIGHT"
                            if direction == "LEFT"
                            else "LEFT"
                        ) if reference == "PERSON" else direction

                        candidate[
                            "resolved_robot_direction"
                        ] = resolved

                        candidate["spatial_relation"] = (
                            "FACE_TO_FACE_ASSUMED"
                            if reference == "PERSON"
                            else "ROBOT_REFERENCE"
                        )

                        candidate[
                            "final_confirmation_received"
                        ] = True

                        candidate[
                            "final_confirmation_request_id"
                        ] = self.pending_dialogue_request_id

                        self._direction_semantic_ready = False
                        self._directional_announcement_complete = False

                        self._directional_finalization = {
                            "scope": self._scope,
                            "interaction_id":
                                self.active_interaction_id,
                            "request_id":
                                self.pending_dialogue_request_id,
                            "candidate": dict(candidate),
                            "phase": "CLOSING_HRI",
                            "announcement_id": "",
                            "started_at": time.monotonic(),
                            "demo_single_turn_consent": True,
                        }

                        self.dialogue_started_at = None

                        if not self._request_hri_close(
                            "DEMO_DIRECTIONAL_PERMISSION"
                        ):
                            self._directional_finalization = None

                            self._publish_wait(
                                "DEMO_HRI_CLOSURE_UNAVAILABLE"
                            )
                            return False

                        self._transition_to(
                            STATE_WAITING,
                            reason=(
                                "DEMO_DIRECTIONAL_"
                                "PERMISSION_CLOSING"
                            ),
                        )

                        self.get_logger().info(
                            "[HRI DEMO] Direction accepted | "
                            f"human={direction} | "
                            f"robot={resolved} | "
                            "additional_confirmation=False"
                        )

                        return True

        # --------------------------------------------------
        # COMMON DIRECTIONAL ROUTING
        #
        # Also reached in demo mode when the human says
        # "on the left" without identifying the perspective.
        # --------------------------------------------------

        candidate = getattr(
            self,
            "_direction_semantic_candidate",
            None,
        )

        direction = self._normalize_direction(
            self.pending_human_direction
        )

        if (
            self.pending_human_intent != "ALLOW_PASSAGE"
            or direction not in {"LEFT", "RIGHT"}
        ):
            return super()._request_intent_confirmation()

        if (
            not isinstance(candidate, dict)
            or candidate.get("scope") != self._scope
            or candidate.get("interaction_id")
                != self.active_interaction_id
            or candidate.get("direction") != direction
        ):
            self._publish_wait(
                "DIRECTION_SEMANTIC_CONTEXT_MISSING"
            )

            self._transition_to(
                STATE_WAITING,
                reason="DIRECTION_SEMANTIC_CONTEXT_MISSING",
            )
            return False

        reference = str(
            candidate.get("reference") or ""
        ).strip().upper()

        if reference == "AMBIGUOUS":
            stage = "DIRECTION_REFERENCE_CLARIFICATION"

        elif reference in {"PERSON", "ROBOT"}:
            stage = "DIRECTION_FINAL_CONFIRMATION"

        else:
            self._publish_wait(
                "INVALID_DIRECTIONAL_REFERENCE"
            )

            self._transition_to(
                STATE_WAITING,
                reason="INVALID_DIRECTIONAL_REFERENCE",
            )
            return False

        # Actually schedule the next directional dialogue.
        self.dialogue_retry_count = 0
        self._directional_next_stage = stage
        self.dialogue_started_at = None

        # Closing the previous turn is necessary before
        # the next question can be published.
        if not self._request_hri_close(
            "DIRECTIONAL_NEXT_TURN"
        ):
            self._directional_next_stage = None

            self._begin_autonomous_readaptation_announcement(
                "HRI_CLOSE_UNAVAILABLE"
            )
            return False

        self._transition_to(
            STATE_WAITING,
            reason="DIRECTIONAL_NEXT_TURN_CLOSING",
        )

        return True
    
    def _directional_question(self, stage, retry_count=0):
        """Build the initial question and at most two polite reminders."""
        candidate = self._direction_semantic_candidate

        direction = str(
            candidate["direction"]
        ).strip().upper()

        reference = str(
            candidate["reference"]
        ).strip().upper()

        if direction not in {"LEFT", "RIGHT"}:
            raise ValueError("INVALID_DIRECTION")

        retry_count = int(retry_count)

        if retry_count not in {0, 1, 2}:
            raise ValueError("DIRECTIONAL_RETRY_LIMIT_EXCEEDED")

        language = str(self.language or "en").lower()[:2]

        if language not in {"fr", "en", "es"}:
            language = "en"

        sides = {
            "fr": {"LEFT": "gauche", "RIGHT": "droite"},
            "en": {"LEFT": "left", "RIGHT": "right"},
            "es": {"LEFT": "izquierda", "RIGHT": "derecha"},
        }

        side = sides[language][direction]

        if stage == "DIRECTION_REFERENCE_CLARIFICATION":
            questions = {
                "fr": (
                    f"Pour être certain de bien comprendre votre indication, "
                    f"parlez-vous de votre {side} ou de la mienne ?",
                    f"Excusez-moi, je n'ai pas reçu votre réponse. "
                    f"Pourriez-vous me préciser s'il s'agit de votre "
                    f"{side} ou de la mienne ?",
                    f"Je vous remercie de votre patience. "
                    f"Pour éviter toute confusion, pourriez-vous me "
                    f"confirmer s'il s'agit de votre {side} "
                    f"ou de la mienne ?",
                ),
                "en": (
                    f"To make sure I understand your instruction "
                    f"correctly, do you mean your {side} or mine?",
                    f"Excuse me, I didn't receive your answer. "
                    f"Could you please clarify whether you mean "
                    f"your {side} or mine?",
                    f"Thank you for your patience. "
                    f"To avoid any misunderstanding, could you "
                    f"please confirm whether you mean your "
                    f"{side} or mine?",
                ),
                "es": (
                    f"Para asegurarme de haber comprendido bien "
                    f"su indicación, ¿se refiere a su {side} "
                    f"o a la mía?",
                    f"Disculpe, no he recibido su respuesta. "
                    f"¿Podría indicarme, por favor, si se refiere "
                    f"a su {side} o a la mía?",
                    f"Le agradezco su paciencia. "
                    f"Para evitar cualquier confusión, "
                    f"¿podría confirmarme si se refiere "
                    f"a su {side} o a la mía?",
                ),
            }

            return questions[language][retry_count]

        if stage != "DIRECTION_FINAL_CONFIRMATION":
            raise ValueError("INVALID_DIRECTIONAL_STAGE")

        if reference == "PERSON":
            resolved = (
                "RIGHT" if direction == "LEFT"
                else "LEFT"
            )

            robot_side = sides[language][resolved]

            statements = {
                "fr": (
                    f"Vous souhaitez que je passe à votre {side}. "
                    f"En supposant que nous nous faisons face, "
                    f"cela correspond à ma {robot_side}. "
                ),
                "en": (
                    f"You would like me to pass on your {side}. "
                    f"Assuming we are facing each other, "
                    f"that corresponds to my {robot_side}. "
                ),
                "es": (
                    f"Desea que pase por su {side}. "
                    f"Suponiendo que estamos frente a frente, "
                    f"eso corresponde a mi {robot_side}. "
                ),
            }

        elif reference == "ROBOT":
            statements = {
                "fr": f"Vous souhaitez que je passe à ma {side}. ",
                "en": f"You would like me to pass on my {side}. ",
                "es": f"Desea que pase por mi {side}. ",
            }

        else:
            raise ValueError("DIRECTION_REFERENCE_UNRESOLVED")

        statement = statements[language]

        prefixes = {
            "fr": (
                "Pour être certain de respecter votre indication, ",
                "Excusez-moi, je n'ai pas reçu votre confirmation. ",
                "Je vous remercie de votre patience. ",
            ),
            "en": (
                "To make sure I respect your instruction, ",
                "Excuse me, I didn't receive your confirmation. ",
                "Thank you for your patience. ",
            ),
            "es": (
                "Para asegurarme de respetar su indicación, ",
                "Disculpe, no he recibido su confirmación. ",
                "Le agradezco su paciencia. ",
            ),
        }

        endings = {
            "fr": (
                "Est-ce bien ce que vous souhaitez ?",
                "Pourriez-vous me confirmer que c'est bien cela ?",
                "Pourriez-vous me confirmer une dernière fois votre choix ?",
            ),
            "en": (
                "Is that what you would like?",
                "Could you please confirm that this is correct?",
                "Could you please confirm your preference one last time?",
            ),
            "es": (
                "¿Es eso lo que desea?",
                "¿Podría confirmarme, por favor, que es correcto?",
                "¿Podría confirmarme su elección una última vez?",
            ),
        }

        return (
            prefixes[language][retry_count]
            + statement
            + endings[language][retry_count]
        )

    def _request_directional_dialogue_turn(self, stage):
        """
        Publish a perspective-aware dialogue turn.

        Preserve the human-requested direction, reference,
        robot-resolved direction, and retry number.
        """
        candidate = getattr(
            self,
            "_direction_semantic_candidate",
            None,
        )

        if (
            not isinstance(candidate, dict)
            or candidate.get("scope") != self._scope
            or candidate.get("interaction_id")
                != self.active_interaction_id
            or self.pending_hri_close is not None
        ):
            return False

        direction = str(
            candidate.get("direction", "")
        ).strip().upper()

        reference = str(
            candidate.get("reference", "")
        ).strip().upper()

        if direction not in {"LEFT", "RIGHT"}:
            return False

        retry_count = int(self.dialogue_retry_count)

        if retry_count not in {0, 1, 2}:
            return False

        if stage == "DIRECTION_REFERENCE_CLARIFICATION":
            if reference != "AMBIGUOUS":
                return False

            resolved = "NONE"

        elif stage == "DIRECTION_FINAL_CONFIRMATION":
            if reference == "PERSON":
                resolved = (
                    "RIGHT" if direction == "LEFT"
                    else "LEFT"
                )

            elif reference == "ROBOT":
                resolved = direction

            else:
                return False

        else:
            return False

        try:
            question = self._directional_question(
                stage,
                retry_count=retry_count,
            )
        except (ValueError, KeyError, TypeError):
            return False

        request = self._new_dialogue_request()

        if request is None:
            self._publish_wait(
                "DIRECTIONAL_HRI_TURN_UNAVAILABLE"
            )
            return False

        request.request_type = stage
        request.context = "DIRECTIONAL_PERMISSION"
        request.question = question
        request.language = self.language

        request.intent_to_confirm = "ALLOW_PASSAGE"
        request.direction_to_confirm = direction

        # Explicit spatial semantics for multilingual fallback.
        request.directional_reference = reference
        request.resolved_robot_direction = resolved
        request.directional_retry_index = retry_count

        request.robot_initiated = True
        request.response_timeout = min(
            15.0,
            float(self.dialogue_response_timeout_sec),
        )

        self.active_dialogue_request_type = stage
        self.waiting_for_human_initiative = False
        self.dialogue_outcome = "PENDING"
        self.dialogue_started_at = time.monotonic()

        self._transition_to(
            STATE_WAITING_FOR_CONFIRMATION,
            reason=stage,
        )

        self.dialogue_request_pub.publish(request)

        self.get_logger().info(
            "[HRI DIRECTION] Question submitted | "
            f"stage={stage} | "
            f"language={request.language} | "
            f"human_direction={direction} | "
            f"reference={reference} | "
            f"robot_direction={resolved} | "
            f"retry={retry_count}/2 | "
            f"request={request.request_id}"
        )

        return True

    def _finalize_demo_clarified_direction(self, candidate):
        """
        Finalize an explicit robot/person directional correction
        in demo mode.

        Accept corrections during:
        - reference clarification;
        - final directional confirmation.

        Preserve HRI ownership and the exact closing barrier.
        Never issue START directly.
        """
        if not getattr(
            self,
            "demo_robot_only_single_yes",
            False,
        ):
            return False

        scope = self._social_scope_key()

        turn_key = (
            self.active_interaction_id,
            self.pending_dialogue_request_id,
        )

        owner = self.mode_d_hri_turns.get(turn_key)

        accepted_stages = {
            "DIRECTION_REFERENCE_CLARIFICATION",
            "DIRECTION_FINAL_CONFIRMATION",
        }

        if not (
            isinstance(candidate, dict)
            and scope
            and scope == self._scope
            and candidate.get("scope") == scope
            and candidate.get("interaction_id")
                == self.active_interaction_id
            and self.pending_dialogue_scope == scope
            and owner is not None
            and owner == self.mode_d_hri_owner
            and self.pending_hri_close is None
            and self.active_dialogue_request_type
                in accepted_stages
        ):
            return False

        direction = self._normalize_direction(
            candidate.get("direction")
        )

        reference = str(
            candidate.get("reference") or ""
        ).strip().upper()

        if (
            direction not in {"LEFT", "RIGHT"}
            or reference not in {"PERSON", "ROBOT"}
        ):
            return False

        resolved = (
            (
                "RIGHT"
                if direction == "LEFT"
                else "LEFT"
            )
            if reference == "PERSON"
            else direction
        )

        candidate["resolved_robot_direction"] = resolved

        candidate["spatial_relation"] = (
            "FACE_TO_FACE_ASSUMED"
            if reference == "PERSON"
            else "ROBOT_REFERENCE"
        )

        candidate["final_confirmation_received"] = True

        candidate["final_confirmation_request_id"] = (
            self.pending_dialogue_request_id
        )

        self._direction_semantic_ready = False
        self._directional_announcement_complete = False
        self._directional_next_stage = None

        self._directional_finalization = {
            "scope": scope,
            "interaction_id": self.active_interaction_id,
            "request_id": self.pending_dialogue_request_id,
            "candidate": dict(candidate),
            "phase": "CLOSING_HRI",
            "announcement_id": "",
            "started_at": time.monotonic(),
            "demo_single_turn_consent": True,
            "demo_reference_clarified": True,
        }

        self.dialogue_started_at = None

        if not self._request_hri_close(
            "DEMO_DIRECTION_CLARIFIED"
        ):
            self._directional_finalization = None

            self._publish_wait(
                "DEMO_DIRECTION_HRI_CLOSE_UNAVAILABLE"
            )

            self._transition_to(
                STATE_WAITING,
                reason="DEMO_DIRECTION_HRI_CLOSE_UNAVAILABLE",
            )

            return False

        self._publish_wait(
            "DEMO_DIRECTION_HRI_CLOSURE_PENDING"
        )

        self._transition_to(
            STATE_WAITING,
            reason="DEMO_DIRECTION_HRI_CLOSURE_PENDING",
        )

        self.get_logger().info(
            "[HRI DEMO] Clarified direction accepted | "
            f"human={direction} | "
            f"robot={resolved} | "
            "final_question_skipped=True | "
            "navigation_authorized=False"
        )

        return True
    
    def _handle_directional_dialogue_response(self, msg):
        """
        Process perspective clarification and final confirmation.

        A perspective correction starts a new confirmation turn.
        An unclear answer triggers bounded retries.
        An explicit refusal retains the goal in WAIT.
        """
        import re

        candidate = getattr(
            self,
            "_direction_semantic_candidate",
            None,
        )

        if (
            not isinstance(candidate, dict)
            or candidate.get("scope") != self._scope
            or candidate.get("interaction_id")
                != self.active_interaction_id
            or str(msg.interaction_id)
                != self.active_interaction_id
            or str(msg.request_id)
                != self.pending_dialogue_request_id
            or self.pending_hri_close is not None
        ):
            self._publish_wait(
                "DIRECTIONAL_CONTEXT_INVALID"
            )
            return

        stage = str(
            self.active_dialogue_request_type or ""
        ).strip().upper()

        intent = str(
            msg.intent or ""
        ).strip().upper()

        utterance = str(
            msg.utterance or ""
        ).strip().lower()

        if intent == "REQUEST_STOP":
            self._direction_semantic_ready = False
            self._publish_wait("HUMAN_REQUESTED_STOP")
            self._transition_to(
                STATE_WAITING,
                reason="HUMAN_REQUESTED_STOP",
            )
            return

        if stage not in {
            "DIRECTION_REFERENCE_CLARIFICATION",
            "DIRECTION_FINAL_CONFIRMATION",
        }:
            self._publish_wait(
                "INVALID_DIRECTIONAL_DIALOGUE_STAGE"
            )
            return

        left = bool(
            re.search(
                r"\b(?:left|gauche|izquierda)\b",
                utterance,
            )
        )

        right = bool(
            re.search(
                r"\b(?:right|droite|derecha)\b",
                utterance,
            )
        )

        # Conflicting directions cannot be treated as consent.
        if left and right:
            self._handle_rejected_human_response()
            return

        requested_direction = str(
            candidate.get("direction", "NONE")
        ).strip().upper()

        mentioned_direction = (
            "LEFT" if left
            else "RIGHT" if right
            else requested_direction
        )

        interpretation = interpret_passage_direction(
            mentioned_direction,
            utterance,
        )

        reference_recognized = bool(
            interpretation is not None
            and interpretation.reference in {"PERSON", "ROBOT"}
        )

        correction = bool(
            reference_recognized
            and (
                stage == "DIRECTION_REFERENCE_CLARIFICATION"
                or interpretation.direction
                    != candidate.get("direction")
                or interpretation.reference
                    != candidate.get("reference")
            )
        )

        if correction:
            self.get_logger().info(
                "[HRI DIRECTION] Perspective correction | "
                f"old_direction={candidate.get('direction')} | "
                f"old_reference={candidate.get('reference')} | "
                f"new_direction={interpretation.direction} | "
                f"new_reference={interpretation.reference}"
            )

            candidate["direction"] = interpretation.direction
            candidate["reference"] = interpretation.reference
            candidate["explicit_reference"] = True
            candidate["final_confirmation_received"] = False
            candidate["resolved_robot_direction"] = "NONE"

            self.pending_human_direction = (
                interpretation.direction
            )

            self._direction_semantic_ready = False
            self.dialogue_retry_count = 0

            if (
                getattr(
                    self, "demo_robot_only_single_yes", False
                )
                and stage == "DIRECTION_REFERENCE_CLARIFICATION"
            ):
                if self._finalize_demo_clarified_direction(
                    candidate
                ):
                    return

                # Do not open another question while an HRI
                # closure is already pending.
                if self.pending_hri_close is not None:
                    return
                
            self._directional_next_stage = (
                "DIRECTION_FINAL_CONFIRMATION"
            )

            self.dialogue_started_at = None

            if not self._request_hri_close(
                "DIRECTIONAL_REFERENCE_UPDATED"
            ):
                self._directional_next_stage = None
                self._publish_wait(
                    "DIRECTIONAL_HRI_CLOSE_UNAVAILABLE"
                )
                self._transition_to(
                    STATE_WAITING,
                    reason="DIRECTIONAL_HRI_CLOSE_UNAVAILABLE",
                )
                return

            self._transition_to(
                STATE_WAITING,
                reason="DIRECTIONAL_REFERENCE_UPDATE_PENDING",
            )
            return

        # A standalone explicit negative answer is not silence.
        if self._handle_explicit_human_refusal(msg):
            return

        # Clarification requires an actual perspective.
        if stage == "DIRECTION_REFERENCE_CLARIFICATION":
            self._hri_retry_notice = "UNCLEAR"
            self._handle_rejected_human_response()
            return

        # In final confirmation, an unclear or rejected response
        # triggers a polite retry, not immediate indefinite WAIT.
        if intent != "CONFIRM":
            candidate["final_confirmation_received"] = False
            self._direction_semantic_ready = False

            self._hri_retry_notice = "UNCLEAR"
            self._handle_rejected_human_response()
            return

        direction = str(
            candidate.get("direction", "")
        ).strip().upper()

        reference = str(
            candidate.get("reference", "")
        ).strip().upper()

        if direction not in {"LEFT", "RIGHT"}:
            self._publish_wait(
                "INVALID_CONFIRMED_DIRECTION"
            )
            return

        if reference == "PERSON":
            resolved = (
                "RIGHT" if direction == "LEFT"
                else "LEFT"
            )
            spatial_relation = "FACE_TO_FACE_ASSUMED"

        elif reference == "ROBOT":
            resolved = direction
            spatial_relation = "ROBOT_REFERENCE"

        else:
            self._hri_retry_notice = "UNCLEAR"
            self._handle_rejected_human_response()
            return

        scope = self._social_scope_key()

        turn_key = (
            self.active_interaction_id,
            self.pending_dialogue_request_id,
        )

        owner = self.mode_d_hri_turns.get(turn_key)

        if (
            scope != candidate["scope"]
            or owner is None
            or owner != self.mode_d_hri_owner
            or self.pending_dialogue_scope != scope
        ):
            self._publish_wait(
                "DIRECTIONAL_FINAL_TURN_MISMATCH"
            )
            return

        candidate["resolved_robot_direction"] = resolved
        candidate["spatial_relation"] = spatial_relation
        candidate["final_confirmation_received"] = True
        candidate["final_confirmation_request_id"] = (
            self.pending_dialogue_request_id
        )

        self._direction_semantic_ready = False
        self._directional_announcement_complete = False

        finalization = {
            "scope": scope,
            "interaction_id": self.active_interaction_id,
            "request_id": self.pending_dialogue_request_id,
            "candidate": dict(candidate),
            "phase": "CLOSING_HRI",
            "announcement_id": "",
            "started_at": time.monotonic(),
        }

        self._directional_finalization = finalization
        self.dialogue_started_at = None

        if not self._request_hri_close(
            "DIRECTIONAL_FINAL_CONFIRMATION"
        ):
            self._directional_finalization = None

            self._publish_wait(
                "DIRECTIONAL_HRI_CLOSE_UNAVAILABLE"
            )

            self._transition_to(
                STATE_WAITING,
                reason="DIRECTIONAL_HRI_CLOSE_UNAVAILABLE",
            )
            return

        self._publish_wait(
            "DIRECTIONAL_HRI_CLOSURE_PENDING"
        )

        self._transition_to(
            STATE_WAITING,
            reason="DIRECTIONAL_HRI_CLOSURE_PENDING",
        )

        self.get_logger().info(
            "[HRI DIRECTION] Final confirmation received | "
            f"requested={direction} | "
            f"reference={reference} | "
            f"resolved_robot_direction={resolved} | "
            f"spatial_relation={spatial_relation} | "
            "navigation_authorized=False"
        )
    
    def _handle_initial_human_intent(self, intent, direction):
        """
        Normalize the initial human response.

        Demo mode:
            - YES without direction -> Mode C, through
              _request_intent_confirmation().
            - Explicit LEFT/RIGHT -> Mode D, through
              _request_intent_confirmation().
            - No additional final confirmation question.

        Normal mode:
            - Preserve the complete confirmation protocol.

        This method never authorizes navigation directly.
        """
        intent = str(intent or "").strip().upper()
        direction = self._normalize_direction(direction)

        requested_directions = {
            "REQUEST_LEFT": "LEFT",
            "REQUEST_RIGHT": "RIGHT",
        }

        # ---------------------------------------------
        # EXPLICIT DIRECTIONAL REQUEST
        # ---------------------------------------------

        if intent in requested_directions:
            expected = requested_directions[intent]

            if direction not in {"NONE", expected}:
                self.get_logger().warning(
                    "[HRI] Contradictory directional request | "
                    f"intent={intent} | "
                    f"direction={direction}"
                )

                self._handle_rejected_human_response()
                return

            self.get_logger().info(
                "[HRI] Explicit directional request | "
                f"direction={expected} | "
                "next=INTENT_CONFIRMATION_ROUTING"
            )

            return super()._handle_initial_human_intent(
                "ALLOW_PASSAGE",
                expected,
            )

        # ---------------------------------------------
        # DIRECTION PROVIDED WITH CONFIRM
        # ---------------------------------------------

        if (
            intent == "CONFIRM"
            and self.waiting_for_human_initiative
            and direction in {"LEFT", "RIGHT"}
        ):
            self.get_logger().info(
                "[HRI] Directed confirmation | "
                f"direction={direction} | "
                "next=INTENT_CONFIRMATION_ROUTING"
            )

            return super()._handle_initial_human_intent(
                "ALLOW_PASSAGE",
                direction,
            )

        # ---------------------------------------------
        # ALL OTHER HUMAN RESPONSES
        # ---------------------------------------------

        # Includes:
        # - YES / CONFIRM without direction
        # - ALLOW_PASSAGE without direction
        # - DENY_PASSAGE
        # - REJECT
        # - REQUEST_STOP
        # - REQUEST_FOLLOW
        #
        # The parent handles intent normalization.
        # _request_intent_confirmation() decides whether
        # the demo shortcut is active.

        return super()._handle_initial_human_intent(
            intent,
            direction,
        )
    
    def _handle_explicit_human_refusal(self, msg):
        """
        Stop negotiation after an explicit, reliable human refusal.

        A rejected ASR interpretation is not automatically a refusal.
        The navigation goal is retained in WAIT.
        """
        import re

        intent = str(msg.intent or "").strip().upper()
        utterance = str(msg.utterance or "").strip().lower()
        confidence = float(msg.confidence)

        normalized = re.sub(
            r"[.!?¡¿,;:]+",
            " ",
            utterance,
        )
        normalized = " ".join(normalized.split())

        explicit_negative = normalized in {
            "no",
            "nope",
            "no thanks",
            "no thank you",
            "non",
            "non merci",
            "no gracias",
            "you cannot pass",
            "you can't pass",
        }

        if not (
            explicit_negative
            and intent in {"REJECT", "DENY_PASSAGE"}
            and confidence >= 0.5
        ):
            return False

        self.get_logger().info(
            "[HRI REFUSAL] Explicit human refusal | "
            f"utterance={utterance!r} | "
            f"confidence={confidence:.3f} | "
            "policy=KEEP_GOAL_IN_WAIT"
        )

        self.dialogue_started_at = None
        self.dialogue_outcome = "DENY_CONFIRMED"

        self.confirmed_human_intent = "DENY_PASSAGE"
        self.confirmed_human_direction = "NONE"
        self.authorized_episode_key = None

        self._direction_semantic_ready = False
        self._directional_next_stage = None
        self._hri_retry_after_close = None

        self._clear_pending_human_preference()

        # Close the exact dialogue turn.
        # A failed closure must not trigger another permission question.
        if self.pending_hri_close is None:
            self._request_hri_close(
                "HUMAN_EXPLICIT_REFUSAL"
            )

        # Keep the active goal and prevent a new route search
        # within the refused social scope.
        self._hold_after_denial()

        return True
    
    def _handle_rejected_human_response(self):
        """
        Retry an unrecognized human response.

        Preserve pending semantic context during confirmation.
        A recognition failure is not a human refusal.
        """
        request_type = str(
            self.active_dialogue_request_type or ""
        ).strip().upper()

        if request_type not in {
            "INTENT_CONFIRMATION",
            "ALTERNATIVE_PROPOSAL",
            "DIRECTION_REFERENCE_CLARIFICATION",
            "DIRECTION_FINAL_CONFIRMATION",
        }:
            # No confirmed proposition is being clarified.
            self._clear_pending_human_preference()

        # During confirmation, retain:
        # - pending_human_intent
        # - pending_human_direction
        # The next question must confirm the same proposition.

        self._hri_retry_notice = "UNCLEAR"

        self.get_logger().warning(
            "[HRI] Unrecognized human response | "
            f"request_type={request_type} | "
            f"pending_intent={self.pending_human_intent} | "
            f"pending_direction={self.pending_human_direction} | "
            "action=RETRY_SAME_QUESTION"
        )

        self._handle_dialogue_timeout()

    # --------------------------------------------------
    # BOUNDED LISTENING RETRIES
    # --------------------------------------------------


    def _handle_dialogue_timeout(self):
        """
        Close a timed-out HRI turn before bounded retries.

        Directional dialogue: maximum two reminders.
        Other dialogue types: preserve the existing limit.
        """
        if (
            not self._scope
            or self._released
            or self._denied_scope
            or self.pending_terminal_stop is not None
            or self.pending_hri_close is not None
            or getattr(
                self,
                "_hri_nav_fallback_active",
                False,
            )
        ):
            return

        if (
            not self.active_interaction_id
            or not self.pending_dialogue_request_id
        ):
            self._begin_autonomous_readaptation_announcement(
                "HRI_TURN_UNAVAILABLE"
            )
            return

        initiative = bool(
            self.waiting_for_human_initiative
        )

        request_type = str(
            self.active_dialogue_request_type or ""
        ).strip().upper()

        directional = request_type in {
            "DIRECTION_REFERENCE_CLARIFICATION",
            "DIRECTION_FINAL_CONFIRMATION",
        }

        max_retries = 2 if directional else 3

        if initiative:
            # Preserve the existing initiative-window behavior.
            self.dialogue_retry_count = 0

        elif self.dialogue_retry_count >= max_retries:
            unclear = (
                getattr(self, "_hri_retry_notice", "MISSED")
                == "UNCLEAR"
            )

            fallback_reason = (
                "UNUSABLE_HUMAN_RESPONSE"
                if unclear
                else "NO_HUMAN_RESPONSE"
            )

            self.get_logger().info(
                "[HRI RETRY] Limit reached | "
                f"stage={request_type} | "
                f"retries={self.dialogue_retry_count} | "
                f"reason={fallback_reason} | "
                "fallback=MODE_C_WITH_ANNOUNCEMENT"
            )

            self._begin_autonomous_readaptation_announcement(
                fallback_reason
            )
            return
        else:
            self.dialogue_retry_count += 1

        self._hri_retry_after_close = {
            "goal": self.active_navigation_goal_id,
            "scope": self._scope,
            "interaction_id": self.active_interaction_id,
            "request_id": self.pending_dialogue_request_id,
            "request_type": self.active_dialogue_request_type,
            "initiative": initiative,
        }

        self.dialogue_started_at = None

        close_reason = (
            "HUMAN_RESPONSE_NOT_UNDERSTOOD"
            if self._hri_retry_notice == "UNCLEAR"
            else "HUMAN_RESPONSE_TIMEOUT"
        )

        if not self._request_hri_close(close_reason):
            self._hri_retry_after_close = None

            self._begin_autonomous_readaptation_announcement(
                "HRI_CLOSE_UNAVAILABLE"
            )
            return

        self._transition_to(
            STATE_WAITING,
            reason="HRI_RETRY_AWAITING_CLOSE",
        )

    def _retry_active_dialogue_request(self):
        """Retry a question after the previous HRI turn closes."""
        if (
            not self._scope
            or self._released
            or self._denied_scope
            or self.pending_hri_close is not None
            or self.pending_dialogue_request_id
            or getattr(self, "_hri_detached_close", None) is not None
        ):
            return

        request_type = str(
            self.active_dialogue_request_type or ""
        ).strip().upper()

        if request_type in {
            "DIRECTION_REFERENCE_CLARIFICATION",
            "DIRECTION_FINAL_CONFIRMATION",
        }:
            candidate = getattr(
                self,
                "_direction_semantic_candidate",
                None,
            )

            if (
                isinstance(candidate, dict)
                and candidate.get("scope") == self._scope
                and not self.active_interaction_id
                and not self.pending_dialogue_request_id
            ):
                self.active_interaction_id = (
                    candidate["interaction_id"]
                )

            return self._request_directional_dialogue_turn(
                request_type
            )
        
        if request_type == "PASSAGE_CLARIFICATION":
            question_key = "passage"
            context = "SOCIAL_CONFLICT"
            intent = ""
            direction = "NONE"
            target_state = STATE_WAITING_FOR_HUMAN

        elif request_type == "INTENT_CONFIRMATION":
            intent = str(
                self.pending_human_intent or ""
            ).strip().upper()

            direction = self._normalize_direction(
                self.pending_human_direction
            )

            if intent == "DENY_PASSAGE":
                question_key = "confirm_deny"

            elif intent == "ALLOW_PASSAGE":
                question_key = {
                    "LEFT": "confirm_allow_left",
                    "RIGHT": "confirm_allow_right",
                }.get(direction, "confirm_allow")

            else:
                self._begin_autonomous_readaptation_announcement(
                    "HRI_CONFIRMATION_CONTEXT_LOST"
                )
                return

            context = "INTENT_REQUIRES_CONFIRMATION"
            target_state = STATE_WAITING_FOR_CONFIRMATION

        elif request_type == "ALTERNATIVE_PROPOSAL":
            direction = self._normalize_direction(
                self.active_alternative_direction
            )

            if direction not in {"LEFT", "RIGHT"}:
                self._begin_autonomous_readaptation_announcement(
                    "HRI_ALTERNATIVE_CONTEXT_LOST"
                )
                return

            question_key = (
                "alternative_left"
                if direction == "LEFT"
                else "alternative_right"
            )

            intent = "ALLOW_PASSAGE"
            context = "REQUESTED_ROUTE_UNAVAILABLE"
            target_state = STATE_WAITING_FOR_CONFIRMATION

        else:
            self._begin_autonomous_readaptation_announcement(
                "HRI_QUESTION_CONTEXT_LOST"
            )
            return

        request = self._new_dialogue_request()

        if request is None:
            self._begin_autonomous_readaptation_announcement(
                "HRI_RETRY_UNAVAILABLE"
            )
            return

        unclear = (
            getattr(self, "_hri_retry_notice", "MISSED")
            == "UNCLEAR"
        )

        if self.language == "fr":
            prefix = (
                "Je n'ai pas compris votre réponse. "
                if unclear
                else "Je n'ai pas pu enregistrer votre réponse. "
            )
        else:
            prefix = (
                "I couldn't understand your answer. "
                if unclear
                else "I couldn't register your answer. "
            )

        request.request_type = request_type
        request.context = context
        request.question = (
            prefix
            + get_dialogue_text(question_key, self.language)
        )

        request.intent_to_confirm = intent
        request.direction_to_confirm = direction
        request.robot_initiated = True
        request.response_timeout = min(
            15.0,
            float(self.dialogue_response_timeout_sec),
        )

        self.active_dialogue_request_type = request_type
        self.waiting_for_human_initiative = False
        self.dialogue_outcome = "PENDING"
        self.dialogue_started_at = time.monotonic()

        self._transition_to(
            target_state,
            reason="HRI_QUESTION_RETRY",
        )

        self.dialogue_request_pub.publish(request)

        self.get_logger().info(
            "[HRI RETRY] Question submitted | "
            f"retry={self.dialogue_retry_count}/3 | "
            f"request={request.request_id}"
        )


    def _directional_final_announcement_text(self, candidate):
        """Announce confirmed robot-relative passage in fr, en, es."""
        direction = str(
            candidate["resolved_robot_direction"]
        ).strip().upper()

        reference = str(
            candidate["reference"]
        ).strip().upper()

        if direction not in {"LEFT", "RIGHT"}:
            raise ValueError("UNRESOLVED_ROBOT_DIRECTION")

        if reference not in {"PERSON", "ROBOT"}:
            raise ValueError("DIRECTION_REFERENCE_UNRESOLVED")

        language = str(self.language or "en").lower()[:2]

        if language not in {"fr", "en", "es"}:
            language = "en"

        sides = {
            "fr": {"LEFT": "gauche", "RIGHT": "droite"},
            "en": {"LEFT": "left", "RIGHT": "right"},
            "es": {"LEFT": "izquierda", "RIGHT": "derecha"},
        }

        side = sides[language][direction]

        if language == "fr":
            if reference == "PERSON":
                return (
                    "J'ai bien compris votre indication. "
                    "En supposant que nous nous faisons face, "
                    f"je vais chercher un passage à ma {side}, "
                    "en respectant votre espace et la distance de sécurité."
                )

            return (
                "J'ai bien compris votre indication. "
                f"Je vais chercher un passage à ma {side}, "
                "en respectant votre espace et la distance de sécurité."
            )

        if language == "es":
            if reference == "PERSON":
                return (
                    "He comprendido su indicación. "
                    "Suponiendo que estamos frente a frente, "
                    f"buscaré un paso por mi {side}, "
                    "respetando su espacio personal "
                    "y manteniendo una distancia segura."
                )

            return (
                "He comprendido su indicación. "
                f"Buscaré un paso por mi {side}, "
                "respetando su espacio personal "
                "y manteniendo una distancia segura."
            )

        if reference == "PERSON":
            return (
                "I understand your instruction. "
                "Assuming we are facing each other, "
                f"I will look for a passage on my {side}, "
                "while respecting your personal space "
                "and maintaining a safe distance."
            )

        return (
            "I understand your instruction. "
            f"I will look for a passage on my {side}, "
            "while respecting your personal space "
            "and maintaining a safe distance."
        )

    # --------------------------------------------------
    # HRI OWNERSHIP CLEANUP
    # --------------------------------------------------

    def _detach_hri_turn(self, *, confirmed_closed=False):
        """Detach optional HRI without altering the recorded decision."""
        pending = self.pending_hri_close

        if (
            pending is None
            and not confirmed_closed
            and self.pending_dialogue_request_id
        ):
            if self._request_hri_close(
                "AUTONOMOUS_NAVIGATION_HANDOFF"
            ):
                pending = self.pending_hri_close

        if pending is not None:
            turn_key = (
                pending["interaction_id"],
                pending["request_id"],
            )

            owner = self.mode_d_hri_turns.get(turn_key)

            if confirmed_closed and owner is not None:
                case = self.mode_d_cases.get(owner)

                if case is not None:
                    case["hri_closed"] = True

                self.mode_d_hri_closures[turn_key] = {
                    "close_id": pending["close_id"],
                    "scope": owner,
                }

                while len(self.mode_d_hri_closures) > 256:
                    oldest = next(iter(self.mode_d_hri_closures))
                    self.mode_d_hri_closures.pop(oldest, None)

            elif not confirmed_closed:
                self._hri_detached_close = dict(pending)

        old_interaction = str(
            self.active_interaction_id or ""
        )

        for key in list(self.mode_d_hri_turns):
            if key[0] == old_interaction:
                self.mode_d_hri_turns.pop(key, None)

        self.mode_d_hri_owner = None
        self.pending_hri_close = None
        self.hri_close_fault_reported = False
        self._hri_retry_after_close = None

        self.pending_dialogue_request_id = ""
        self.pending_dialogue_scope = None
        self.active_interaction_id = ""
        self.active_dialogue_request_type = ""

        self.dialogue_started_at = None
        self.waiting_for_human_initiative = False

        self.dialogue_audio_recovery_pending = False
        self.audio_recovery_interaction_id = ""
        self.audio_recovery_request_id = ""

    # --------------------------------------------------
    # DIRECTIONAL MASK REQUEST
    # --------------------------------------------------

    def _prepare_directional_mask_request(self, direction):
        """Build an optional mask request from bounded-age identity evidence."""
        direction = str(direction or "").strip().upper()

        def reject(reason):
            self.get_logger().warning(
                "[DIRECTIONAL MASK] Request unavailable | "
                f"reason={reason}",
                throttle_duration_sec=2.0,
            )
            return None

        if direction not in {"LEFT", "RIGHT"}:
            return reject("INVALID_DIRECTION")


        # Validate the exact human-confirmed directional transaction.
        finalization = getattr(
            self,
            "_directional_finalization",
            None,
        )

        if not isinstance(finalization, dict):
            return reject("FINALIZATION_MISSING")

        candidate = finalization.get("candidate")

        if not isinstance(candidate, dict):
            return reject("DIRECTIONAL_CANDIDATE_MISSING")

        if (
            finalization.get("phase") != "ANNOUNCED"
            or not getattr(
                self,
                "_directional_announcement_complete",
                False,
            )
            or not getattr(
                self,
                "_direction_semantic_ready",
                False,
            )
            or candidate.get("final_confirmation_received") is not True
        ):
            return reject("DIRECTIONAL_CONFIRMATION_INCOMPLETE")

        requested = str(
            candidate.get("direction", "")
        ).strip().upper()

        reference = str(
            candidate.get("reference", "")
        ).strip().upper()

        resolved = str(
            candidate.get("resolved_robot_direction", "")
        ).strip().upper()

        relation = str(
            candidate.get("spatial_relation", "")
        ).strip().upper()

        if (
            requested not in {"LEFT", "RIGHT"}
            or reference not in {"PERSON", "ROBOT"}
            or resolved not in {"LEFT", "RIGHT"}
        ):
            return reject("INVALID_DIRECTIONAL_SEMANTICS")

        if reference == "PERSON":
            if relation != "FACE_TO_FACE_ASSUMED":
                return reject("PERSON_REFERENCE_UNRESOLVED")

            expected = (
                "RIGHT" if requested == "LEFT"
                else "LEFT"
            )

        else:
            if relation != "ROBOT_REFERENCE":
                return reject("ROBOT_REFERENCE_UNRESOLVED")

            expected = requested

        if direction != expected or resolved != expected:
            return reject("RESOLVED_ROBOT_DIRECTION_MISMATCH")

        scope = self._social_scope_key()

        if (
            candidate.get("scope") != scope
            or finalization.get("scope") != scope
            or candidate.get("interaction_id")
                != finalization.get("interaction_id")
            or candidate.get("final_confirmation_request_id")
                != finalization.get("request_id")
            or not finalization.get("interaction_id")
            or not finalization.get("request_id")
        ):
            return reject("DIRECTIONAL_TRANSACTION_MISMATCH")

        if (
            self._normalize_direction(
                self.confirmed_human_direction
            ) != requested
        ):
            return reject("ORIGINAL_HUMAN_DIRECTION_MISMATCH")

        if not (
            self.interaction == "dialogue"
            and self.confirmed_human_intent == "ALLOW_PASSAGE"
            and self.dialogue_outcome == "ALLOW_CONFIRMED"
            and self._permission_applies_to_current_episode()
            and self._normalize_direction(
                self.confirmed_human_direction
            ) == requested
            and self.pending_terminal_stop is None
        ):
            return reject("PERMISSION_MISMATCH")

        scope = self._social_scope_key()

        if (
            not scope
            or len(scope) != 5
            or not all(scope)
            or scope[0] != self.active_navigation_goal_id
            or scope[4] != self.active_person_id
            or scope != getattr(
                self,
                "_directional_confirmed_scope",
                None,
            )
        ):
            return reject("SCOPE_MISMATCH")

        interaction_id = str(
            getattr(
                self,
                "_directional_confirmed_interaction_id",
                "",
            ) or ""
        ).strip()

        if not interaction_id:
            return reject("CONFIRMED_INTERACTION_MISSING")

        snapshot = getattr(
            self,
            "directional_identity_snapshot",
            None,
        )
        received_at = getattr(
            self,
            "directional_identity_received_at",
            None,
        )

        if snapshot is None or received_at is None:
            return reject("IDENTITY_SNAPSHOT_MISSING")

        max_age = float(self.social_decision_max_age_sec)

        if not math.isfinite(max_age) or max_age <= 0.0:
            return reject("INVALID_MAX_AGE")

        receipt_age = time.monotonic() - received_at

        if not 0.0 <= receipt_age <= max_age:
            return reject(
                f"IDENTITY_RECEIPT_STALE:"
                f"age={receipt_age:.3f},limit={max_age:.3f}"
            )

        if (
            str(snapshot.tracking_session_id).strip() != scope[3]
            or not str(snapshot.header.frame_id).strip()
        ):
            return reject("IDENTITY_SNAPSHOT_CONTEXT_INVALID")

        stamp = snapshot.header.stamp
        stamp_ns = (
            int(stamp.sec) * 1_000_000_000
            + int(stamp.nanosec)
        )

        source_age = (
            self.get_clock().now().nanoseconds - stamp_ns
        ) / 1e9

        if (
            stamp_ns <= 0
            or not math.isfinite(source_age)
            or not -0.1 <= source_age <= max_age
        ):
            return reject(
                f"IDENTITY_SOURCE_STALE:"
                f"age={source_age:.3f},limit={max_age:.3f}"
            )

        persons = [
            person
            for person in snapshot.persons
            if (
                str(person.id).strip() == scope[4]
                and str(person.tracking_session_id).strip()
                    == scope[3]
                and str(person.tracking_state).strip() == "TRACKED"
                and bool(person.observed_in_current_array)
                and int(person.identity_generation) > 0
            )
        ]

        if len(persons) != 1:
            return reject("IDENTITY_NOT_UNIQUELY_OBSERVED")

        request = DirectionalConstraintRequest()
        request.header.stamp = self.get_clock().now().to_msg()
        # This header identifies the robot-relative direction frame.
        # The social observation retains its own original frame.
        request.header.frame_id = self.directional_robot_frame

        request.request_id = f"DIR_{uuid.uuid4().hex}"
        request.navigation_goal_id = scope[0]
        request.interaction_id = interaction_id
        request.person_id = scope[4]
        request.tracking_session_id = scope[3]
        request.identity_generation = int(
            persons[0].identity_generation
        )

        request.action = "INSTALL"
        request.directional_constraint = direction
        request.validity_duration.sec = 30
        request.validity_duration.nanosec = 0

        self.get_logger().info(
            "[DIRECTIONAL MASK] Request prepared | "
            f"request={request.request_id} | "
            f"direction={direction} | "
            f"source_age={source_age:.3f} | "
            f"receipt_age={receipt_age:.3f} | "
            f"max_age={max_age:.3f}"
        )

        return request
    
    # --------------------------------------------------
    # ACTUAL ROS HRI CLOSE CALLBACK
    # --------------------------------------------------

    def _publish_directional_final_announcement(self):
        """Publish the confirmed side without initiating movement."""
        from g1_social_msgs.msg import ExplanationText

        finalization = self._directional_finalization

        if (
            finalization is None
            or finalization["phase"] != "READY_TO_ANNOUNCE"
            or self._social_scope_key() != finalization["scope"]
        ):
            return False

        candidate = finalization["candidate"]

        decision_id = (
            f"DIRECTION_FINAL_{uuid.uuid4().hex}"
        )

        message = ExplanationText()
        message.header.stamp = self.get_clock().now().to_msg()
        message.decision_id = decision_id

        message.language = self.language
        message.text = (
            self._directional_final_announcement_text(
                candidate
            )
        )

        message.fallback_language = "en"

        # English is constructed from the confirmed semantics,
        # not from an unverified free-form translation.
        original_language = self.language

        try:
            self.language = "en"
            message.fallback_text = (
                self._directional_final_announcement_text(
                    candidate
                )
            )
        finally:
            self.language = original_language

        message.provider = "g1_robot_collaboration"
        message.model = ""
        message.used_fallback = False
        message.used_cache = False
        message.speech_kind = "DIALOGUE"

        message.context_stamp = message.header.stamp
        message.navigation_goal_id = finalization["scope"][0]
        message.navigation_state = self.navigation_state
        message.navigation_active = False
        message.social_conflict_active = True

        finalization["announcement_id"] = decision_id
        finalization["phase"] = "ANNOUNCING"
        finalization["started_at"] = time.monotonic()

        self.directional_announcement_pub.publish(
            message
        )

        return True


    def _complete_directional_final_speech(self, msg):
        """
        Commit confirmed directional consent after the exact
        final announcement completes.

        Return True only if this speech status belongs to
        the directional finalization transaction.
        """
        finalization = getattr(
            self,
            "_directional_finalization",
            None,
        )

        if not isinstance(finalization, dict):
            return False

        if (
            finalization.get("phase") != "ANNOUNCING"
            or str(msg.decision_id or "").strip()
                != finalization.get("announcement_id")
        ):
            return False

        state = str(msg.state or "").strip().upper()

        if state not in {"COMPLETED", "FAILED", "CANCELED"}:
            return True

        if state != "COMPLETED":
            finalization["phase"] = "FAILED"
            self._directional_announcement_complete = False
            self._direction_semantic_ready = False

            self._begin_autonomous_readaptation_announcement(
                "HRI_TECHNICAL_FAILURE"
            )
            return True

        candidate = finalization.get("candidate")
        scope = self._social_scope_key()

        if (
            not isinstance(candidate, dict)
            or scope != finalization.get("scope")
            or scope != candidate.get("scope")
            or self._released
            or self._denied_scope
            or self.pending_terminal_stop is not None
            or candidate.get("final_confirmation_received") is not True
            or candidate.get("final_confirmation_request_id")
                != finalization.get("request_id")
            or candidate.get("interaction_id")
                != finalization.get("interaction_id")
        ):
            finalization["phase"] = "FAILED"
            self._begin_autonomous_readaptation_announcement(
                "HRI_TURN_UNAVAILABLE"
            )
            return True

        requested = self._normalize_direction(
            candidate.get("direction")
        )

        resolved = self._normalize_direction(
            candidate.get("resolved_robot_direction")
        )

        if (
            requested not in {"LEFT", "RIGHT"}
            or resolved not in {"LEFT", "RIGHT"}
        ):
            finalization["phase"] = "FAILED"
            self._begin_autonomous_readaptation_announcement(
                "HRI_TURN_UNAVAILABLE"
            )
            return True

        # Record the original human permission.
        self.confirmed_human_intent = "ALLOW_PASSAGE"
        self.confirmed_human_direction = requested
        self.dialogue_outcome = "ALLOW_CONFIRMED"
        self.authorized_episode_key = scope
        self.executed_social_scope = None

        person_scope = self._mode_d_person_scope(
            self.active_person_id
        )

        case = self._mode_d_case(person_scope)

        if case is not None:
            case["confirmed_intent"] = "ALLOW_PASSAGE"
            case["confirmed_direction"] = requested
            case["dialogue_outcome"] = "ALLOW_CONFIRMED"
            case["hri_closed"] = True
            
        # Preserve exact confirmation identity after HRI closure.
        self._directional_confirmed_scope = scope
        self._directional_confirmed_interaction_id = (
            finalization["interaction_id"]
        )

        self._clear_pending_human_preference()

        self.dialogue_retry_count = 0
        self.dialogue_speech_retry_count = 0

        # No additional confirmation gate is introduced.
        finalization["phase"] = "ANNOUNCED"
        self._directional_announcement_complete = True
        self._direction_semantic_ready = True

        self.get_logger().info(
            "[HRI DIRECTION] Confirmed and announced | "
            f"human={requested} | robot={resolved} | "
            "next=DIRECTIONAL_NAV2_HANDOFF"
        )

        self._release_navigation(requested)
        return True

    def _hri_dialogue_control_result_callback(self, msg):
        """Process exact HRI closure and preserve the chosen policy."""
        data = self._json(msg, "hri_manager")

        if not data:
            return

        finalization = getattr(
            self,
            "_directional_finalization",
            None,
        )

        if (
            finalization is not None
            and finalization.get("phase") == "CLOSING_HRI"
        ):
            data = self._json(msg, "hri_manager")

            pending = self.pending_hri_close

            if not data or pending is None:
                return

            if any(
                data.get(field) != pending.get(field)
                for field in (
                    "close_id",
                    "interaction_id",
                    "request_id",
                )
            ):
                return

            if (
                pending["interaction_id"]
                != finalization["interaction_id"]
                or pending["request_id"]
                != finalization["request_id"]
            ):
                return

            status = str(
                data.get("status") or ""
            ).strip().upper()

            if status != "CLOSED":
                self._publish_wait(
                    "DIRECTIONAL_HRI_CLOSURE_UNCONFIRMED"
                )

                finalization["phase"] = "FAILED"
                return

            if (
                self._social_scope_key()
                != finalization["scope"]
            ):
                finalization["phase"] = "FAILED"
                self._publish_wait(
                    "DIRECTIONAL_SOCIAL_SCOPE_CHANGED"
                )
                return

            self._detach_hri_turn(
                confirmed_closed=True
            )

            finalization["phase"] = "READY_TO_ANNOUNCE"

            self._publish_directional_final_announcement()
            return

        status = str(
            data.get("status") or ""
        ).strip().upper()

        next_stage = getattr(
            self,
            "_directional_next_stage",
            None,
        )

        if next_stage is not None:
            pending = self.pending_hri_close

            if pending is None:
                return

            if any(
                data.get(field) != pending.get(field)
                for field in (
                    "close_id",
                    "interaction_id",
                    "request_id",
                )
            ):
                return

            if str(data.get("status", "")).upper() != "CLOSED":
                self._directional_next_stage = None
                self._begin_autonomous_readaptation_announcement(
                    "HRI_CLOSE_UNCONFIRMED"
                )
                return

            candidate = self._direction_semantic_candidate
            interaction_id = pending["interaction_id"]
            scope = self._social_scope_key()

            if (
                candidate is None
                or candidate.get("scope") != scope
                or candidate.get("interaction_id") != interaction_id
            ):
                self._directional_next_stage = None
                self._begin_autonomous_readaptation_announcement(
                    "HRI_TURN_UNAVAILABLE"
                )
                return

            self._detach_hri_turn(confirmed_closed=True)

            # Preserve the interaction across its sequential HRI turns.
            self.active_interaction_id = interaction_id
            self._directional_next_stage = None

            if not self._request_directional_dialogue_turn(
                next_stage
            ):
                self._begin_autonomous_readaptation_announcement(
                    "HRI_TURN_UNAVAILABLE"
                )

            return

        detached = getattr(
            self,
            "_hri_detached_close",
            None,
        )

        if detached is not None and all(
            data.get(key) == detached[key]
            for key in (
                "close_id",
                "interaction_id",
                "request_id",
            )
        ):
            if status == "CLOSED":
                self._hri_detached_close = None
                self._try_silence_announcement()

            return

        pending = self.pending_hri_close

        if pending is None or any(
            data.get(key) != pending[key]
            for key in (
                "close_id",
                "interaction_id",
                "request_id",
            )
        ):
            return

        recovery = getattr(
            self,
            "_hri_retry_after_close",
            None,
        )

        retry_matches = bool(
            recovery is not None
            and recovery["interaction_id"]
                == pending["interaction_id"]
            and recovery["request_id"]
                == pending["request_id"]
        )

        outcome = self.dialogue_outcome
        direction = self.confirmed_human_direction

        permission = bool(
            outcome == "ALLOW_CONFIRMED"
            and self._permission_applies_to_current_episode()
        )

        # Preserve the interaction before clearing HRI ownership.
        # Both closure outcomes retain their original semantics.
        if permission and status in {
            "CLOSED",
            "TERMINATION_UNCONFIRMED",
        }:
            self._directional_confirmed_interaction_id = str(
                pending["interaction_id"]
            ).strip()
            self._directional_confirmed_scope = (
                self._social_scope_key()
            )

        if status == "TERMINATION_UNCONFIRMED":
            self._detach_hri_turn()

            if outcome == "DENY_CONFIRMED":
                self._hold_after_denial()

            elif permission:
                # Preserve intentional release after confirmed consent.
                self._release_navigation(direction)

            else:
                self._begin_autonomous_readaptation_announcement(
                    "HRI_CLOSE_UNCONFIRMED"
                )

            return

        if status != "CLOSED":
            return

        self._detach_hri_turn(confirmed_closed=True)

        if retry_matches:
            if (
                recovery["goal"] != self.active_navigation_goal_id
                or recovery["scope"] != self._scope
                or self._released
                or self.pending_terminal_stop is not None
            ):
                return

            self.active_dialogue_request_type = (
                recovery["request_type"]
            )
            self.dialogue_outcome = "PENDING"

            if recovery["initiative"]:
                self._request_passage_clarification()
            else:
                self._retry_active_dialogue_request()

            return

        if outcome == "DENY_CONFIRMED":
            self._hold_after_denial()

        elif permission:
            self._release_navigation(direction)

        elif self._scope and not self._released:
            self._begin_autonomous_readaptation_announcement(
                "HRI_CLOSED_WITHOUT_RESOLUTION"
            )

    def _on_hri_dialogue_control_result(self, msg):
        """Support the modular alternative callback name."""
        return self._hri_dialogue_control_result_callback(msg)

    # --------------------------------------------------
    # BOUNDED HRI WATCHDOG
    # --------------------------------------------------

    def _watch_hri_close(self, now):
        """Bound HRI recovery while preserving the current policy."""
        pending = self.pending_hri_close
        announcement_id = getattr(
            self,
            "_silence_announcement_id",
            "",
        )

        announcement_started = getattr(
            self,
            "_silence_announcement_started_at",
            None,
        )

        if (
            announcement_id
            and announcement_started is not None
            and now - announcement_started
                >= float(self.speech_completion_timeout_sec)
        ):
            self.get_logger().warning(
                "[HRI FALLBACK] Mode-C announcement timeout | "
                "continuing technical recovery"
            )

            self._silence_announcement_fault = True
            self._complete_silence_announcement()

        if pending is not None:
            last_sent = pending.get(
                "last_sent_at",
                pending["requested_at"],
            )

            if (
                pending.get("send_count", 1) < 3
                and now - last_sent >= 2.0
            ):
                self._send_pending_hri_close()

                pending["last_sent_at"] = now
                pending["send_count"] = (
                    pending.get("send_count", 1) + 1
                )

            if now - pending["requested_at"] >= 10.0:
                outcome = self.dialogue_outcome
                direction = self.confirmed_human_direction

                permission = bool(
                    outcome == "ALLOW_CONFIRMED"
                    and self._permission_applies_to_current_episode()
                )

                if permission:
                    self._directional_confirmed_interaction_id = str(
                        pending["interaction_id"]
                    ).strip()
                    self._directional_confirmed_scope = (
                        self._social_scope_key()
                    )

                self._detach_hri_turn()

                if outcome == "DENY_CONFIRMED":
                    self._hold_after_denial()

                elif permission:
                    # Keep the original timeout release policy.
                    self._release_navigation(direction)

                else:
                    self._begin_autonomous_readaptation_announcement(
                        "HRI_CLOSE_TIMEOUT"
                    )

            return

        detached = getattr(
            self,
            "_hri_detached_close",
            None,
        )

        if detached is not None:
            last_sent = detached.get(
                "last_sent_at",
                detached["requested_at"],
            )

            if (
                detached.get("send_count", 1) < 3
                and now - last_sent >= 2.0
            ):
                payload = {
                    key: detached[key]
                    for key in (
                        "close_id",
                        "interaction_id",
                        "request_id",
                        "reason",
                    )
                }

                payload.update(
                    version=1,
                    command="CLOSE",
                )

                self.hri_dialogue_control_pub.publish(
                    String(data=json.dumps(payload))
                )

                detached["last_sent_at"] = now
                detached["send_count"] = (
                    detached.get("send_count", 1) + 1
                )

            if now - detached["requested_at"] >= 10.0:
                self.get_logger().warning(
                    "[HRI CLOSE] Unconfirmed detached audio; "
                    "route planning remains independent."
                )

                self._hri_detached_close = None
                self._try_silence_announcement()

        if (
            not self._scope
            or self._released
            or self._denied_scope
            or self.pending_terminal_stop is not None
            or getattr(self, "_hri_nav_fallback_active", False)
            or self.pending_hri_close is not None
            or self.dialogue_started_at is None
            or not self.pending_dialogue_request_id
        ):
            return

        listening = (
            self.human_initiative_timeout_sec
            if self.waiting_for_human_initiative
            else self.dialogue_response_timeout_sec
        )

        speech_allowance = (
            0.0
            if self.waiting_for_human_initiative
            else (
                self.initial_speech_start_timeout_sec
                + self.speech_completion_timeout_sec
            )
        )

        deadline = min(
            60.0,
            listening + speech_allowance + 2.0,
        )

        if now - self.dialogue_started_at >= deadline:
            self._begin_autonomous_readaptation_announcement(
                "HRI_RESULT_MISSING"
            )

    # --------------------------------------------------
    # EXPLICIT HUMAN REFUSAL
    # --------------------------------------------------

    def _hold_after_denial(self):
        """
        Confirmed human refusal:
        retain G in WAIT and start one 180-second timer.

        Repeated refusal callbacks must not restart the timer.
        """
        if not self._scope:
            return

        now = time.monotonic()
        goal = str(
            self.active_navigation_goal_id or ""
        )

        same_refusal = bool(
            getattr(self, "_denial_wait_started_at", None)
                is not None
            and getattr(self, "_denial_wait_goal", "")
                == goal
            and getattr(self, "_denial_wait_scope", None)
                == self._scope
        )

        if not same_refusal:
            self._denial_wait_started_at = now
            self._denial_wait_goal = goal
            self._denial_wait_scope = self._scope

            self.get_logger().info(
                "[SOCIAL DENIAL] Timer started | "
                f"goal={goal} | "
                "deadline_sec=180.0 | "
                "next=FRESH_DIALOGUE"
            )

        self._denied_scope = self._scope
        self._handled.add(self._scope)

        self._silence_pending = False
        self._released = False
        self._nav_phase = "SOCIAL_DENIED"

        self._transition_to(
            STATE_WAITING,
            reason="HUMAN_DENIED_PASSAGE",
        )

        self._publish_wait(
            "HUMAN_DENIED_PASSAGE"
        )
        
    # --------------------------------------------------
    # AUTONOMOUS HRI FALLBACK
    # --------------------------------------------------
    def _spatial_response_is_ambiguous(self, msg):
        """
        Detect an unresolved spatial instruction.

        Never correct ASR words automatically.
        A direction absent from a spatial expression is
        not equivalent to an unrestricted permission.
        """
        import re

        text = " ".join(
            str(msg.utterance or "").lower().split()
        )

        intent = str(
            msg.intent or ""
        ).strip().upper()

        if intent not in {
            "CONFIRM",
            "ALLOW_PASSAGE",
            "REQUEST_LEFT",
            "REQUEST_RIGHT",
        }:
            return False

        left = bool(re.search(
            r"\b(left|gauche|izquierda)\b",
            text,
        ))

        right = bool(re.search(
            r"\b(right|droite|derecha)\b",
            text,
        ))

        parsed = self._normalize_direction(
            msg.direction
        )

        # Contradictory directions.
        if left and right:
            return True

        # A direction is mentioned but the interpreter
        # has not retained it correctly.
        if left and parsed not in {"LEFT"}:
            return True

        if right and parsed not in {"RIGHT"}:
            return True

        # Suspicious spatial expression:
        # "on the lift", "to my ...", "sur la ...", etc.
        # The side cannot be established reliably.
        spatial_expression = bool(re.search(
            r"\b("
            r"on\s+(?:the|my|your)\s+\w+"
            r"|to\s+(?:the|my|your)\s+\w+"
            r"|(?:my|your)\s+side"
            r"|(?:à|sur)\s+(?:ma|ta|votre|la)\s+\w+"
            r"|(?:a|por)\s+(?:la|mi|tu|su)\s+\w+"
            r")\b",
            text,
        ))

        if spatial_expression and not (left or right):
            return True

        return False
    
    def _begin_autonomous_readaptation_announcement(
        self,
        reason,
    ):
        """
        Start announced Mode-C recovery.

        Silence and technical HRI failure use the same
        autonomous recovery, with different messages.

        An explicit human NO remains a refusal and
        cannot enter this fallback.

        Navigation remains subject to the ordinary
        stop, HRI closure, mask revocation and Gate checks.
        """
        reason = str(
            reason or "HRI_UNAVAILABLE"
        ).strip().upper()

        if (
            not self._scope
            or self._released
            or self.pending_terminal_stop is not None
            or not self.active_navigation_goal_id
        ):
            return False

        if (
            self._denied_scope
            or self.dialogue_outcome == "DENY_CONFIRMED"
            or self.confirmed_human_intent == "DENY_PASSAGE"
        ):
            return False

        # A valid human permission belongs to its own
        # navigation handoff, not to autonomous fallback.
        if self.dialogue_outcome == "ALLOW_CONFIRMED":
            return False

        if getattr(
            self,
            "_hri_nav_fallback_active",
            False,
        ):
            return False

        technical_reasons = {
            "INITIAL_SPEECH_UNAVAILABLE",
            "HRI_TECHNICAL_FAILURE",
            "HRI_CLOSE_UNCONFIRMED",
            "HRI_CLOSE_TIMEOUT",
            "HRI_RESULT_MISSING",
            "HRI_TURN_UNAVAILABLE",
            "HRI_CLOSE_UNAVAILABLE",
            "HRI_RETRY_UNAVAILABLE",
        }

        if reason == "NO_HUMAN_RESPONSE":
            cause = "NO_USABLE_HUMAN_RESPONSE"

        elif reason == "UNUSABLE_HUMAN_RESPONSE":
            cause = "HUMAN_RESPONSE_UNCLEAR"

        elif reason == "AUTONOMOUS_RESUME_AFTER_WAIT_TIMEOUT":
            cause = "GLOBAL_WAIT_TIMEOUT"

        elif reason in technical_reasons:
            cause = "HRI_TECHNICAL_FAILURE"

        else:
            cause = "HRI_UNRESOLVED"

        # Reserve the exact recovery transaction.
        self._hri_nav_fallback_active = True
        self._hri_fallback_reason = reason
        self._hri_fallback_cause = cause
        self._hri_fallback_goal = (
            self.active_navigation_goal_id
        )
        self._hri_fallback_scope = self._scope

        self.get_logger().warning(
            "[HRI FALLBACK] Started | "
            f"goal={self._hri_fallback_goal} | "
            f"reason={reason} | "
            f"cause={cause} | "
            "navigation_authorized=False"
        )

        # Clear unconfirmed human permission.
        if cause == "HUMAN_RESPONSE_UNCLEAR":
            self.dialogue_outcome = "NOT_UNDERSTOOD"

        elif cause == "HRI_TECHNICAL_FAILURE":
            self.dialogue_outcome = "HRI_TECHNICAL_FAILURE"

        else:
            self.dialogue_outcome = "NO_RESPONSE"

        self.confirmed_human_intent = "UNKNOWN"
        self.confirmed_human_direction = "NONE"
        self.authorized_episode_key = None

        self._direction_semantic_ready = False
        self._directional_next_stage = None
        self._directional_finalization = None
        self._directional_announcement_complete = False

        self._hri_retry_after_close = None
        self.dialogue_started_at = None

        self._clear_pending_human_preference()

        # Preserve the existing HRI closure protocol.
        self._detach_hri_turn()

        # Select autonomous social navigation without
        # changing the launch-wide interaction parameter.
        self._episode_mode = "C"
        self._direction = "NONE"

        self._silence_announcement_fault = False

        # The announcement is required for BOTH human
        # silence and technical failure.
        self._silence_pending = True

        self.get_logger().info(
            "[HRI FALLBACK] Announcement requested | "
            f"cause={cause} | "
            "next=MODE_C_AFTER_ANNOUNCEMENT"
        )

        # If the old HRI turn is still closing,
        # _try_silence_announcement() will defer.
        self._try_silence_announcement()

        return True
    
    def _freeze_after_audio_fault(self, reason):
        """Escalate an HRI audio failure through the existing fallback."""
        self.get_logger().warning(
            "[HRI AUDIO] Unavailable | "
            f"reason={reason} | fallback=REPLAN"
        )

        self._begin_autonomous_readaptation_announcement(
            "HRI_TECHNICAL_FAILURE"
        )

    # --------------------------------------------------
    # BEST-EFFORT FINAL ANNOUNCEMENT
    # --------------------------------------------------

    def _try_silence_announcement(self):
        """
        Announce the exact autonomous recovery cause.

        Only the wording differs between human silence
        and technical HRI failure.

        Never publish a second announcement for the same
        recovery transaction.
        """
        from g1_social_msgs.msg import ExplanationText

        if not self._silence_pending:
            return False

        if (
            not self._scope
            or self._released
            or self._denied_scope
            or self.pending_terminal_stop is not None
        ):
            self._silence_pending = False
            return False

        if (
            self.dialogue_outcome == "DENY_CONFIRMED"
            or self.confirmed_human_intent == "DENY_PASSAGE"
        ):
            self._silence_pending = False
            return False

        if (
            self._hri_fallback_goal
                != self.active_navigation_goal_id
            or self._hri_fallback_scope != self._scope
        ):
            self._silence_pending = False
            return False

        # Never overlap the previous HRI speech/turn.
        if getattr(
            self,
            "_hri_detached_close",
            None,
        ) is not None:
            return False

        if self._silence_announcement_id:
            return False

        language = str(
            self.language or "en"
        ).strip().lower()[:2]

        if language not in {"fr", "en", "es"}:
            language = "en"

        reason = str(
            self._hri_fallback_reason or ""
        ).strip().upper()

        cause = str(
            self._hri_fallback_cause or ""
        ).strip().upper()

        if cause == "HRI_TECHNICAL_FAILURE":
            texts = {
                "fr": (
                    "Un problème technique empêche de "
                    "poursuivre notre dialogue. Je vais "
                    "rechercher un autre itinéraire, "
                    "en respectant les distances de sécurité."
                ),
                "en": (
                    "A technical problem prevents me from "
                    "continuing our dialogue. I will look "
                    "for another route while maintaining "
                    "safe distances."
                ),
                "es": (
                    "Un problema técnico impide continuar "
                    "nuestro diálogo. Buscaré otra ruta "
                    "manteniendo las distancias de seguridad."
                ),
            }

        elif reason == "NO_HUMAN_RESPONSE":
            texts = {
                "fr": (
                    "Je n'ai pas reçu de réponse. "
                    "Je vais rechercher un autre itinéraire "
                    "en tenant compte de votre présence et "
                    "en respectant une distance de sécurité."
                ),
                "en": (
                    "I did not receive an answer. "
                    "I will look for another route, "
                    "taking your presence into account "
                    "and maintaining a safe distance."
                ),
                "es": (
                    "No he recibido respuesta. "
                    "Buscaré otra ruta teniendo en cuenta "
                    "su presencia y manteniendo "
                    "una distancia segura."
                ),
            }

        elif reason == "UNUSABLE_HUMAN_RESPONSE":
            texts = {
                "fr": (
                    "Je n'ai pas pu comprendre suffisamment "
                    "votre réponse. Je vais rechercher un "
                    "autre itinéraire en respectant "
                    "les distances de sécurité."
                ),
                "en": (
                    "I could not understand your answer "
                    "well enough. I will look for another "
                    "route while maintaining safe distances."
                ),
                "es": (
                    "No he podido comprender bien su respuesta. "
                    "Buscaré otra ruta manteniendo "
                    "las distancias de seguridad."
                ),
            }

        else:
            texts = {
                "fr": (
                    "Je n'ai pas pu finaliser notre échange. "
                    "Je vais reprendre la recherche d'un "
                    "itinéraire adapté en respectant "
                    "les distances de sécurité."
                ),
                "en": (
                    "I could not complete our exchange. "
                    "I will resume searching for a suitable "
                    "route while maintaining safe distances."
                ),
                "es": (
                    "No he podido finalizar nuestro diálogo. "
                    "Volveré a buscar una ruta adecuada "
                    "manteniendo las distancias de seguridad."
                ),
            }

        decision_id = (
            f"HRI_AUTONOMOUS_RESUME_{uuid.uuid4().hex}"
        )

        message = ExplanationText()

        message.header.stamp = (
            self.get_clock().now().to_msg()
        )

        message.decision_id = decision_id
        message.language = language
        message.text = texts[language]

        message.fallback_language = "en"
        message.fallback_text = texts["en"]

        message.provider = "g1_robot_collaboration"
        message.model = ""

        message.used_fallback = False
        message.used_cache = False
        message.speech_kind = "DIALOGUE"

        message.context_stamp = message.header.stamp
        message.navigation_goal_id = (
            self.active_navigation_goal_id
        )
        message.navigation_state = self.navigation_state
        message.navigation_active = False
        message.social_conflict_active = True

        # Reserve speech ownership before publication.
        self._silence_announcement_id = decision_id
        self._silence_announcement_started_at = (
            time.monotonic()
        )

        self._silence_announcement_fault = False
        self._silence_pending = False

        try:
            self.directional_announcement_pub.publish(
                message
            )

        except Exception as exc:
            self.get_logger().error(
                "[HRI ANNOUNCEMENT] Publication failed | "
                f"decision={decision_id} | "
                f"error={exc}"
            )

            self._silence_announcement_fault = True
            self._complete_silence_announcement()
            return False

        self.get_logger().info(
            "[HRI ANNOUNCEMENT] Published | "
            f"decision={decision_id} | "
            f"cause={cause} | "
            f"goal={self.active_navigation_goal_id} | "
            "navigation_authorized=False | "
            "waiting_for_speech_completion=True"
        )

        return True
    
    def _complete_silence_announcement(self):
        """
        Continue Mode C after the announcement has ended,
        or when its audio transaction cannot complete.
        """
        announcement_id = self._silence_announcement_id

        self._silence_announcement_id = ""
        self._silence_announcement_started_at = None
        self._silence_pending = False

        if (
            self._hri_fallback_goal != self.active_navigation_goal_id
            or self._hri_fallback_scope != self._scope
            or self._released
            or self._denied_scope
            or self.pending_terminal_stop is not None
        ):
            return

        self.get_logger().info(
            "[HRI FALLBACK] Mode-C announcement ended | "
            f"announcement={announcement_id} | "
            f"audio_fault={self._silence_announcement_fault} | "
            "next=AUTONOMOUS_ROUTE_SEARCH"
        )

        self._release_navigation("NONE")

    # --------------------------------------------------
    # NAVIGATION HANDOFF
    # --------------------------------------------------

    def _release_navigation(self, direction):
        """
        Prepare the Nav2 handoff without losing directional consent.

        Mode D:
            - preserve the confirmed robot-relative direction;
            - require an exact directional mask;
            - never silently fall back to Mode C;
            - delegate mask failures to the bounded D recovery.

        Mode C:
            - require the ordinary autonomous authorization;
            - remove any previous directional mask before handoff.

        The public navigation goal is never replaced here.
        """
        if (
            self._released
            or not self._scope
            or self._denied_scope
            or self.pending_terminal_stop is not None
            or not self.active_navigation_goal_id
            or self.active_navigation_goal_pose is None
            or self._control is not None
        ):
            return False

        direction = self._normalize_direction(direction)

        mode = str(
            self._episode_mode or ""
        ).strip().upper()

        fallback = bool(
            getattr(
                self,
                "_hri_nav_fallback_active",
                False,
            )
        )

        permission = (
            self._permission_applies_to_current_episode()
        )

        # --------------------------------------------------
        # 1. HUMAN PERMISSION AND DIRECTION SEMANTICS
        # --------------------------------------------------

        if permission:
            requested = self._normalize_direction(
                self.confirmed_human_direction
            )

            if direction != requested:
                return False

            if requested in {"LEFT", "RIGHT"}:
                finalization = getattr(
                    self,
                    "_directional_finalization",
                    None,
                )

                candidate = (
                    finalization.get("candidate")
                    if isinstance(finalization, dict)
                    else None
                )

                if not (
                    isinstance(candidate, dict)
                    and finalization.get("phase") == "ANNOUNCED"
                    and getattr(
                        self,
                        "_directional_announcement_complete",
                        False,
                    )
                    and getattr(
                        self,
                        "_direction_semantic_ready",
                        False,
                    )
                    and candidate.get(
                        "final_confirmation_received"
                    ) is True
                    and candidate.get("direction") == requested
                    and candidate.get("scope") == self._scope
                    and finalization.get("scope") == self._scope
                    and candidate.get("interaction_id")
                        == finalization.get("interaction_id")
                    and candidate.get(
                        "final_confirmation_request_id"
                    ) == finalization.get("request_id")
                ):
                    self.get_logger().warning(
                        "[DIRECTIONAL HANDOFF] "
                        "Semantic confirmation not finalized"
                    )
                    return False

                resolved = str(
                    candidate.get(
                        "resolved_robot_direction",
                        "",
                    )
                ).strip().upper()

                reference = str(
                    candidate.get("reference", "")
                ).strip().upper()

                relation = str(
                    candidate.get("spatial_relation", "")
                ).strip().upper()

                if reference == "PERSON":
                    if relation != "FACE_TO_FACE_ASSUMED":
                        return False

                    expected = (
                        "RIGHT"
                        if requested == "LEFT"
                        else "LEFT"
                    )

                elif reference == "ROBOT":
                    if relation != "ROBOT_REFERENCE":
                        return False

                    expected = requested

                else:
                    return False

                if resolved != expected:
                    return False

                # Convert the human instruction once.
                direction = resolved
                mode = "D"

                self.get_logger().info(
                    "[DIRECTIONAL RESOLUTION] "
                    f"human_direction={requested} | "
                    f"reference={reference} | "
                    f"robot_direction={direction} | "
                    f"relation={relation}"
                )

            else:
                direction = "NONE"
                mode = "C"

            self.authorized_episode_key = self._scope
            self._directional_confirmed_scope = self._scope

        elif (
            fallback
            or (
                mode == "C"
                and self.interaction != "dialogue"
            )
            or (
                mode == "C"
                and getattr(
                    self,
                    "_adaptation_c_scope",
                    None,
                ) == self._scope
            )
        ):
            direction = "NONE"
            mode = "C"

        else:
            return False

        # --------------------------------------------------
        # 2. MODE D — MANDATORY DIRECTIONAL MASK
        # --------------------------------------------------

        if mode == "D":
            if direction not in {"LEFT", "RIGHT"}:
                return False

            publisher = getattr(
                self,
                "directional_constraint_pub",
                None,
            )

            # Keep any matching INSTALL transaction.
            # Do not revoke it merely because a handoff
            # is being retried.
            current_request = getattr(
                self,
                "_directional_request",
                None,
            )

            current_valid = bool(
                current_request is not None
                and str(current_request.action).upper()
                    == "INSTALL"
                and str(
                    current_request.navigation_goal_id
                ) == str(self.active_navigation_goal_id)
                and str(
                    current_request.directional_constraint
                ).upper() == direction
                and str(current_request.person_id)
                    == str(self.active_person_id)
                and str(
                    current_request.tracking_session_id
                ) == str(self._scope[3])
            )

            if current_request is not None and not current_valid:
                if not self._revoke_optional_directional_mask(
                    "REPLACE_DIRECTIONAL_REQUEST"
                ):
                    return False

                current_request = None

            request = (
                current_request
                if current_valid
                else self._prepare_directional_mask_request(
                    direction
                )
            )

            if request is None or publisher is None:
                reason = (
                    "DIRECTIONAL_PUBLISHER_UNAVAILABLE"
                    if publisher is None
                    else "DIRECTIONAL_IDENTITY_OR_MASK_UNAVAILABLE"
                )

                self.get_logger().warning(
                    "[MODE D] Directional request unavailable | "
                    f"goal={self.active_navigation_goal_id} | "
                    f"direction={direction} | "
                    f"reason={reason} | "
                    "fallback_to_c=False"
                )

                # Do not change direction to NONE.
                # Retain Mode D and let its bounded
                # recovery supervisor process the failure.
                self._episode_mode = "D"
                self._direction = direction
                self._directional_retry_pending = True

                self._released = True
                self._nav_phase = "DIRECT_NAV2_WAIT_MASK"

                self._drive_directional_attempts(reason)
                return False

            if not current_valid:
                try:
                    publisher.publish(request)

                except Exception as exc:
                    self.get_logger().error(
                        "[MODE D] INSTALL publication failed | "
                        f"error={exc} | "
                        "fallback_to_c=False"
                    )

                    self._episode_mode = "D"
                    self._direction = direction
                    self._directional_retry_pending = True

                    self._released = True
                    self._nav_phase = (
                        "DIRECT_NAV2_WAIT_MASK"
                    )

                    self._drive_directional_attempts(
                        "OPTIONAL_MASK_INSTALL_FAILED"
                    )
                    return False

                self._directional_request = request
                self._directional_status = None

                self._directional_last_renewal_at = (
                    time.monotonic()
                )

            self._episode_mode = "D"

        # --------------------------------------------------
        # 3. MODE C — REMOVE ANY OLD D MASK FIRST
        # --------------------------------------------------

        elif mode == "C":
            if direction != "NONE":
                return False

            if not self._revoke_optional_directional_mask(
                "MODE_C_NAVIGATION_HANDOFF"
            ):
                return False

            self._episode_mode = "C"

        else:
            return False

        # --------------------------------------------------
        # 4. PREPARE THE ORIGINAL NAV2 HANDOFF
        # --------------------------------------------------

        if not self._begin_direct_nav2_handoff(direction):
            return False

        self._directional_retry_pending = False

        self._handled.add(self._scope)
        self._attempts = 0
        self._assisted = False
        self._next_attempt = time.monotonic()

        self._transition_to(
            STATE_REPLANNING,
            reason="SOCIAL_EPISODE_NAV2_HANDOFF_PREPARED",
        )

        self.get_logger().info(
            "[SOCIAL RELEASE] Direct handoff prepared | "
            f"goal={self.active_navigation_goal_id} | "
            f"mode={self._episode_mode} | "
            f"human_direction="
            f"{self.confirmed_human_direction} | "
            f"effective_direction={self._direction} | "
            f"mask_required={mode == 'D'} | "
            "navigation_started=False"
        )

        return True
    



