"""Optional negotiated recovery layered over the current C/D implementation."""

import json
import math
import time
import uuid
import copy

from std_msgs.msg import String
from g1_social_msgs.msg import DialogueRequest, ExplanationText
from g1_social_common.collaboration_states import STATE_WAITING


class NegotiatedAdaptationMixin:
    """Preserve the original behavior when negotiated recovery is disabled."""

    def __init__(self):
        self._adaptation_enabled = False
        self._adaptation = None
        self._adaptation_c_scope = None
        self._adaptation_start_id = ""
        self._directional_attempt_state = None

        super().__init__()

        self.declare_parameter(
            "negotiated_adaptation_enabled",
            False,
        )

        configured = bool(
            self.get_parameter(
                "negotiated_adaptation_enabled"
            ).value
        )

        # Mode D always owns negotiated directional recovery.
        # Mode C remains autonomous and independent.
        self._adaptation_enabled = bool(
            configured or self.interaction == "dialogue"
        )

        self._adaptation_speech_pub = self.create_publisher(
            ExplanationText,
            "/explainability/text",
            10,
        )
        
    def _command(self, action, reason, path_id=""):
        """Use the shared Gate contract and retain START identity."""
        action = str(action or "").strip().upper()
        reason = str(reason or "").strip().upper()
        path_id = str(path_id or "").strip()

        if action not in {"START", "WAIT", "REPLAN", "STOP"}:
            return ""

        if path_id:
            return ""

        if action == "START":
            if self._adaptation is not None:
                return ""

            if (
                self._scope
                and self._episode_mode == "D"
                and not self._direct_directional_ready()
            ):
                return ""

        request_id = super()._command(
            action,
            reason,
            path_id,
        )

        if request_id and action == "START":
            mode = (
                "INITIAL"
                if not self._scope
                else str(
                    self._episode_mode or ""
                ).strip().upper()
            )

            self._nav2_last_start = {
                "goal": self.active_navigation_goal_id,
                "request_id": request_id,
                "mode": mode,
                "scope": self._scope,
                "robot_direction": (
                    "NONE"
                    if mode in {"INITIAL", "C"}
                    else str(
                        self._direction or "NONE"
                    ).strip().upper()
                ),
                "human_direction": str(
                    self.confirmed_human_direction or "NONE"
                ).strip().upper(),
            }

            self._adaptation_start_id = request_id

        return request_id

    def _fresh_reinstall_request(self, previous):
        """
        Create a fresh INSTALL for the same authorized person.

        A retry does not copy permission to a new identity.
        New geometry is still required by the mask node.
        """
        request = copy.deepcopy(previous)
        request.request_id = "DIR_" + uuid.uuid4().hex
        request.action = "INSTALL"
        request.header.stamp = self.get_clock().now().to_msg()
        return request
    
    def _after_confirmed_no_to_c(self, recovery):
        """
        Human refused C, not the original LEFT/RIGHT.

        Perform exactly one final D installation.
        """
        old = self._directional_request
        tx = self._directional_attempt_state

        if old is None and isinstance(tx, dict):
            old = tx.get("old")

        direction = str(
            self._direction or ""
        ).strip().upper()

        if direction not in {"LEFT", "RIGHT"}:
            return False

        self._directional_attempt_state = {
            "key": (
                self.active_navigation_goal_id,
                self._scope,
                direction,
            ),
            "count": 2,
            "phase": "CLEAR",
            "started": time.monotonic(),
            "old": copy.deepcopy(old),
            "revoke_sent": False,
        }

        if (
            str(recovery.get("reason") or "").upper()
            == "TWO_DIRECTIONAL_ROUTE_FAILURES"
        ):
            self._d_route_after_c_refusal = True
            self._d_route_after_c_refusal_goal = (
                self.active_navigation_goal_id
            )

            # The existing D mask remains installed.
            # No REVOKE, no new INSTALL.
            self._directional_attempt_state = None
            self._adaptation = None
            self._denied_scope = None

            self.dialogue_outcome = "ALLOW_CONFIRMED"
            self._episode_mode = "D"
            self._released = True
            self._control = None
            self._nav_phase = "DIRECT_NAV2_WAIT_MASK"

            return True

        self._adaptation = None
        self._denied_scope = None

        # Restore the original confirmed D authorization.
        # This is NOT an authorization to navigate in C.
        self.dialogue_outcome = "ALLOW_CONFIRMED"

        self._episode_mode = "D"
        self._released = True
        self._nav_phase = "DIRECT_NAV2_WAIT_MASK"
        self._control = None

        return True

    def _announce_final_directional_failure(self):
        """Announce exhaustion of D without terminating G."""
        speech = ExplanationText()

        speech.header.stamp = (
            self.get_clock().now().to_msg()
        )
        speech.context_stamp = speech.header.stamp

        speech.decision_id = (
            "D_FINAL_" + uuid.uuid4().hex
        )

        texts = {
            "fr": (
                "Je ne parviens toujours pas à suivre "
                "la direction demandée. Je reste arrêté "
                "et reprendrai notre échange."
            ),
            "en": (
                "I still cannot follow the requested direction. "
                "I will stay stopped and ask again."
            ),
            "es": (
                "Todavía no puedo seguir la dirección solicitada. "
                "Permaneceré detenido y volveré a preguntar."
            ),
        }

        language = str(
            self.language or "en"
        ).lower()[:2]

        speech.language = (
            language if language in texts else "en"
        )
        speech.text = texts[speech.language]

        speech.fallback_language = "en"
        speech.fallback_text = texts["en"]

        speech.provider = "deterministic"
        speech.model = "directional_recovery"
        speech.speech_kind = "SOCIAL_EXPLANATION"

        speech.navigation_goal_id = (
            self.active_navigation_goal_id
        )
        speech.navigation_state = "WAITING"
        speech.navigation_active = False
        speech.social_conflict_active = True

        self._adaptation_speech_pub.publish(speech)

    def _watch_directional_constraint(self, now):
        """Treat D as mandatory and C as unconstrained."""
        if self._episode_mode != "D":
            return super()._watch_directional_constraint(now)

        if self._adaptation is not None:
            return False

        if self._direct_directional_ready():
            return super()._watch_directional_constraint(now)

        return self._drive_directional_attempts(
            "DIRECTIONAL_CONSTRAINT_LOST"
        )
    
    def _direct_directional_ready(self):
        """
        Require exact directional authorization and actual
        confirmation from both Nav2 costmaps.

        The human-requested direction and robot-resolved
        direction are checked independently.
        """
        if not self._adaptation_enabled:
            return super()._direct_directional_ready()

        request = getattr(
            self,
            "_directional_request",
            None,
        )

        status = getattr(
            self,
            "_directional_status",
            None,
        )

        scope = self._social_scope_key()

        if (
            request is None
            or status is None
            or not scope
            or len(scope) != 5
        ):
            return False

        identity_fields = (
            "request_id",
            "navigation_goal_id",
            "interaction_id",
            "person_id",
            "tracking_session_id",
            "identity_generation",
            "directional_constraint",
        )

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

        if not isinstance(candidate, dict):
            return False

        requested = str(
            candidate.get("direction", "")
        ).strip().upper()

        reference = str(
            candidate.get("reference", "")
        ).strip().upper()

        resolved = str(
            candidate.get(
                "resolved_robot_direction",
                "",
            )
        ).strip().upper()

        relation = str(
            candidate.get(
                "spatial_relation",
                "",
            )
        ).strip().upper()

        if (
            requested not in {"LEFT", "RIGHT"}
            or reference not in {"PERSON", "ROBOT"}
            or resolved not in {"LEFT", "RIGHT"}
        ):
            return False

        # Independently verify the spatial conversion.
        if reference == "PERSON":
            if relation != "FACE_TO_FACE_ASSUMED":
                return False

            expected = (
                "RIGHT" if requested == "LEFT"
                else "LEFT"
            )

        else:
            if relation != "ROBOT_REFERENCE":
                return False

            expected = requested

        if resolved != expected:
            return False

        return bool(
            self._direction in {"LEFT", "RIGHT"}

            # Exact directional request identity.
            and request.action == "INSTALL"
            and request.navigation_goal_id == scope[0]
            and request.tracking_session_id == scope[3]
            and request.person_id == scope[4]
            and request.directional_constraint == self._direction

            # Preserve the original human request.
            and self.confirmed_human_direction == requested

            # Match the direction actually used by the robot.
            and self._direction == resolved

            # Require complete semantic confirmation.
            and candidate.get(
                "final_confirmation_received"
            ) is True
            and finalization.get("phase") == "ANNOUNCED"
            and finalization.get("scope") == scope
            and candidate.get("scope") == scope
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

            # Preserve exact social permission.
            and self._permission_applies_to_current_episode()

            # Preserve the exact request/status identity.
            and all(
                getattr(request, field)
                == getattr(status, field)
                for field in identity_fields
            )

            # Require actual mask application.
            and status.state in {"PUBLISHED", "ACTIVE"}
            and bool(str(status.mask_id).strip())
            and status.global_costmap_confirmed
            and status.local_costmap_confirmed

            # Reject stale status information.
            and self._message_is_recent(status, 1.0)
        )

    def _drive_directional_attempts(self, reason):
        """
        Two attempts maximum to obtain a directional mask.

        An existing published INSTALL counts as attempt 1.
        Every additional INSTALL uses a new request_id.

        Preparation/publication failures are tracked separately
        and cannot create an infinite retry loop.

        No automatic Mode-C fallback:
        after two failures, start the adaptation question.

        No Nav2 START without a validated directional mask.
        """
        if self._adaptation is not None:
            return False

        if (
            str(self._episode_mode).upper() != "D"
            or not self._scope
            or not self.active_navigation_goal_id
            or self.pending_terminal_stop is not None
        ):
            return False

        direction = str(
            self._direction or "NONE"
        ).strip().upper()

        if direction not in {"LEFT", "RIGHT"}:
            return False

        now = time.monotonic()

        key = (
            self.active_navigation_goal_id,
            self._scope,
            direction,
        )

        request = getattr(
            self, "_directional_request", None
        )

        tx = self._directional_attempt_state

        if not isinstance(tx, dict) or tx.get("key") != key:
            existing_install = bool(
                request is not None
                and str(request.action).upper() == "INSTALL"
                and request.navigation_goal_id
                    == self.active_navigation_goal_id
                and str(
                    request.directional_constraint
                ).upper() == direction
            )

            tx = {
                "key": key,
                "count": 1 if existing_install else 0,
                "phase": (
                    "INSTALL"
                    if existing_install
                    else "PREPARE"
                ),
                "started": now,
                "request_id": (
                    str(request.request_id)
                    if existing_install
                    else ""
                ),
                "old": None,
                "revoke_sent": False,
                "prepare_failures": 0,
            }

            self._directional_attempt_state = tx

        if tx["phase"] == "AWAIT_C_DECISION":
            return False

        if self._direct_directional_ready():
            self._directional_attempt_state = None
            return True

        # Physical hold is mandatory before changing masks.
        if (
            not self.nav2_cancel_confirmed
            or not self._stopped()
        ):
            if self._control is None:
                self._command(
                    "WAIT",
                    "DIRECTIONAL_INSTALL_HOLD",
                )

            self._nav_phase = "DIRECT_NAV2_WAIT_MASK"
            return False

        if self._control is not None:
            return False

        def begin_adaptation():
            tx["phase"] = "AWAIT_C_DECISION"

            self.get_logger().warning(
                "[MODE D MASK] Two attempts exhausted | "
                f"goal={self.active_navigation_goal_id} | "
                f"published_installs={tx['count']} | "
                f"preparation_failures="
                f"{tx['prepare_failures']} | "
                "next=ASK_MODE_C_PERMISSION"
            )

            self._start_adaptation(
                "TWO_DIRECTIONAL_INSTALLATIONS_FAILED"
            )

            return False

        def prepare_and_publish():
            current = getattr(
                self, "_directional_request", None
            )

            if current is not None:
                # Never overwrite an unrevoked request.
                return False

            publisher = getattr(
                self, "directional_constraint_pub", None
            )

            new_request = None

            if publisher is not None:
                new_request = (
                    self._prepare_directional_mask_request(
                        direction
                    )
                )

            if new_request is None:
                tx["prepare_failures"] += 1
                tx["started"] = now

                self.get_logger().warning(
                    "[MODE D MASK] INSTALL preparation failed | "
                    f"attempt={tx['count'] + 1}/2 | "
                    f"reason={reason}"
                )

                if (
                    tx["count"] + tx["prepare_failures"]
                    >= 2
                ):
                    return begin_adaptation()

                tx["phase"] = "PREPARE"
                return False

            if str(new_request.action).upper() != "INSTALL":
                tx["prepare_failures"] += 1

                if (
                    tx["count"] + tx["prepare_failures"]
                    >= 2
                ):
                    return begin_adaptation()

                tx["started"] = now
                return False

            # Count a submission attempt exactly once.
            # A publish exception is not recorded as success.
            try:
                publisher.publish(new_request)

            except Exception as exc:
                tx["prepare_failures"] += 1
                tx["started"] = now

                self.get_logger().error(
                    "[MODE D MASK] INSTALL publish failed | "
                    f"error={exc}"
                )

                if (
                    tx["count"] + tx["prepare_failures"]
                    >= 2
                ):
                    return begin_adaptation()

                tx["phase"] = "PREPARE"
                return False

            tx["count"] += 1
            tx["phase"] = "INSTALL"
            tx["started"] = now
            tx["request_id"] = str(new_request.request_id)

            self._directional_request = new_request
            self._directional_status = None
            self._directional_last_renewal_at = now

            self.get_logger().info(
                "[MODE D MASK] INSTALL published | "
                f"attempt={tx['count']}/2 | "
                f"request={new_request.request_id}"
            )

            return False

        # ----------------------------------------------
        # PREPARE NEXT INSTALL
        # ----------------------------------------------
        if tx["phase"] == "PREPARE":
            if now - tx["started"] < 1.0:
                return False

            return prepare_and_publish()

        # ----------------------------------------------
        # REVOKE FAILED ATTEMPT 1
        # ----------------------------------------------
        if tx["phase"] == "CLEAR":
            old = tx.get("old")

            if old is not None:
                if not tx["revoke_sent"]:
                    publisher = getattr(
                        self,
                        "directional_constraint_pub",
                        None,
                    )

                    if publisher is None:
                        return False

                    revoke = copy.deepcopy(old)
                    revoke.action = "REVOKE"
                    revoke.header.stamp = (
                        self.get_clock().now().to_msg()
                    )

                    try:
                        publisher.publish(revoke)
                    except Exception as exc:
                        self.get_logger().error(
                            "[MODE D MASK] REVOKE failed | "
                            f"error={exc}"
                        )
                        return False

                    tx["revoke_sent"] = True
                    tx["started"] = now
                    return False

                status = self._directional_status

                fields = (
                    "request_id",
                    "navigation_goal_id",
                    "interaction_id",
                    "person_id",
                    "tracking_session_id",
                    "identity_generation",
                    "directional_constraint",
                )

                exact = bool(
                    status is not None
                    and all(
                        getattr(status, field) ==
                        getattr(old, field)
                        for field in fields
                    )
                )

                revoked = bool(
                    exact
                    and str(status.state).upper()
                        == "REVOKED"
                    and (
                        str(status.reason).upper()
                        == "NO_DIRECTIONAL_MASK_WAS_PUBLISHED"
                        or (
                            status.global_costmap_confirmed
                            and status.local_costmap_confirmed
                        )
                    )
                )

                if not revoked:
                    if now - tx["started"] >= 2.0:
                        tx["revoke_sent"] = False
                    return False

            self._directional_request = None
            self._directional_status = None

            tx["old"] = None
            tx["revoke_sent"] = False
            tx["phase"] = "PREPARE"
            tx["started"] = now - 1.0

            return prepare_and_publish()

        # ----------------------------------------------
        # MONITOR CURRENT INSTALL
        # ----------------------------------------------
        if tx["phase"] != "INSTALL":
            return False

        request = self._directional_request

        if request is None:
            tx["phase"] = "PREPARE"
            tx["started"] = now - 1.0
            return False

        if str(request.request_id) != tx["request_id"]:
            # Another transaction must not silently
            # consume this attempt's budget.
            return False

        status = self._directional_status

        exact_status = bool(
            status is not None
            and str(status.request_id)
                == tx["request_id"]
        )

        failed = bool(
            exact_status
            and str(status.state).upper() in {
                "FAILED",
                "REVOKED",
                "INACTIVE",
            }
        )

        timed_out = (
            now - tx["started"] >= 5.0
        )

        if not failed and not timed_out:
            return False

        if tx["count"] >= 2:
            return begin_adaptation()

        # Revoke the first failed mask before installing
        # another one with a new request_id.
        tx["old"] = copy.deepcopy(request)
        tx["phase"] = "CLEAR"
        tx["started"] = now
        tx["revoke_sent"] = False

        return False
    
    def _try_direct_nav2_handoff(self):
        """Apply the two-attempt D policy before the ordinary START."""
        if self._adaptation is not None:
            return False

        if self._episode_mode == "D":
            if not self._drive_directional_attempts(
                "DIRECTIONAL_ENFORCEMENT_UNAVAILABLE"
            ):
                return False

        return super()._try_direct_nav2_handoff()
    
    def _use_unconstrained_nav2(self, reason):
        """Never silently turn a confirmed D into C."""
        if self._episode_mode == "D":
            return self._drive_directional_attempts(reason)

        return super()._use_unconstrained_nav2(reason)

    def _begin_direct_nav2_handoff(self, direction):
        """Prevent an earlier release call from continuing into Mode C."""
        if (
            self._adaptation_enabled
            and self._adaptation is not None
        ):
            return False

        return super()._begin_direct_nav2_handoff(direction)

    def _release_navigation(self, direction):
        """Apply the resolved adaptation policy to the same social scope."""
        if (
            self._adaptation_enabled
            and self._adaptation is not None
        ):
            return False

        if (
            self._adaptation_enabled
            and self._scope
            and self._adaptation_c_scope == self._scope
        ):
            self._episode_mode = "C"
            self._hri_nav_fallback_active = True

            # Autonomous recovery must not reuse directional consent.
            self.confirmed_human_direction = "NONE"
            self.confirmed_human_intent = "UNKNOWN"
            self.dialogue_outcome = "AUTONOMOUS_ADAPTATION"
            self.authorized_episode_key = None
            self._directional_confirmed_scope = None

            direction = "NONE"

        return super()._release_navigation(direction)

    def _fail_direct_nav2_handoff(self, reason):
        """Route a D handoff failure through bounded recovery."""
        if self._adaptation is not None:
            return

        if self._episode_mode == "D":
            self._drive_directional_attempts(reason)
            return

        return super()._fail_direct_nav2_handoff(reason)
    
    def _block_direct_handoff(self, reason):
        """Preserve D recovery instead of an indefinite blocked phase."""
        if self._episode_mode == "D":
            self._drive_directional_attempts(reason)
            return

        return super()._block_direct_handoff(reason)
    
    def _evaluate_stop_barrier(self):
        """Keep the ordinary stop barrier from reopening the old dialogue."""
        if (
            self._adaptation_enabled
            and self._adaptation is not None
        ):
            return

        return super()._evaluate_stop_barrier()

    def _start_adaptation(self, reason):
        """Own one bounded recovery transaction for the current scope."""
        if (
            self._adaptation is not None
            or not self._scope
            or not self.active_navigation_goal_id
            or self.pending_terminal_stop is not None
        ):
            return

        self._adaptation = {
            "goal": self.active_navigation_goal_id,
            "scope": self._scope,
            "reason": str(reason),
            "phase": "STOP",
            "hold_id": "",
            "held": False,
            "last_wait": 0.0,
            "attempts": 0,
            "interaction_id": "",
            "request_id": "",
            "close_id": "",
            "close_sent": 0.0,
            "decision": "",
            "speech_id": "",
            "deadline": 0.0,
        }

        # Reserve the episode while WAIT supersedes any earlier START.
        self._released = True
        self._nav_phase = "ADAPTATION_STOP"
        self.nav2_cancel_confirmed = False

        self.stop_request_stamp_sec = (
            self.get_clock().now().nanoseconds / 1e9
        )

        self._transition_to(
            STATE_WAITING,
            reason="ADAPTATION_REQUESTED",
        )
        self._adaptation_wait()

    def _adaptation_wait(self):
        """Submit a hold whose exact acknowledgement is required."""
        recovery = self._adaptation

        recovery["last_wait"] = time.monotonic()
        recovery["hold_id"] = self._command(
            "WAIT",
            "ADAPTATION_STOP_REQUIRED",
        )

    def _on_gate_command_result(self, msg):
        """
        Route exact Gate results.

        D without motor recovery:
            One aborted attempt -> negotiate Mode C.
            Never consume the Mode-C retry budget.

        C:
            Preserve the existing one-retry policy.

        Explicit human refusal:
            Preserve the existing refusal procedure.

        Duplicate D failure events are idempotent.
        """
        data = self._json(
            msg,
            "navigation_goal_gate",
        )

        if data is None:
            return super()._on_gate_command_result(msg)

        goal_id = str(
            data.get("navigation_goal_id") or ""
        ).strip()

        if goal_id != self.active_navigation_goal_id:
            return super()._on_gate_command_result(msg)

        action = str(
            data.get("action") or ""
        ).strip().upper()

        status = str(
            data.get("status") or ""
        ).strip().upper()

        reason = str(
            data.get("reason") or ""
        ).strip().upper()

        request_id = str(
            data.get("request_id") or ""
        ).strip()

        # ==============================================
        # A. MODE D — NO-RECOVERY EXECUTION FAILURE
        # ==============================================
        if (
            action == "START"
            and status == "EXECUTION_FAILED"
            and reason in {
                "NAV2_D_NO_RECOVERY_ABORT",
                "D_BT_XML_NOT_CONFIGURED",
                "D_BT_FIELD_UNAVAILABLE",
            }
        ):
            last = getattr(
                self,
                "_nav2_last_start",
                None,
            )

            exact_d_start = bool(
                isinstance(last, dict)
                and last.get("goal") == goal_id
                and last.get("scope") == self._scope
                and last.get("mode") == "D"
                and last.get("request_id") == request_id
                and self._scope
                and self.pending_terminal_stop is None
            )

            if not exact_d_start:
                return

            handled = getattr(
                self,
                "_handled_d_failure_requests",
                None,
            )

            if handled is None:
                self._handled_d_failure_requests = set()
                handled = self._handled_d_failure_requests

            failure_key = (goal_id, request_id)

            if failure_key in handled:
                return

            handled.add(failure_key)

            # No second automatic D execution.
            self._nav2_same_goal_retry = None

            pending = self._control

            if (
                pending is not None
                and pending.get("action") == "START"
                and pending.get("request_id") == request_id
            ):
                self._control = None

            if (
                getattr(
                    self,
                    "_d_route_after_c_refusal",
                    False,
                )
                and getattr(
                    self,
                    "_d_route_after_c_refusal_goal",
                    None,
                ) == goal_id
            ):
                # Existing policy after an explicit
                # human refusal of Mode C.
                self._nav_phase = "D_FINAL_WAIT"
                self._released = True

                self._command(
                    "WAIT",
                    "FINAL_DIRECTIONAL_ROUTE_FAILURE",
                )

                self._announce_final_directional_failure()

            else:
                self.get_logger().warning(
                    "[MODE D] No-recovery execution failed | "
                    f"goal={goal_id} | "
                    f"reason={reason} | "
                    "same_goal=True | "
                    "next=ASK_MODE_C"
                )

                self._start_adaptation(
                    "D_DIRECTIONAL_NAVIGATION_UNAVAILABLE"
                )

            return

        # ==============================================
        # B. LEGACY D FAILURE RESULT
        # ==============================================
        if (
            action == "START"
            and status == "EXECUTION_FAILED"
            and reason == "NAV2_D_ROUTE_RETRY_EXHAUSTED"
        ):
            last = getattr(
                self,
                "_nav2_last_start",
                None,
            )

            if (
                isinstance(last, dict)
                and last.get("goal") == goal_id
                and last.get("mode") == "D"
                and last.get("request_id") == request_id
            ):
                if (
                    getattr(
                        self,
                        "_d_route_after_c_refusal",
                        False,
                    )
                    and getattr(
                        self,
                        "_d_route_after_c_refusal_goal",
                        None,
                    ) == goal_id
                ):
                    self._nav_phase = "D_FINAL_WAIT"
                    self._released = True

                    self._command(
                        "WAIT",
                        "FINAL_DIRECTIONAL_ROUTE_FAILURE",
                    )

                    self._announce_final_directional_failure()

                else:
                    self._start_adaptation(
                        "TWO_DIRECTIONAL_ROUTE_FAILURES"
                    )

                return

        # ==============================================
        # C. EXISTING SAME-GOAL RETRY
        # ==============================================
        retry = getattr(
            self,
            "_nav2_same_goal_retry",
            None,
        )

        if retry is not None:
            if (
                retry["goal"] != goal_id
                or retry["scope"] != self._scope
                or self.pending_terminal_stop is not None
            ):
                self._nav2_same_goal_retry = None
                retry = None

        if (
            retry is not None
            and action == "WAIT"
            and request_id == retry["wait_id"]
        ):
            if status == "HOLD_CONFIRMED":
                retry["held"] = True
                self.nav2_cancel_confirmed = True

                pending = self._control

                if (
                    pending is not None
                    and pending["action"] == "WAIT"
                    and pending["request_id"] == request_id
                ):
                    self._control = None

                self.get_logger().info(
                    "[NAV2 SAME-GOAL RETRY] "
                    "Hold confirmed | "
                    f"goal={goal_id} | "
                    "new_dialogue=False"
                )

            elif status in {"REJECTED", "EXPIRED"}:
                retry["wait_id"] = ""
                retry["held"] = False

                pending = self._control

                if (
                    pending is not None
                    and pending["request_id"] == request_id
                ):
                    self._control = None

                self.get_logger().warning(
                    "[NAV2 SAME-GOAL RETRY] "
                    "Hold rejected | "
                    f"goal={goal_id} | "
                    f"status={status}"
                )

            return

        # ==============================================
        # D. FIRST NON-DIRECTIONAL NAV2 ABORT
        # ==============================================
        last_start = getattr(
            self,
            "_nav2_last_start",
            None,
        )

        exact_start = bool(
            isinstance(last_start, dict)
            and last_start["goal"] == goal_id
            and last_start["request_id"] == request_id
            and last_start["scope"] == self._scope
        )

        if (
            action == "START"
            and status == "EXECUTION_FAILED"
            and reason == "NAV2_ABORT_RETRY_AVAILABLE"
            and exact_start
        ):
            if retry is not None:
                return

            if (
                not self._scope
                or self.pending_terminal_stop is not None
                or not self.active_navigation_goal_id
                or self.active_navigation_goal_pose is None
                or self._denied_scope
                or self.dialogue_outcome == "DENY_CONFIRMED"
            ):
                return

            mode = last_start["mode"]

            if mode not in {"C", "D"}:
                return

            human_direction = last_start[
                "human_direction"
            ]

            if mode == "D":
                if (
                    human_direction not in {"LEFT", "RIGHT"}
                    or not self._permission_applies_to_current_episode()
                ):
                    self.get_logger().warning(
                        "[NAV2 SAME-GOAL RETRY] "
                        "Directional permission no longer valid."
                    )
                    return

            retried_goals = getattr(
                self,
                "_nav2_retried_goals",
                None,
            )

            if retried_goals is None:
                self._nav2_retried_goals = set()
                retried_goals = self._nav2_retried_goals

            if goal_id in retried_goals:
                return

            retried_goals.add(goal_id)

            now = time.monotonic()

            self._nav2_same_goal_retry = {
                "goal": goal_id,
                "scope": self._scope,
                "mode": mode,
                "human_direction": human_direction,
                "robot_direction": last_start[
                    "robot_direction"
                ],
                "wait_id": "",
                "held": False,
                "last_sent_at": 0.0,
                "started_at": now,
            }

            self._control = None

            self.get_logger().warning(
                "[NAV2 SAME-GOAL RETRY] "
                "First Nav2 abort | "
                f"goal={goal_id} | "
                f"mode={mode} | "
                "original_destination_retained=True | "
                "new_dialogue=False | "
                "retry_budget=1"
            )

            wait_id = self._command(
                "WAIT",
                "NAV2_SAME_GOAL_RETRY_HOLD",
            )

            retry = self._nav2_same_goal_retry
            retry["wait_id"] = wait_id or ""
            retry["last_sent_at"] = time.monotonic()

            return

        # ==============================================
        # E. ORDINARY NEGOTIATED ADAPTATION
        # ==============================================
        adaptation = (
            self._adaptation
            if self._adaptation_enabled
            else None
        )

        if adaptation is not None:
            if (
                action == "WAIT"
                and request_id == adaptation["hold_id"]
                and status == "HOLD_CONFIRMED"
            ):
                adaptation["held"] = True
                self.nav2_cancel_confirmed = True
                self._control = None

            return

        # ==============================================
        # F. ORDINARY GATE RESULTS
        # ==============================================
        return super()._on_gate_command_result(msg)
    
    def _restart_adaptation_dialogue(self):
        """
        Restart Mode-D dialogue after the 180-second recovery WAIT.

        Keep the public goal and physical stop evidence.
        Discard previous HRI permission and directional semantics.
        Never authorize movement merely because time expired.
        """
        recovery = self._adaptation

        if recovery is None or recovery["phase"] != "DENIED":
            return False

        if (
            recovery["goal"] != self.active_navigation_goal_id
            or recovery["scope"] != self._scope
            or not self._scope
            or self.pending_terminal_stop is not None
            or not self.nav2_cancel_confirmed
            or not self._stopped()
            or self._control is not None
        ):
            return False

        # Do not create a new HRI turn while another closure
        # still owns the dialogue manager.
        if (
            self.pending_hri_close is not None
            or getattr(self, "_hri_detached_close", None) is not None
        ):
            return False

        # Remove the previous directional constraint before
        # restarting negotiation.
        if not self._revoke_optional_directional_mask(
            "DIALOGUE_RESTART_AFTER_180_SECONDS"
        ):
            return False

        goal_id = self.active_navigation_goal_id
        scope = self._scope

        # The previous recovery transaction is finished.
        self._adaptation = None
        self._directional_attempt_state = None
        self._adaptation_c_scope = None

        # Clear previous human decision and refusal.
        self._denied_scope = None
        self.authorized_episode_key = None
        self._retained_same_person_consent = None

        self.confirmed_human_intent = "UNKNOWN"
        self.confirmed_human_direction = "NONE"
        self.pending_human_intent = "UNKNOWN"
        self.pending_human_direction = "NONE"

        self.dialogue_outcome = "NONE"
        self.dialogue_resolution_complete = False

        self.dialogue_retry_count = 0
        self.dialogue_speech_retry_count = 0
        self.dialogue_started_at = None

        self.pending_dialogue_request_id = ""
        self.pending_dialogue_scope = None
        self.active_dialogue_request_type = ""
        self.active_alternative_direction = "NONE"
        self.active_interaction_id = ""

        self._hri_retry_after_close = None
        self._hri_retry_notice = "MISSED"
        self._hri_nav_fallback_active = False

        # A previous directional confirmation is no longer
        # transferable into this fresh negotiation.
        self._direction_semantic_candidate = None
        self._direction_semantic_ready = False
        self._directional_finalization = None
        self._directional_announcement_complete = False
        self._directional_confirmed_scope = None
        self._directional_confirmed_interaction_id = ""

        # A fresh dialogue always starts in Mode D.
        self._episode_mode = "D"
        self._direction = "NONE"

        # Keep the public goal and the exact confirmed hold.
        self._released = False
        self._nav_phase = "HELD"

        # Prevent the ordinary social timeout from interpreting
        # this reset as permission for autonomous navigation.
        self._social_wait_started_at = None
        self._social_wait_goal_id = ""
        self._social_wait_timeout_triggered = False

        self._transition_to(
            STATE_WAITING,
            reason="MODE_D_DIALOGUE_RESET_AFTER_180_SECONDS",
        )

        self.get_logger().info(
            "[MODE D] Dialogue restarted from zero | "
            f"goal={goal_id} | "
            f"scope={scope} | "
            "nav2_start=False | "
            "old_permission_reused=False | "
            "nav2_abort_retry_budget_preserved=True"
        )

        # Launch the normal HRI entry point rather than
        # creating a second dialogue implementation.
        self._start_human_initiative_window()

        return True
      
    def _on_timer(self):
        """
        Supervise Nav2 retry and negotiated C/D adaptation.

        Only an explicit confirmed NO starts WAIT 180 s.

        YES, silence and audio failure:
            bounded HRI cleanup,
            appropriate announcement,
            Mode C toward the same goal G.

        Physical stop, Gate and mask revocation
        remain mandatory.
        """
        now = time.monotonic()

        # ==============================================
        # A. SAME-GOAL NAV2 RETRY
        # ==============================================
        retry = getattr(
            self, "_nav2_same_goal_retry", None
        )

        if retry is not None:
            goal_id = retry["goal"]

            if (
                goal_id != self.active_navigation_goal_id
                or retry["scope"] != self._scope
                or self.pending_terminal_stop is not None
            ):
                self._nav2_same_goal_retry = None
                return

            if (
                self._denied_scope
                or self.dialogue_outcome == "DENY_CONFIRMED"
            ):
                self._nav2_same_goal_retry = None
                return

            if not retry["held"]:
                if self._control is not None:
                    return

                if now - retry["last_sent_at"] >= 5.0:
                    retry["wait_id"] = (
                        self._command(
                            "WAIT",
                            "NAV2_SAME_GOAL_RETRY_HOLD",
                        ) or ""
                    )
                    retry["last_sent_at"] = now

                return

            if (
                not self.nav2_cancel_confirmed
                or not self._stopped()
                or self._control is not None
            ):
                return

            mode = str(
                retry["mode"] or ""
            ).strip().upper()

            if mode == "C":
                release_direction = "NONE"

            elif mode == "D":
                if not self._permission_applies_to_current_episode():
                    self._nav2_same_goal_retry = None
                    return

                release_direction = retry["human_direction"]

                if release_direction not in {"LEFT", "RIGHT"}:
                    self._nav2_same_goal_retry = None
                    return

            else:
                self._nav2_same_goal_retry = None
                return

            self._released = False
            self._nav_phase = "HELD"

            if mode == "D":
                self._released = True
                self._nav_phase = "DIRECT_NAV2_WAIT_MASK"
                released = self._try_direct_nav2_handoff()

            else:
                released = self._release_navigation(
                    release_direction
                )

            if released or self._adaptation is not None:
                self._nav2_same_goal_retry = None

            return

        # ==============================================
        # B. NORMAL SUPERVISION
        # ==============================================
        if not self._adaptation_enabled:
            return super()._on_timer()

        recovery = self._adaptation

        if recovery is None:
            result = super()._on_timer()

            if self._adaptation is not None:
                self._nav_phase = (
                    "ADAPTATION_"
                    + self._adaptation["phase"]
                )

            return result

        # ==============================================
        # C. TRANSACTION OWNERSHIP
        # ==============================================
        if (
            recovery["goal"]
                != self.active_navigation_goal_id
            or recovery["scope"] != self._scope
            or self.pending_terminal_stop is not None
        ):
            if (
                recovery["interaction_id"]
                and recovery["close_id"]
                and recovery["phase"] == "CLOSING"
            ):
                self._send_adaptation_close()

            if recovery["speech_id"]:
                self._ros_cancel_speech(
                    recovery["speech_id"]
                )

            self._adaptation = None
            return

        self._watch_terminal_stop(now)

        phase = recovery["phase"]

        # ==============================================
        # D. CONFIRMED PHYSICAL STOP
        # ==============================================
        if phase == "STOP":
            if not recovery["held"]:
                if now - recovery["last_wait"] >= 5.0:
                    self._adaptation_wait()
                return

            if not self._stopped():
                return

            if (
                self.motion_state_stamp_sec is None
                or self.motion_state_stamp_sec
                    < self.stop_request_stamp_sec
            ):
                return

            self._ask_adaptation()
            return

        # ==============================================
        # E. HRI QUESTION
        # ==============================================
        if phase == "QUESTION":
            if now >= recovery["deadline"]:
                self._close_adaptation("NO_RESPONSE")
            return

        # ==============================================
        # F. BOUNDED HRI CLOSURE
        # ==============================================
        if phase == "CLOSING":
            if now < recovery["deadline"]:
                if now - recovery["close_sent"] >= 2.0:
                    self._send_adaptation_close()
                return

            # Exact CLOSED was not received in time.
            # Do not pretend HRI confirmed closure.
            # Preserve any explicit human refusal.
            recovery["audio_closure_unconfirmed"] = True

            self.get_logger().warning(
                "[ADAPTATION HRI] Closure timeout | "
                f"goal={recovery['goal']} | "
                f"decision={recovery['decision']} | "
                "closure_confirmed=False"
            )

            if recovery["decision"] != "DENY":
                recovery["decision"] = "TECHNICAL_FAILURE"

            # HRI cleanup was attempted, but must not
            # become an unbounded software dependency.
            self._announce_adaptation()
            return

        # ==============================================
        # G. BOUNDED ANNOUNCEMENT
        # ==============================================
        if phase == "ANNOUNCE":
            if now < recovery["deadline"]:
                return

            speech_id = recovery.get("speech_id", "")

            if speech_id:
                try:
                    self._ros_cancel_speech(speech_id)
                except Exception as exc:
                    self.get_logger().warning(
                        "[ADAPTATION] Speech cancel failed | "
                        f"error={exc}"
                    )

            self.get_logger().warning(
                "[ADAPTATION] Speech deadline exceeded | "
                f"decision={recovery['decision']} | "
                "next=MODE_C_RECOVERY"
            )

            # A speech timeout is not a human NO.
            recovery["announcement_failed"] = True
            recovery["phase"] = "RESUME"
            return

        # ==============================================
        # H. MODE C RESUMPTION
        # ==============================================
        if phase == "RESUME":
            if recovery["decision"] not in {
                "ALLOW",
                "NO_RESPONSE",
                "TECHNICAL_FAILURE",
            }:
                return

            # Physical stopping is a separate safety barrier.
            if (
                not self.nav2_cancel_confirmed
                or not self._stopped()
                or self._control is not None
            ):
                return

            # C must not inherit an active directional mask.
            if not self._revoke_optional_directional_mask(
                "NEGOTIATED_MODE_C"
            ):
                return

            self._adaptation_c_scope = self._scope
            self._directional_attempt_state = None

            # Temporarily release adaptation ownership
            # to enter the existing Mode-C handoff.
            self._adaptation = None
            self._released = False

            released = self._release_navigation("NONE")

            if released:
                self.get_logger().info(
                    "[ADAPTATION] Mode C handoff prepared | "
                    f"goal={recovery['goal']} | "
                    "direction=NONE"
                )
                return

            # A deferred Gate/Nav2 handoff must be retried.
            # Do not lose the adaptation transaction.
            self._adaptation = recovery
            self._released = True
            recovery["phase"] = "RESUME"

            self.get_logger().warning(
                "[ADAPTATION] Mode C handoff deferred | "
                f"goal={recovery['goal']} | "
                "retry_on_next_timer=True",
                throttle_duration_sec=5.0,
            )
            return

        # ==============================================
        # I. EXPLICIT NO — WAIT 180 SECONDS
        # ==============================================
        if phase == "DENIED":
            if recovery["decision"] != "DENY":
                return

            started_at = recovery.get(
                "refusal_started_at"
            )

            if started_at is None:
                started_at = now
                recovery["refusal_started_at"] = now

            if "wait_restart_at" not in recovery:
                recovery["wait_restart_at"] = started_at

                self.get_logger().info(
                    "[ADAPTATION REFUSAL] "
                    f"goal={recovery['goal']} | "
                    "wait=180s | "
                    "goal_preserved=True"
                )

            if now - started_at < 180.0:
                return

            if (
                not self.nav2_cancel_confirmed
                or not self._stopped()
                or self._control is not None
            ):
                return

            self._restart_adaptation_dialogue()
            return
        
    def _adaptation_text(self, key):
        """Return the adaptation question or outcome announcement."""
        language = str(
            self.language or "en"
        ).strip().lower()[:2]

        if language not in {"fr", "en", "es"}:
            language = "en"

        texts = {
            "question": {
                "fr": (
                    "Je ne parviens pas à garantir un passage "
                    "du côté demandé. M'autorisez-vous à "
                    "rechercher un autre itinéraire sans "
                    "contrainte directionnelle, en respectant "
                    "votre espace personnel ?"
                ),
                "en": (
                    "I cannot guarantee passage on the "
                    "requested side. May I look for another "
                    "route without a directional constraint, "
                    "while respecting your personal space?"
                ),
                "es": (
                    "No puedo garantizar el paso por el lado "
                    "solicitado. ¿Me permite buscar otra ruta "
                    "sin restricción direccional, respetando "
                    "su espacio personal?"
                ),
            },
            "ALLOW": {
                "fr": (
                    "Merci pour votre accord. Je vais rechercher "
                    "un autre itinéraire sans contrainte "
                    "directionnelle, en respectant les "
                    "distances de sécurité."
                ),
                "en": (
                    "Thank you for your permission. I will look "
                    "for another route without a directional "
                    "constraint, while maintaining safe distances."
                ),
                "es": (
                    "Gracias por su autorización. Buscaré otra "
                    "ruta sin restricción direccional, "
                    "manteniendo las distancias de seguridad."
                ),
            },
            "DENY": {
                "fr": (
                    "J'ai compris votre refus. Je reste arrêté "
                    "pendant trois minutes, puis je reprendrai "
                    "notre échange."
                ),
                "en": (
                    "I understand your refusal. I will remain "
                    "stopped for three minutes, then I will "
                    "start our dialogue again."
                ),
                "es": (
                    "He comprendido su negativa. Permaneceré "
                    "detenido durante tres minutos y después "
                    "volveré a iniciar nuestro diálogo."
                ),
            },
            "NO_RESPONSE": {
                "fr": (
                    "Je n'ai pas reçu de réponse. Je vais "
                    "rechercher un autre itinéraire sans "
                    "contrainte directionnelle, en respectant "
                    "les distances de sécurité."
                ),
                "en": (
                    "I did not receive an answer. I will look "
                    "for another route without a directional "
                    "constraint, while maintaining safe distances."
                ),
                "es": (
                    "No he recibido respuesta. Buscaré otra "
                    "ruta sin restricción direccional, "
                    "manteniendo las distancias de seguridad."
                ),
            },
            "TECHNICAL_FAILURE": {
                "fr": (
                    "Un problème technique empêche de poursuivre "
                    "notre dialogue. Je vais rechercher un autre "
                    "itinéraire sans contrainte directionnelle, "
                    "en respectant les distances de sécurité."
                ),
                "en": (
                    "A technical problem prevents me from "
                    "continuing our dialogue. I will look for "
                    "another route without a directional "
                    "constraint, while maintaining safe distances."
                ),
                "es": (
                    "Un problema técnico impide continuar nuestro "
                    "diálogo. Buscaré otra ruta sin restricción "
                    "direccional, manteniendo las distancias "
                    "de seguridad."
                ),
            },
        }

        return texts[key][language]
    
    def _ask_adaptation(self):
        """Open a fresh HRI turn after the exact hold is confirmed."""
        recovery = self._adaptation

        recovery["attempts"] += 1
        recovery["interaction_id"] = "ADAPT_" + uuid.uuid4().hex
        recovery["request_id"] = "TURN_" + uuid.uuid4().hex
        recovery["phase"] = "QUESTION"

        # Includes speech, cue, capture, transcription and transport.
        # The actual response window remains five seconds.
        recovery["deadline"] = time.monotonic() + 60.0

        request = DialogueRequest()
        request.header.stamp = self.get_clock().now().to_msg()
        request.interaction_id = recovery["interaction_id"]
        request.request_id = recovery["request_id"]
        request.person_id = self.active_person_id
        request.request_type = "INTENT_CONFIRMATION"
        request.context = "AUTONOMOUS_ADAPTATION_PERMISSION"
        request.question = self._adaptation_text("question")
        request.intent_to_confirm = "ALLOW_PASSAGE"
        request.direction_to_confirm = "NONE"
        request.robot_initiated = True
        request.response_timeout = 5.0
        request.language = self.language

        self.dialogue_request_pub.publish(request)

    def _human_intent_callback(self, msg):
        """
        Process the exact adaptation dialogue response.

        YES:
            Announce and resume in Mode C.

        Explicit, reliable NO:
            Retain G in WAIT for 180 seconds.

        Human silence:
            Announce absence of response, then Mode C.

        Technical audio failure:
            Announce technical difficulty, then Mode C.

        An unclear recognition is never treated as consent
        or refusal.
        """
        import re

        recovery = (
            self._adaptation
            if self._adaptation_enabled
            else None
        )

        if recovery is None:
            return super()._human_intent_callback(msg)

        if (
            recovery["phase"] not in {
                "QUESTION",
                "CLOSING",
                "ANNOUNCE",
                "RESUME",
            }
            or str(msg.interaction_id)
                != recovery["interaction_id"]
            or str(msg.request_id)
                != recovery["request_id"]
            or str(msg.person_id)
                != str(self.active_person_id)
            or not self._message_is_recent(msg, 2.0)
        ):
            return

        intent = str(
            msg.intent or ""
        ).strip().upper()

        source = str(
            msg.source or ""
        ).strip().upper()

        direction = str(
            msg.direction or "NONE"
        ).strip().upper()

        utterance = re.sub(
            r"[.!?¡¿,;:]+",
            " ",
            str(msg.utterance or "").lower(),
        )

        utterance = " ".join(utterance.split())

        try:
            confidence = float(msg.confidence)
        except (ValueError, TypeError):
            confidence = float("nan")

        reliable = (
            math.isfinite(confidence)
            and confidence >= 0.5
        )

        technical_sources = {
            "ROBOT_SPEECH_FAILED",
            "ROBOT_SPEECH_TIMEOUT",
            "DIALOGUE_REQUEST_REJECTED",
            "AUDIO_TERMINATION_UNCONFIRMED",
        }

        explicit_negative = utterance in {
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

        if source in technical_sources:
            decision = "TECHNICAL_FAILURE"

        elif source in {
            "HUMAN_RESPONSE_TIMEOUT",
            "TIMEOUT",
        }:
            decision = "NO_RESPONSE"

        elif (
            intent in {"DENY_PASSAGE", "REJECT"}
            and reliable
            and explicit_negative
        ):
            decision = "DENY"

        elif (
            intent in {"CONFIRM", "ALLOW_PASSAGE"}
            and reliable
            and direction == "NONE"
        ):
            decision = "ALLOW"

        else:
            decision = "RETRY"

        if (
            recovery["phase"] != "QUESTION"
            and decision != "DENY"
        ):
            return

        if decision == "DENY":
            recovery.setdefault(
                "refusal_started_at",
                time.monotonic(),
            )

        if recovery["phase"] != "QUESTION":
            if recovery.get("speech_id"):
                self._ros_cancel_speech(
                    recovery["speech_id"]
                )

            recovery["decision"] = "DENY"
            self._announce_adaptation()
            return

        self._close_adaptation(decision)

    def _close_adaptation(self, decision):
        """Close the exact HRI turn before retrying or resolving recovery."""
        recovery = self._adaptation

        recovery["decision"] = decision
        recovery["phase"] = "CLOSING"
        recovery["close_id"] = "CLOSE_" + uuid.uuid4().hex
        recovery["deadline"] = time.monotonic() + 10.0

        self._send_adaptation_close()

    def _send_adaptation_close(self):
        """Send an idempotent closure request for the owned HRI turn."""
        recovery = self._adaptation
        recovery["close_sent"] = time.monotonic()

        payload = {
            "version": 1,
            "command": "CLOSE",
            "close_id": recovery["close_id"],
            "interaction_id": recovery["interaction_id"],
            "request_id": recovery["request_id"],
            "reason": "ADAPTATION_TURN_FINISHED",
        }

        self.hri_dialogue_control_pub.publish(
            String(data=json.dumps(payload))
        )

    def _hri_dialogue_control_result_callback(self, msg):
        """
        Process an exact HRI closure acknowledgment.

        Never interpret TERMINATION_UNCONFIRMED as CLOSED.
        A technical audio failure remains separate from
        human silence and human refusal.
        """
        recovery = self._adaptation

        if recovery is None:
            return super()._hri_dialogue_control_result_callback(
                msg
            )

        data = self._json(msg, "hri_manager")

        if (
            data is None
            or recovery["phase"] != "CLOSING"
        ):
            return

        if any(
            data.get(key) != recovery.get(key)
            for key in (
                "close_id",
                "interaction_id",
                "request_id",
            )
        ):
            return

        status = str(
            data.get("status") or ""
        ).strip().upper()

        if status == "TERMINATION_UNCONFIRMED":
            self.get_logger().error(
                "[ADAPTATION HRI] Audio closure unconfirmed | "
                f"goal={recovery['goal']} | "
                f"interaction={recovery['interaction_id']} | "
                "navigation_remains_held=True"
            )

            recovery["audio_closure_unconfirmed"] = True
            return

        if status != "CLOSED":
            return

        recovery["audio_closure_unconfirmed"] = False

        decision = str(
            recovery.get("decision") or ""
        ).strip().upper()

        if decision == "RETRY":
            if recovery["attempts"] < 2:
                self._ask_adaptation()
                return

            decision = "NO_RESPONSE"
            recovery["decision"] = decision

        if decision not in {
            "ALLOW",
            "DENY",
            "NO_RESPONSE",
            "TECHNICAL_FAILURE",
        }:
            recovery["decision"] = "TECHNICAL_FAILURE"

        self._announce_adaptation()

    def _announce_adaptation(self):
        """
        Announce the current adaptation outcome.

        DENY:
            remain stopped and restart negotiation after
            the refusal deadline.

        ALLOW / NO_RESPONSE / TECHNICAL_FAILURE:
            announce the appropriate outcome, then Mode C.
        """
        recovery = self._adaptation

        if recovery is None:
            return

        decision = str(
            recovery.get("decision") or ""
        ).strip().upper()

        allowed_decisions = {
            "ALLOW",
            "DENY",
            "NO_RESPONSE",
            "TECHNICAL_FAILURE",
        }

        if decision not in allowed_decisions:
            decision = "TECHNICAL_FAILURE"
            recovery["decision"] = decision

        speech = ExplanationText()
        speech.header.stamp = self.get_clock().now().to_msg()
        speech.context_stamp = speech.header.stamp

        speech.decision_id = (
            "ADAPT_SPEECH_" + uuid.uuid4().hex
        )

        speech.language = self.language
        speech.text = self._adaptation_text(decision)
        speech.fallback_language = "en"

        original_language = self.language

        try:
            self.language = "en"
            speech.fallback_text = self._adaptation_text(
                decision
            )
        finally:
            self.language = original_language

        speech.provider = "deterministic"
        speech.model = "negotiated_adaptation"
        speech.speech_kind = "SOCIAL_EXPLANATION"

        speech.navigation_goal_id = recovery["goal"]
        speech.navigation_state = "WAITING"
        speech.navigation_active = False
        speech.social_conflict_active = True

        recovery["speech_id"] = speech.decision_id
        recovery["deadline"] = (
            time.monotonic()
            + float(self.speech_completion_timeout_sec)
        )

        if decision == "DENY":
            recovery["phase"] = "DENIED"
            self._denied_scope = self._scope
            self.dialogue_outcome = "DENY_CONFIRMED"

            recovery.setdefault(
                "refusal_started_at",
                time.monotonic(),
            )

        else:
            recovery["phase"] = "ANNOUNCE"

            if decision == "ALLOW":
                self.dialogue_outcome = "ALLOW_CONFIRMED"

            elif decision == "NO_RESPONSE":
                self.dialogue_outcome = "NO_RESPONSE"

            elif decision == "TECHNICAL_FAILURE":
                self.dialogue_outcome = (
                    "HRI_TECHNICAL_FAILURE"
                )

        self._adaptation_speech_pub.publish(speech)

        self.get_logger().info(
            "[ADAPTATION] Outcome announced | "
            f"decision={decision} | "
            f"phase={recovery['phase']} | "
            f"goal={recovery['goal']} | "
            f"mode_c_next="
            f"{decision in {'ALLOW', 'NO_RESPONSE', 'TECHNICAL_FAILURE'}}"
        )

    def _on_speech_status(self, msg):
        """
        Track the exact adaptation announcement.

        Only an explicit DENY produces a refusal WAIT.
        All other adaptation outcomes proceed to Mode C
        after announcement completion or terminal failure.
        """
        recovery = (
            self._adaptation
            if self._adaptation_enabled
            else None
        )

        if recovery is None:
            return super()._on_speech_status(msg)

        speech_id = str(
            msg.decision_id or ""
        ).strip()

        if (
            not recovery["speech_id"]
            or speech_id != recovery["speech_id"]
        ):
            return super()._on_speech_status(msg)

        state = str(
            msg.state or ""
        ).strip().upper()

        if state not in {
            "COMPLETED",
            "FAILED",
            "CANCELED",
            "TIMEOUT",
        }:
            return

        if (
            recovery["phase"] == "DENIED"
            and recovery["decision"] == "DENY"
        ):
            recovery["denial_announcement_done"] = True
            return

        if recovery["phase"] != "ANNOUNCE":
            return

        if recovery["decision"] not in {
            "ALLOW",
            "NO_RESPONSE",
            "TECHNICAL_FAILURE",
        }:
            return

        if state != "COMPLETED":
            self.get_logger().warning(
                "[ADAPTATION] Speech not completed | "
                f"state={state} | "
                f"decision={recovery['decision']} | "
                "next=MODE_C"
            )

        recovery["phase"] = "RESUME"


    