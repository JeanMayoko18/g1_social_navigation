#!/usr/bin/env python3
"""Annonce pré-goal indépendante de la FSM de collaboration."""

import time
import uuid
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosidl_runtime_py.utilities import get_message

from g1_social_msgs.msg import (
    ExplanationText,
    NavigationState,
    SocialNavigationDecision,
    SpeechCancelRequest,
    SpeechStatus,
)


class PreGoalAnnouncementNode(Node):

    def __init__(self):
        super().__init__("pre_goal_announcement")

        self.declare_parameter("language", "en")
        self.declare_parameter("cooldown_sec", 35.0)
        self.declare_parameter("max_event_age_sec", 2.0)

        # État local au nœud, indépendant de la FSM.
        self._goal_id = ""
        self._nav_state_received = False
        self._seen = {}
        self._last_announcement_at = float("-inf")

        self._speech_id = ""
        self._speech_sent_at = 0.0

        self._goal_context_sub = None

        # Perception sociale.
        self.create_subscription(
            SocialNavigationDecision,
            "/social_navigation/decision",
            self._on_decision,
            qos_profile_sensor_data,
        )

        # État de navigation en lecture seule.
        self.create_subscription(
            NavigationState,
            "/social_navigation/navigation_state",
            self._on_navigation_state,
            qos_profile_sensor_data,
        )

        # Résultats de nos propres annonces.
        self.create_subscription(
            SpeechStatus,
            "/explainability/speech_status",
            self._on_speech_status,
            10,
        )

        # Publication directe vers le pipeline TTS.
        self._text_pub = self.create_publisher(
            ExplanationText,
            "/explainability/text",
            10,
        )

        self._cancel_pub = self.create_publisher(
            SpeechCancelRequest,
            "/explainability/speech_cancel_request",
            10,
        )

        # Découverte du type ROS du contexte de goal.
        # Cela évite de supposer le nom exact du message.
        self._goal_context_timer = self.create_timer(
            1.0,
            self._subscribe_goal_context,
        )

        self._maintenance_timer = self.create_timer(
            5.0,
            self._maintenance,
        )

    # --------------------------------------------------
    # SURVEILLANCE DU GOAL PUBLIC
    # --------------------------------------------------

    def _subscribe_goal_context(self):
        if self._goal_context_sub is not None:
            return

        topic = "/social_navigation/navigation_goal_context"

        for name, types in self.get_topic_names_and_types():
            if name != topic or len(types) != 1:
                continue

            try:
                msg_type = get_message(types[0])

                self._goal_context_sub = self.create_subscription(
                    msg_type,
                    topic,
                    self._on_goal_context,
                    10,
                )

            except Exception as exc:
                self.get_logger().warning(
                    "[PRE-GOAL] Cannot subscribe "
                    f"to goal context: {exc}"
                )
                return

            self.get_logger().info(
                "[PRE-GOAL] Goal context monitored"
            )

            self.destroy_timer(self._goal_context_timer)
            return

    def _on_goal_context(self, msg):
        goal_id = str(
            getattr(msg, "navigation_goal_id", "") or ""
        ).strip()

        if goal_id:
            self._goal_id = goal_id
            self._cancel_own_announcement()

    def _on_navigation_state(self, msg):
        self._nav_state_received = True

        goal_id = str(
            msg.navigation_goal_id or ""
        ).strip()

        state = str(
            msg.navigation_state or ""
        ).strip().upper()

        if goal_id and state in {
            "SUCCEEDED",
            "ABORTED",
            "CANCELED",
        }:
            if goal_id == self._goal_id:
                self._goal_id = ""
            return

        if goal_id:
            self._goal_id = goal_id
            self._cancel_own_announcement()

    # --------------------------------------------------
    # DÉTECTION AVANT GOAL
    # --------------------------------------------------

    def _on_decision(self, msg):

        if (
            str(msg.event).strip().upper()
            != "INTRUSION_CONFIRMED"
        ):
            return

        # Aucune autorisation de modifier la FSM.
        # On vérifie uniquement la présence d'un goal.
        if (
            not self._nav_state_received
            or self._goal_id
            or str(msg.navigation_goal_id or "").strip()
            or bool(msg.navigation_active)
        ):
            return

        person = str(msg.person_id or "").strip()

        if not person:
            return

        # Rejeter les événements trop anciens.
        stamp = msg.header.stamp

        stamp_ns = (
            int(stamp.sec) * 1_000_000_000
            + int(stamp.nanosec)
        )

        age_sec = (
            self.get_clock().now().nanoseconds - stamp_ns
        ) / 1e9

        max_age_sec = float(
            self.get_parameter("max_event_age_sec").value
        )

        if not (-0.2 <= age_sec <= max_age_sec):
            return

        now = time.monotonic()

        cooldown = float(
            self.get_parameter("cooldown_sec").value
        )

        key = (
            str(msg.tracking_session_id or "").strip(),
            person,
        )

        # Éviter les annonces répétitives.
        if (
            now - self._last_announcement_at < cooldown
            or now - self._seen.get(
                key,
                float("-inf"),
            ) < cooldown
            or self._speech_id
        ):
            return

        self._publish_greeting(
            person,
            key,
            now,
            distance=msg.person_distance,
        )

    # --------------------------------------------------
    # CONSTRUCTION DE L'ANNONCE
    # --------------------------------------------------

    def _publish_greeting(self, person, key, now, *, distance):
        """Announce presence and conditional future navigation awareness."""
        language = str(
            self.get_parameter("language").value
        ).strip().lower()

        try:
            distance_m = float(distance)
        except (TypeError, ValueError, OverflowError):
            distance_m = float("nan")

        # Only describe a usable measured distance.
        distance_available = (
            math.isfinite(distance_m)
            and distance_m > 0.0
        )

        if language.startswith("fr"):
            language = "fr"

            if distance_available:
                distance_text = (
                    f"{distance_m:.1f}".replace(".", ",")
                )

                greeting = (
                    "Bonjour, je vous ai repéré "
                    f"à environ {distance_text} mètre"
                    f"{'s' if round(distance_m, 1) > 1.0 else ''}. "
                    "Si je commence à naviguer, "
                    "je tiendrai compte de votre présence "
                    "pour adapter mon trajet."
                )
            else:
                greeting = (
                    "Bonjour, j'ai repéré votre présence "
                    "à proximité. "
                    "Si je commence à naviguer, "
                    "je tiendrai compte de votre présence "
                    "pour adapter mon trajet."
                )
        
        elif language.startswith("es"):
            language = "es"

            if distance_available:
                distance_text = (
                    f"{distance_m:.1f}".replace(".", ",")
                )
                unit = (
                    "metro"
                    if round(distance_m, 1) == 1.0
                    else "metros"
                )

                greeting = (
                    "Hola, he detectado su presencia "
                    f"a unos {distance_text} {unit} de distancia. "
                    "Si empiezo a desplazarme, "
                    "tendré en cuenta su presencia "
                    "para adaptar mi recorrido."
                )
            else:
                greeting = (
                    "Hola, he detectado su presencia cerca de mí. "
                    "Si empiezo a desplazarme, "
                    "tendré en cuenta su presencia "
                    "para adaptar mi recorrido."
                )

        else:
            language = "en"

            if distance_available:
                distance_text = f"{distance_m:.1f}"

                greeting = (
                    "Hello, I noticed you "
                    f"about {distance_text} "
                    f"{'meter' if round(distance_m, 1) == 1.0 else 'meters'} "
                    "away. "
                    "If I start navigating, "
                    "I will take your presence into account "
                    "when planning my route."
                )
            else:
                greeting = (
                    "Hello, I noticed you nearby. "
                    "If I start navigating, "
                    "I will take your presence into account "
                    "when planning my route."
                )

        speech = ExplanationText()

        speech.header.stamp = (
            self.get_clock().now().to_msg()
        )
        speech.context_stamp = speech.header.stamp

        speech.decision_id = (
            "PRE_GOAL_GREETING_" + uuid.uuid4().hex
        )

        speech.text = greeting
        speech.language = language
        speech.fallback_text = greeting
        speech.fallback_language = language

        speech.provider = "deterministic"
        speech.model = "pre_goal_greeting_v2"
        speech.used_fallback = False
        speech.used_cache = False
        speech.speech_kind = "SOCIAL_EXPLANATION"

        # This greeting does not represent a navigation transaction.
        speech.navigation_goal_id = ""
        speech.navigation_state = "UNKNOWN"
        speech.navigation_active = False
        speech.social_conflict_active = False

        self._seen[key] = now
        self._last_announcement_at = now

        self._speech_id = speech.decision_id
        self._speech_sent_at = now

        self._text_pub.publish(speech)

        distance_log = (
            f"{distance_m:.2f}"
            if distance_available
            else "UNKNOWN"
        )

        self.get_logger().info(
            "[PRE-GOAL] Greeting published | "
            f"person={person} | "
            f"distance_m={distance_log} | "
            f"decision={speech.decision_id}"
        )

    # --------------------------------------------------
    # ANNULATION LORSQU'UN GOAL ARRIVE
    # --------------------------------------------------

    def _cancel_own_announcement(self):

        if not self._speech_id:
            return

        cancel = SpeechCancelRequest()
        cancel.decision_id = self._speech_id

        self._cancel_pub.publish(cancel)

        self.get_logger().info(
            "[PRE-GOAL] Cancel own speech | "
            f"decision={self._speech_id}"
        )

        self._speech_id = ""

    # --------------------------------------------------
    # RÉSULTAT AUDIO INDÉPENDANT
    # --------------------------------------------------

    def _on_speech_status(self, msg):

        if msg.decision_id != self._speech_id:
            return

        if msg.state in {
            "COMPLETED",
            "FAILED",
            "CANCELED",
        }:
            self.get_logger().info(
                "[PRE-GOAL] Speech result="
                f"{msg.state}"
            )

            self._speech_id = ""

    # --------------------------------------------------
    # ENTRETIEN LOCAL
    # --------------------------------------------------

    def _maintenance(self):

        now = time.monotonic()

        if (
            self._speech_id
            and now - self._speech_sent_at > 45.0
        ):
            self.get_logger().warning(
                "[PRE-GOAL] Speech timeout"
            )

            self._cancel_own_announcement()

        # Nettoyage de l'historique, sans effet ROS.
        self._seen = {
            key: last
            for key, last in self._seen.items()
            if now - last < 180.0
        }


def main(args=None):
    rclpy.init(args=args)

    node = PreGoalAnnouncementNode()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()