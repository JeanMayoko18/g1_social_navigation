#!/usr/bin/env python3

import json
import re
import unicodedata
import rclpy
from rclpy.node import Node

from hri_msgs.msg import LiveSpeech, IdsList
from hri_actions_msgs.msg import Intent



class ROS4HRIBackend:
    """
    ROS4HRI backend.

    Responsibilities:
        - Listen to ROS4HRI speech topics.
        - Listen to /intents.
        - Convert ROS4HRI data into a normalized representation.
        - Forward the normalized intent to the HRI manager.

    This backend does NOT publish HumanIntent.

    Architecture:

        ROS4HRI
           |
           +---- /intents
           |
           v
        ROS4HRIBackend
           |
           v
        HRIManagerNode
           |
           v
        HRIManager Core
    """

    def __init__(
        self,
        node: Node,
        intent_callback,
    ):
        self.node = node
        self.intent_callback = intent_callback

        # ---------------------------------------------------------
        # ROS4HRI voice tracking
        # ---------------------------------------------------------

        self.voice_ids_sub = node.create_subscription(
            IdsList,
            "/humans/voices/tracked",
            self._voice_ids_callback,
            10,
        )

        self.voice_speech_subscriptions = {}

        # ---------------------------------------------------------
        # ROS4HRI interpreted intents
        # ---------------------------------------------------------

        self.intent_sub = node.create_subscription(
            Intent,
            "/intents",
            self._intent_callback,
            10,
        )

        self.node.get_logger().info(
            "ROS4HRI backend initialized."
        )

        self.node.get_logger().info(
            "Subscribed to /intents."
        )

    # =========================================================
    # Backend lifecycle
    # =========================================================

    def is_available(self) -> bool:
        return rclpy.ok()

    def shutdown(self) -> None:
        pass

    # =========================================================
    # Voice tracking
    # =========================================================

    def _voice_ids_callback(
        self,
        msg: IdsList,
    ) -> None:

        for voice_id in msg.ids:

            if voice_id in self.voice_speech_subscriptions:
                continue

            topic = (
                f"/humans/voices/"
                f"{voice_id}/speech"
            )

            self.voice_speech_subscriptions[voice_id] = (
                self.node.create_subscription(
                    LiveSpeech,
                    topic,
                    lambda speech_msg, vid=voice_id:
                        self._speech_callback(
                            vid,
                            speech_msg,
                        ),
                    10,
                )
            )

            self.node.get_logger().info(
                f"ROS4HRI voice detected: {voice_id}"
            )

    def _speech_callback(
        self,
        voice_id: str,
        msg: LiveSpeech,
    ) -> None:

        if not msg.final:
            return

        self.node.get_logger().info(
            f"ROS4HRI speech [{voice_id}]: "
            f"{msg.final}"
        )

    # =========================================================
    # Intent
    # =========================================================

    def _intent_callback(
        self,
        msg: Intent,
    ) -> None:
        """
        Receives a ROS4HRI Intent.

        The backend does NOT create HumanIntent here.

        It extracts the JSON data and forwards the normalized
        information to HRIManagerNode.
        """

        data = {}

        self.node.get_logger().info(
            "[ROS4HRI RAW INTENT] "
            f"intent={msg.intent!r} | "
            f"data={msg.data!r} | "
            f"confidence={msg.confidence:.2f} | "
            f"source={msg.source!r} | "
            f"modality={msg.modality!r}"
        )

        if msg.data:
            try:
                decoded = json.loads(msg.data)

                if isinstance(decoded, dict):
                    data = decoded
                else:
                    self.node.get_logger().warning(
                        "ROS4HRI Intent.data must decode to "
                        "a JSON object. Using empty data."
                    )

            except json.JSONDecodeError:
                self.node.get_logger().warning(
                    "Invalid ROS4HRI Intent.data JSON. "
                    "Using empty data."
                )

        normalized_intent = self._map_intent(
            msg.intent,
            data,
        )

        direction = self._extract_direction(
            data
        )

        target = self._extract_target(
            data
        )

        constraint = self._extract_constraint(
            data
        )

        utterance = self._extract_utterance(
            data
        )

        source = self._map_source(
            msg.source
        )
        modality = self._map_modality(msg.modality)

        self.node.get_logger().info(
            "ROS4HRI intent received: "
            f"intent={normalized_intent}, "
            f"target={target}, "
            f"direction={direction}, "
            f"confidence={msg.confidence:.2f}, "
            f"source={source},"
            f"modality={modality}"
        )

        # -----------------------------------------------------
        # Forward to HRIManagerNode
        # -----------------------------------------------------

        self.intent_callback(
            intent=normalized_intent,
            data={
                "direction": direction,
                "target": target,
                "constraint": constraint,
            },
            confidence=float(msg.confidence),
            source=source,
            modality=modality,
            utterance=utterance,
        )

    # =========================================================
    # Intent mapping
    # =========================================================


    @staticmethod
    def _map_intent(
        ros4hri_intent: str,
        data: dict,
    ) -> str:
        """
        Convert ROS4HRI intent names to the project's
        internal vocabulary.

        Explicit project-specific intents contained in
        the JSON data have priority.

        RAW_USER_INPUT is interpreted conservatively from
        the recognized utterance. Ambiguous input remains
        UNKNOWN.
        """

        internal_intent = str(
            data.get("intent", "")
        ).strip().upper()

        allowed_internal = {
            "ALLOW_PASSAGE",
            "DENY_PASSAGE",
            "REQUEST_STOP",
            "REQUEST_FOLLOW",
            "REQUEST_LEFT",
            "REQUEST_RIGHT",
            "CONFIRM",
            "REJECT",
            "UNKNOWN",
        }

        if internal_intent in allowed_internal:
            return internal_intent

        raw_intent = str(
            ros4hri_intent or ""
        ).strip()

        if raw_intent.upper() == "RAW_USER_INPUT":
            utterance = str(
                data.get(
                    "input",
                    data.get("utterance", ""),
                )
            ).strip()

            return ROS4HRIBackend._map_raw_user_input(
                utterance
            )

        mapping = {
            Intent.STOP_ACTIVITY: "REQUEST_STOP",
            Intent.SUSPEND: "REQUEST_STOP",
            Intent.GUIDE: "REQUEST_FOLLOW",
        }

        return mapping.get(
            raw_intent,
            raw_intent.upper()
            if raw_intent
            else "UNKNOWN",
        )

    @staticmethod
    def _normalize_speech_text(text: str) -> str:
        """
        Normalize recognized speech for conservative
        deterministic intent matching.
        """

        text = unicodedata.normalize(
            "NFKD",
            str(text or ""),
        )

        text = "".join(
            char
            for char in text
            if not unicodedata.combining(char)
        )

        text = text.lower()

        text = re.sub(
            r"[^\w\s'-]",
            " ",
            text,
        )

        return " ".join(text.split())

    @staticmethod
    def _map_raw_user_input(
        utterance: str,
    ) -> str:
        """
        Interpret short direct human responses.

        This mapping is intentionally conservative:
        unrecognized or ambiguous speech becomes UNKNOWN.
        """

        text = ROS4HRIBackend._normalize_speech_text(
            utterance
        )

        if not text:
            return "UNKNOWN"

        # --------------------------------------------------
        # Explicit stop requests have highest priority.
        # --------------------------------------------------

        stop_patterns = (
            r"\bstop\b",
            r"\barrete\b",
            r"\barretez\b",
            r"\barreter\b",
            r"\bimmobilise\b",
            r"\bimmobilisez\b",
            r"\bdetente\b",
            r"\bdetengase\b",
            r"\bpara\b",
        )

        if any(
            re.search(pattern, text)
            for pattern in stop_patterns
        ):
            return "REQUEST_STOP"

        # --------------------------------------------------
        # Explicit denial of passage.
        # Check before positive passage expressions.
        # --------------------------------------------------

        deny_passage_patterns = (
            r"\bne passe pas\b",
            r"\bne passez pas\b",
            r"\bvous ne pouvez pas passer\b",
            r"\btu ne peux pas passer\b",
            r"\bdon'?t pass\b",
            r"\bdo not pass\b",
            r"\byou cannot pass\b",
            r"\byou can'?t pass\b",
            r"\bno pase\b",
            r"\bno pases\b",
            r"\bno puede pasar\b",
            r"\bno puedes pasar\b",
        )

        if any(
            re.search(pattern, text)
            for pattern in deny_passage_patterns
        ):
            return "DENY_PASSAGE"

        # --------------------------------------------------
        # Explicit permission to pass.
        # --------------------------------------------------

        allow_passage_patterns = (
            r"\bvous pouvez passer\b",
            r"\btu peux passer\b",
            r"\ballez[- ]y\b",
            r"\bpassez\b",
            r"\bgo ahead\b",
            r"\byou can pass\b",
            r"\byou may pass\b",
            r"\bpuede pasar\b",
            r"\bpuedes pasar\b",
            r"\badelante\b",
        )

        if any(
            re.search(pattern, text)
            for pattern in allow_passage_patterns
        ):
            return "ALLOW_PASSAGE"

        # --------------------------------------------------
        # Follow request.
        # --------------------------------------------------

        follow_patterns = (
            r"\bsuis[- ]moi\b",
            r"\bsuivez[- ]moi\b",
            r"\bfollow me\b",
            r"\bsigueme\b",
            r"\bsigame\b",
        )

        if any(
            re.search(pattern, text)
            for pattern in follow_patterns
        ):
            return "REQUEST_FOLLOW"

        # --------------------------------------------------
        # Directional requests.
        # --------------------------------------------------

        left_patterns = (
            r"\ba gauche\b",
            r"\bvers la gauche\b",
            r"\bon the left\b",
            r"\bto the left\b",
            r"\bleft\b",
            r"\ba la izquierda\b",
            r"\bizquierda\b",
        )

        right_patterns = (
            r"\ba droite\b",
            r"\bvers la droite\b",
            r"\bon the right\b",
            r"\bto the right\b",
            r"\bright\b",
            r"\ba la derecha\b",
            r"\bderecha\b",
        )

        # A directional phrase combined with an affirmative
        # response remains a confirmation. Direction is
        # extracted separately by _extract_direction().
        affirmative_patterns = (
            r"\boui\b",
            r"\bouais\b",
            r"\bd'accord\b",
            r"\bok\b",
            r"\bokay\b",
            r"\byes\b",
            r"\byeah\b",
            r"\byep\b",
            r"\bsi\b",
            r"\bclaro\b",
        )

        negative_patterns = (
            r"\bnon\b",
            r"\bno\b",
            r"\bnope\b",
        )

        if any(
            re.search(pattern, text)
            for pattern in negative_patterns
        ):
            return "REJECT"

        if any(
            re.search(pattern, text)
            for pattern in affirmative_patterns
        ):
            return "CONFIRM"

        if any(
            re.search(pattern, text)
            for pattern in left_patterns
        ):
            return "REQUEST_LEFT"

        if any(
            re.search(pattern, text)
            for pattern in right_patterns
        ):
            return "REQUEST_RIGHT"

        return "UNKNOWN"
    
    # =========================================================
    # Data extraction
    # =========================================================

    @staticmethod
    def _extract_direction(
        data: dict,
    ) -> str:

        direction = str(
            data.get("direction", "")
        ).strip().upper()

        allowed = {
            "LEFT",
            "RIGHT",
            "FRONT",
            "BACK",
            "NONE",
        }

        if direction in allowed:
            return direction

        utterance = str(
            data.get(
                "input",
                data.get("utterance", ""),
            )
        ).strip()

        text = ROS4HRIBackend._normalize_speech_text(
            utterance
        )

        left_patterns = (
            r"\ba gauche\b",
            r"\bvers la gauche\b",
            r"\bon the left\b",
            r"\bto the left\b",
            r"\bleft\b",
            r"\ba la izquierda\b",
            r"\bizquierda\b",
        )

        right_patterns = (
            r"\ba droite\b",
            r"\bvers la droite\b",
            r"\bon the right\b",
            r"\bto the right\b",
            r"\bright\b",
            r"\ba la derecha\b",
            r"\bderecha\b",
        )

        if any(
            re.search(pattern, text)
            for pattern in left_patterns
        ):
            return "LEFT"

        if any(
            re.search(pattern, text)
            for pattern in right_patterns
        ):
            return "RIGHT"

        return "NONE"

    @staticmethod
    def _extract_target(
        data: dict,
    ) -> str:

        for key in (
            "target",
            "goal",
            "object",
            "recipient",
        ):
            value = data.get(key)

            if value is not None:
                value = str(value).strip()

                if value:
                    return value

        return ""

    @staticmethod
    def _extract_constraint(
        data: dict,
    ) -> str:

        value = data.get(
            "constraint",
            "",
        )

        return str(value).strip()

    @staticmethod
    def _extract_utterance(
        data: dict,
    ) -> str:

        value = data.get(
            "utterance",
            data.get("input", ""),
        )

        return str(value).strip()

    @staticmethod
    def _map_modality(modality: str) -> str:
        mapping = {
            Intent.MODALITY_SPEECH: "SPEECH",
            Intent.MODALITY_MOTION: "MOTION",
            Intent.MODALITY_TOUCHSCREEN: "TOUCHSCREEN",
            Intent.MODALITY_OTHER: "OTHER",
            Intent.MODALITY_INTERNAL: "INTERNAL",
        }

        return mapping.get(
            modality,
            "UNKNOWN",
        )

    @staticmethod
    def _map_source(
        source: str,
    ) -> str:
        """
        ROS4HRI source describes the agent.

        HumanIntent.source describes how the human intention
        was obtained. For a ROS4HRI speech intent we use
        DIRECT_SPEECH when the source is not the robot.
        """

        if source == "UNKNOWN":
            return "UNKNOWN"

        return "DIRECT_SPEECH"