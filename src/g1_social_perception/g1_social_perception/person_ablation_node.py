
#!/usr/bin/env python3
"""
Experimental M1-M4 identity ablation for the G1.

All variants receive the same enriched detections,
timestamp and camera geometry.

Each variant owns an independent MultimodalIdentityEngine.

No experiment publishes to production identity topics.
No persistent identity memory is written.

IMPORTANT:
Weight ablation changes association weighting, but
does not necessarily remove every use of a cue from
initialization, gating or dormant-track management.
"""

import copy
import json
import time
import uuid

from dataclasses import replace

import rclpy

from std_msgs.msg import String
from g1_social_msgs.msg import PersonDetectionArray

from g1_qos_manager.profiles import reliable_qos

from g1_social_perception.person_tracking_node import (
    PersonTrackingNode,
)

from g1_social_perception.identity.multimodal_engine import (
    MultimodalIdentityEngine,
)

from g1_social_perception.mathematical_model.types import (
    AssociationWeights,
)


VARIANTS = {
    "M1": (1.00, 0.00, 0.00, 0.00),
    "M2": (0.50, 0.50, 0.00, 0.00),
    "M3": (0.30, 0.30, 0.40, 0.00),
    "M4": (0.25, 0.35, 0.30, 0.10),
}


class PersonAblationNode(PersonTrackingNode):

    def _configure_old(self):
        super()._configure()

        # Never use the production identity output topic.
        self.output_topic = (
            "/social_perception/benchmark/ablation/unused"
        )

        # The experimental node does not use SQLite.
        self.persistent_memory_enabled = False

    def _configure(self):
        super()._configure()

        # Experimental identities must never enter production.
        self.output_topic = (
            "/social_perception/benchmark/ablation/unused"
        )

        # Disable persistence for every ablation engine.
        self.persistent_memory_enabled = False

        if self.output_topic in (
            "/social_perception/detections_identified",
            "/social_perception/detections_identified_odom",
        ):
            raise RuntimeError(
                "ABLATION_PRODUCTION_OUTPUT_FORBIDDEN"
            )

    def _initialize_state(self):
        super()._initialize_state()

        self._ablation_domain = self.tracking_session_id
        self._initialize_ablation_engines()

    def _initialize_ablation_engines(self):
        self.engines = {}
        self.sessions = {}

        base_config = self._engine_config()

        for name, values in VARIANTS.items():
            g, m, a, h = values

            weights = AssociationWeights(
                g=g,
                m=m,
                a=a,
                h=h,
                eta_upper=float(self.eta_upper),
                eta_lower=float(self.eta_lower),
                alpha=float(self.alpha),
                beta=float(self.beta),
            )

            config = replace(
                base_config,
                weights=weights,
            )

            session_id = uuid.uuid4().hex

            self.sessions[name] = session_id

            self.engines[name] = MultimodalIdentityEngine(
                session_id=session_id,
                config=config,
            )

    def _create_interfaces(self):
        # Reuse the production input validation, ordered queue,
        # GeometryAdapter and QoS setup.
        #
        # The inherited publisher is created on the unused
        # benchmark topic, never on detections_identified.
        super()._create_interfaces()


        # Fail closed if an inherited publisher targets production.
        for publisher in self.publishers:
            if publisher.topic_name in (
                "/social_perception/detections_identified",
                "/social_perception/detections_identified_odom",
                "/social_perception/tracked_persons",
                "/social_perception/tracked_persons_odom",
            ):
                raise RuntimeError(
                    "ABLATION_PRODUCTION_PUBLISHER_DETECTED:"
                    + publisher.topic_name
                )


        self.ablation_publishers = {}

        for name in VARIANTS:
            topic = (
                "/social_perception/benchmark/ablation/"
                + name.lower()
            )

            self.ablation_publishers[name] = (
                self.create_publisher(
                    PersonDetectionArray,
                    topic,
                    reliable_qos(depth=10),
                )
            )

        self.ablation_timing_publisher = (
            self.create_publisher(
                String,
                "/social_perception/benchmark/ablation/timing",
                reliable_qos(depth=100),
            )
        )

    def _process_observation(self, entry, geometry):
        """
        Run each independent engine on the same source frame.

        A failure in one variant is reported independently.
        """
        if self._ablation_domain != self.tracking_session_id:
            self._ablation_domain = self.tracking_session_id
            self._initialize_ablation_engines()

        msg = entry.message

        candidates = self._decode_candidates(msg)

        receipt = time.monotonic()

        for name, weights in VARIANTS.items():
            engine = self.engines[name]

            started_ns = time.perf_counter_ns()

            try:
                decisions = engine.process(
                    candidates,
                    stamp_ns=entry.source_ns,
                    receipt=receipt,
                    current_view=geometry.view,
                )

                if len(decisions) != len(msg.candidates):
                    raise RuntimeError(
                        "DECISION_COUNT_MISMATCH"
                    )

                processing_ms = (
                    time.perf_counter_ns() - started_ns
                ) / 1_000_000.0

                output = copy.deepcopy(msg)

                output.processing_stage = (
                    PersonDetectionArray.IDENTITY
                )

                output.identity_evaluated = True
                output.tracking_session_id = (
                    self.sessions[name]
                )

                for candidate, decision in zip(
                    output.candidates,
                    decisions,
                ):
                    candidate.person_id = decision.person_id
                    candidate.identity_state = decision.state
                    candidate.identity_confirmed = (
                        decision.confirmed
                    )
                    candidate.identity_generation = (
                        decision.generation
                    )

                self.ablation_publishers[name].publish(
                    output
                )

                trace = {
                    "schema_version": 1,
                    "method": name,
                    "tracking_session_id": (
                        self.sessions[name]
                    ),
                    "source_session_id": (
                        entry.source_session_id
                    ),
                    "source_sequence": (
                        entry.source_sequence
                    ),
                    "source_stamp_ns": entry.source_ns,
                    "source_frame_id": (
                        msg.image_header.frame_id
                    ),
                    "candidate_count": len(candidates),
                    "weights": {
                        "g": weights[0],
                        "m": weights[1],
                        "a": weights[2],
                        "h": weights[3],
                    },
                    "processing_ms": processing_ms,
                    "geometry_status": geometry.reason,
                    "status": "OK",
                }

            except Exception as exc:
                self.get_logger().error(
                    f"[ABLATION {name}] "
                    f"{type(exc).__name__}: {exc}"
                )

                trace = {
                    "schema_version": 1,
                    "method": name,
                    "source_session_id": (
                        entry.source_session_id
                    ),
                    "source_sequence": (
                        entry.source_sequence
                    ),
                    "source_stamp_ns": entry.source_ns,
                    "status": "ERROR",
                    "error_type": type(exc).__name__,
                }

            self.ablation_timing_publisher.publish(
                String(
                    data=json.dumps(
                        trace,
                        allow_nan=False,
                    )
                )
            )


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = PersonAblationNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.discard_pending_on_shutdown()

            if node.persistent_memory is not None:
                node.persistent_memory.close()

            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
