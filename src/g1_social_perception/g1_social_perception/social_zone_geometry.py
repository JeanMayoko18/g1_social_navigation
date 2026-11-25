
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from g1_social_msgs.msg import SocialZone


@dataclass(frozen=True)
class IndividualSocialZoneResult:
    """
    Qualified social-zone result for one tracked person.

    A valid zone does not imply complete scene coverage.
    Directional reference is distinct from body orientation.
    """

    tracking_session_id: str
    person_id: str
    identity_generation: int

    status: str
    reason: str

    zone: Optional[SocialZone]

    direction_basis: str
    body_orientation_known: bool


def _validate_parameters(
    *,
    static_radius_m,
    dynamic_speed_threshold_mps,
    dynamic_base_longitudinal_radius_m,
    dynamic_base_lateral_radius_m,
    dynamic_speed_gain,
    maximum_longitudinal_radius_m,
):
    """Validate the shared social-zone model parameters."""

    values = (
        float(static_radius_m),
        float(dynamic_speed_threshold_mps),
        float(dynamic_base_longitudinal_radius_m),
        float(dynamic_base_lateral_radius_m),
        float(dynamic_speed_gain),
        float(maximum_longitudinal_radius_m),
    )

    if not all(math.isfinite(value) for value in values):
        raise ValueError("NON_FINITE_ZONE_PARAMETER")

    if (
        values[0] <= 0.0
        or values[1] < 0.0
        or values[2] <= 0.0
        or values[3] <= 0.0
        or values[4] < 0.0
        or values[5] <= 0.0
    ):
        raise ValueError("INVALID_ZONE_PARAMETER")

    return values


def build_individual_social_zone(
    person,
    *,
    static_radius_m,
    dynamic_speed_threshold_mps,
    dynamic_base_longitudinal_radius_m,
    dynamic_base_lateral_radius_m,
    dynamic_speed_gain,
    maximum_longitudinal_radius_m,
) -> IndividualSocialZoneResult:
    """
    Build and qualify social geometry for one person.

    Validate identity, current observation and geometry
    independently of all other tracked persons.

    The function does not infer body orientation,
    reconstruct missing observations or apply scene-level
    completeness rules.
    """

    parameters = _validate_parameters(
        static_radius_m=static_radius_m,
        dynamic_speed_threshold_mps=(
            dynamic_speed_threshold_mps
        ),
        dynamic_base_longitudinal_radius_m=(
            dynamic_base_longitudinal_radius_m
        ),
        dynamic_base_lateral_radius_m=(
            dynamic_base_lateral_radius_m
        ),
        dynamic_speed_gain=dynamic_speed_gain,
        maximum_longitudinal_radius_m=(
            maximum_longitudinal_radius_m
        ),
    )

    (
        static_radius,
        speed_threshold,
        dynamic_longitudinal,
        dynamic_lateral,
        speed_gain,
        maximum_longitudinal,
    ) = parameters

    session_id = str(person.tracking_session_id).strip()
    person_id = str(person.id).strip()
    generation = int(person.identity_generation)

    identity = {
        "tracking_session_id": session_id,
        "person_id": person_id,
        "identity_generation": generation,
    }

    def unavailable(reason):
        return IndividualSocialZoneResult(
            **identity,
            status="UNAVAILABLE",
            reason=reason,
            zone=None,
            direction_basis="UNKNOWN",
            body_orientation_known=False,
        )

    if (
        not session_id
        or not person_id
        or generation <= 0
    ):
        return unavailable("INVALID_IDENTITY")

    if str(person.tracking_state) != "TRACKED":
        return unavailable("PERSON_NOT_TRACKED")

    if not bool(person.observed_in_current_array):
        return unavailable("NO_CURRENT_OBSERVATION")

    stamp = person.last_observation_stamp

    if (
        int(stamp.sec) < 0
        or int(stamp.nanosec) < 0
        or int(stamp.nanosec) >= 1_000_000_000
        or (
            int(stamp.sec) == 0
            and int(stamp.nanosec) == 0
        )
    ):
        return unavailable("INVALID_OBSERVATION_STAMP")

    speed = float(person.speed)

    position = (
        float(person.position.x),
        float(person.position.y),
        float(person.position.z),
    )

    if (
        not math.isfinite(speed)
        or speed < 0.0
        or not all(math.isfinite(v) for v in position)
    ):
        return unavailable("INVALID_PERSON_GEOMETRY")

    zone = SocialZone()

    zone.person_id = person_id
    zone.tracking_session_id = session_id
    zone.identity_generation = generation

    zone.tracking_state = "TRACKED"
    zone.person_speed = speed

    zone.center.x = position[0]
    zone.center.y = position[1]
    zone.center.z = position[2]

    zone.observation_stamp.sec = int(stamp.sec)
    zone.observation_stamp.nanosec = int(stamp.nanosec)

    if speed < speed_threshold:
        # A circular zone does not require body orientation.
        # Heading zero is a geometric convention only.
        zone.zone_type = "STATIC"
        zone.heading = 0.0
        zone.radius_longitudinal = static_radius
        zone.radius_lateral = static_radius

        direction_basis = "UNKNOWN"

    else:
        motion_heading = float(person.motion_heading)

        if not math.isfinite(motion_heading):
            return unavailable("INVALID_MOTION_HEADING")

        longitudinal_radius = min(
            dynamic_longitudinal + speed_gain * speed,
            maximum_longitudinal,
        )

        if longitudinal_radius <= 0.0:
            return unavailable("INVALID_DYNAMIC_RADIUS")

        zone.zone_type = "DYNAMIC"

        zone.heading = math.atan2(
            math.sin(motion_heading),
            math.cos(motion_heading),
        )

        zone.radius_longitudinal = longitudinal_radius
        zone.radius_lateral = dynamic_lateral

        # Motion direction is not necessarily body orientation.
        direction_basis = "MOTION_DIRECTION"

    return IndividualSocialZoneResult(
        **identity,
        status="OBSERVED",
        reason="VALID_CURRENT_OBSERVATION",
        zone=zone,
        direction_basis=direction_basis,
        body_orientation_known=False,
    )


def build_social_zone(
    person,
    *,
    static_radius_m,
    dynamic_speed_threshold_mps,
    dynamic_base_longitudinal_radius_m,
    dynamic_base_lateral_radius_m,
    dynamic_speed_gain,
    maximum_longitudinal_radius_m,
) -> SocialZone:
    """
    Preserve the original SocialZone-producing interface.

    A missing or invalid individual observation cannot be
    represented as a current observed social zone.
    """

    result = build_individual_social_zone(
        person,
        static_radius_m=static_radius_m,
        dynamic_speed_threshold_mps=(
            dynamic_speed_threshold_mps
        ),
        dynamic_base_longitudinal_radius_m=(
            dynamic_base_longitudinal_radius_m
        ),
        dynamic_base_lateral_radius_m=(
            dynamic_base_lateral_radius_m
        ),
        dynamic_speed_gain=dynamic_speed_gain,
        maximum_longitudinal_radius_m=(
            maximum_longitudinal_radius_m
        ),
    )

    if result.zone is None:
        raise ValueError(
            f"INDIVIDUAL_SOCIAL_ZONE_UNAVAILABLE:"
            f"{result.reason}"
        )

    return result.zone
