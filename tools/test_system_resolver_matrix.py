#!/usr/bin/env python3
"""
Static matrix validation for the G1 system resolver.

No ROS nodes are started.
No launch files are executed.
No hardware is accessed.

The test validates:
- profile capabilities;
- interaction constraints;
- deployment role placement;
- local role ownership;
- ROS4HRI Humble ownership constraint.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


WS = Path.home() / "g1_ws"
RESOLVER = (
    WS
    / "src"
    / "g1_social_bringup"
    / "launch"
    / "system_resolver.py"
)


def load_resolver():
    spec = importlib.util.spec_from_file_location(
        "g1_system_resolver",
        RESOLVER,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Unable to load resolver: {RESOLVER}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resolver = load_resolver()


EXPECTED_ROLES = {
    "pc_all": {
        "vision_inference": "pc",
        "social_perception": "pc",
        "social_decision": "pc",
        "navigation": "pc",
        "explanation": "pc",
        "speech": "pc",
        "dialogue": "pc",
        "visualization": "pc",
    },
    "robot_all": {
        "vision_inference": "robot",
        "social_perception": "robot",
        "social_decision": "robot",
        "navigation": "robot",
        "explanation": "robot",
        "speech": "robot",
        "dialogue": "robot",
        "visualization": "pc",
    },
    "robot_perception": {
        "vision_inference": "pc",
        "social_perception": "robot",
        "social_decision": "pc",
        "navigation": "pc",
        "explanation": "pc",
        "speech": "robot",
        "dialogue": "pc",
        "visualization": "pc",
    },
    "pc_perception": {
        "vision_inference": "pc",
        "social_perception": "pc",
        "social_decision": "pc",
        "navigation": "pc",
        "explanation": "pc",
        "speech": "robot",
        "dialogue": "pc",
        "visualization": "pc",
    },
}


EXPECTED_PROFILES = {
    "A": {
        "camera_obstacles": False,
        "social_navigation": False,
        "explanation": False,
        "speech_required": False,
    },
    "B": {
        "camera_obstacles": True,
        "social_navigation": False,
        "explanation": False,
        "speech_required": False,
    },
    "C": {
        "camera_obstacles": False,
        "social_navigation": True,
        "explanation": True,
        "speech_required": True,
    },
    "D": {
        "camera_obstacles": True,
        "social_navigation": True,
        "explanation": True,
        "speech_required": True,
    },
}


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def make_plan(
    *,
    profile="C",
    interaction="none",
    deployment="pc_perception",
    host="pc",
    ros_distro="humble",
):
    return resolver.resolve_system(
        profile=profile,
        interaction=interaction,
        deployment=deployment,
        host=host,
        ros_distro=ros_distro,
    )


def expect_rejected(label, **kwargs):
    try:
        make_plan(**kwargs)
    except RuntimeError:
        print(f"[PASS] {label}: rejected")
        return

    raise AssertionError(
        f"{label}: configuration should have been rejected"
    )


def test_profiles():
    print()
    print("PROFILE MATRIX")
    print("-" * 68)

    for profile, expected in EXPECTED_PROFILES.items():
        plan = make_plan(
            profile=profile,
            interaction="none",
        )

        for field, value in expected.items():
            actual = getattr(plan, field)

            check(
                actual == value,
                (
                    f"Profile {profile}: {field}={actual!r}, "
                    f"expected {value!r}"
                ),
            )

        check(
            plan.dialogue is False,
            f"Profile {profile}: none must disable dialogue",
        )

        print(
            f"[PASS] {profile}: "
            f"camera={plan.camera_obstacles} "
            f"social={plan.social_navigation} "
            f"explanation={plan.explanation} "
            f"speech={plan.speech_required}"
        )


def test_interactions():
    print()
    print("INTERACTION MATRIX")
    print("-" * 68)

    for profile in ("C", "D"):
        none_plan = make_plan(
            profile=profile,
            interaction="none",
        )

        dialogue_plan = make_plan(
            profile=profile,
            interaction="dialogue",
        )

        check(
            none_plan.dialogue is False,
            f"{profile}/none unexpectedly enables dialogue",
        )

        check(
            dialogue_plan.dialogue is True,
            f"{profile}/dialogue did not enable dialogue",
        )

        # Dialogue must not alter intrinsic social capabilities.
        for field in (
            "camera_obstacles",
            "social_navigation",
            "explanation",
            "speech_required",
        ):
            check(
                getattr(none_plan, field)
                == getattr(dialogue_plan, field),
                (
                    f"{profile}: dialogue changed intrinsic "
                    f"capability {field}"
                ),
            )

        print(
            f"[PASS] {profile}: none/dialogue separation"
        )

    expect_rejected(
        "A/dialogue",
        profile="A",
        interaction="dialogue",
    )

    expect_rejected(
        "B/dialogue",
        profile="B",
        interaction="dialogue",
    )


def test_deployments():
    print()
    print("DEPLOYMENT MATRIX")
    print("-" * 68)

    for deployment, expected in EXPECTED_ROLES.items():
        plan = make_plan(
            deployment=deployment,
        )

        check(
            plan.role_hosts == expected,
            (
                f"{deployment}: role_hosts mismatch\n"
                f"actual={plan.role_hosts}\n"
                f"expected={expected}"
            ),
        )

        print(
            f"[PASS] {deployment}"
        )


def test_local_role_ownership():
    print()
    print("LOCAL ROLE OWNERSHIP")
    print("-" * 68)

    for deployment, expected in EXPECTED_ROLES.items():
        for host in ("pc", "robot"):
            plan = make_plan(
                deployment=deployment,
                host=host,
            )

            expected_local = {
                role
                for role, role_host in expected.items()
                if role_host == host
            }

            actual_local = set(plan.local_roles)

            check(
                actual_local == expected_local,
                (
                    f"{deployment}/{host}: local_roles mismatch\n"
                    f"actual={sorted(actual_local)}\n"
                    f"expected={sorted(expected_local)}"
                ),
            )

            for role in expected:
                check(
                    plan.owns(role)
                    == (expected[role] == host),
                    (
                        f"{deployment}/{host}: "
                        f"owns({role!r}) mismatch"
                    ),
                )

            print(
                f"[PASS] {deployment}/{host}: "
                f"{', '.join(sorted(actual_local)) or 'none'}"
            )


def test_dialogue_distro_constraint():
    print()
    print("ROS4HRI DISTRIBUTION CONSTRAINT")
    print("-" * 68)

    # pc_perception owns dialogue on PC.
    expect_rejected(
        "pc_perception dialogue owner PC/Foxy",
        profile="C",
        interaction="dialogue",
        deployment="pc_perception",
        host="pc",
        ros_distro="foxy",
    )

    pc_humble = make_plan(
        profile="C",
        interaction="dialogue",
        deployment="pc_perception",
        host="pc",
        ros_distro="humble",
    )

    check(
        pc_humble.dialogue,
        "PC/Humble dialogue should be valid",
    )

    print(
        "[PASS] pc_perception dialogue owner PC/Humble"
    )

    # The robot does not own dialogue in pc_perception, so Foxy is valid
    # on that machine.
    robot_foxy = make_plan(
        profile="C",
        interaction="dialogue",
        deployment="pc_perception",
        host="robot",
        ros_distro="foxy",
    )

    check(
        not robot_foxy.owns("dialogue"),
        "pc_perception robot unexpectedly owns dialogue",
    )

    print(
        "[PASS] pc_perception non-owner robot/Foxy"
    )

    # robot_all moves dialogue ownership to the robot.
    expect_rejected(
        "robot_all dialogue owner robot/Foxy",
        profile="C",
        interaction="dialogue",
        deployment="robot_all",
        host="robot",
        ros_distro="foxy",
    )

    robot_humble = make_plan(
        profile="C",
        interaction="dialogue",
        deployment="robot_all",
        host="robot",
        ros_distro="humble",
    )

    check(
        robot_humble.owns("dialogue"),
        "robot_all robot should own dialogue",
    )

    print(
        "[PASS] robot_all dialogue owner robot/Humble"
    )

    # PC is not the dialogue owner for robot_all. Its distro must not
    # incorrectly invalidate the global deployment.
    pc_foxy = make_plan(
        profile="C",
        interaction="dialogue",
        deployment="robot_all",
        host="pc",
        ros_distro="foxy",
    )

    check(
        not pc_foxy.owns("dialogue"),
        "robot_all PC unexpectedly owns dialogue",
    )

    print(
        "[PASS] robot_all non-owner PC/Foxy"
    )


def test_invalid_inputs():
    print()
    print("INVALID INPUTS")
    print("-" * 68)

    expect_rejected(
        "invalid profile",
        profile="Z",
    )

    expect_rejected(
        "invalid interaction",
        interaction="invalid",
    )

    expect_rejected(
        "invalid deployment",
        deployment="invalid",
    )

    expect_rejected(
        "invalid host",
        host="invalid",
    )

    expect_rejected(
        "invalid ros_distro",
        ros_distro="invalid",
    )


def main():
    print("=" * 68)
    print("STEP 11B - SYSTEM RESOLVER MATRIX")
    print("=" * 68)

    test_profiles()
    test_interactions()
    test_deployments()
    test_local_role_ownership()
    test_dialogue_distro_constraint()
    test_invalid_inputs()

    print()
    print("=" * 68)
    print("STEP 11B PASS")
    print("=" * 68)


if __name__ == "__main__":
    main()
