#!/usr/bin/env python3

"""
Validate the image network boundary for every deployment.

This test is purely static. It does not start ROS, DDS, camera hardware,
YOLO, social perception, or visualization.

Expected camera placement:
    camera host = robot

Expected network matrix:
    pc_all:
        RGB        -> network
        depth/info -> network

    pc_perception:
        RGB        -> network
        depth/info -> network

    robot_perception:
        RGB        -> network
        depth/info -> local

    robot_all:
        RGB        -> local
        depth/info -> local
"""

import importlib.util
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]

RESOLVER_PATH = (
    WORKSPACE
    / "src"
    / "g1_social_bringup"
    / "launch"
    / "system_resolver.py"
)

CAMERA_HOST = "robot"


def load_resolver():
    spec = importlib.util.spec_from_file_location(
        "g1_system_resolver",
        RESOLVER_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Unable to load resolver: {RESOLVER_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_matrix():
    return {
        "pc_all": {
            "pc": (True, True),
            "robot": (False, False),
        },
        "pc_perception": {
            "pc": (True, True),
            "robot": (False, False),
        },
        "robot_perception": {
            "pc": (True, False),
            "robot": (False, False),
        },
        "robot_all": {
            "pc": (False, False),
            "robot": (False, False),
        },
    }


def resolve_boundary(plan):
    vision_is_local = plan.owns("vision_inference")
    social_perception_is_local = plan.owns(
        "social_perception"
    )

    remote_rgb_required = (
        vision_is_local
        and plan.host != CAMERA_HOST
    )

    remote_depth_required = (
        social_perception_is_local
        and plan.host != CAMERA_HOST
    )

    return remote_rgb_required, remote_depth_required


def main():
    resolver = load_resolver()

    expected = expected_matrix()

    print("=" * 72)
    print("STEP 11D.1 - IMAGE NETWORK BOUNDARY MATRIX")
    print("=" * 72)

    for deployment, hosts in expected.items():
        for host, expected_boundary in hosts.items():

            # Profile C activates the complete social perception path
            # without adding the camera-obstacle-layer distinction.
            #
            # Humble is used so the test remains compatible with every
            # possible role placement even though interaction is none.
            plan = resolver.resolve_system(
                profile="C",
                interaction="none",
                deployment=deployment,
                host=host,
                ros_distro="humble",
            )

            actual = resolve_boundary(plan)

            if actual != expected_boundary:
                raise AssertionError(
                    f"{deployment}/{host}: "
                    f"expected RGB/depth={expected_boundary}, "
                    f"received {actual}"
                )

            rgb = "network" if actual[0] else "local/none"
            depth = "network" if actual[1] else "local/none"

            print(
                f"[PASS] {deployment:18s} "
                f"host={host:5s} "
                f"RGB={rgb:10s} "
                f"depth/info={depth}"
            )

    print()
    print("=" * 72)
    print("STEP 11D.1 PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()