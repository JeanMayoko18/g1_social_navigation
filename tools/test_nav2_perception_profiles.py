#!/usr/bin/env python3
"""
Static validation of Nav2 perception profiles A/B/C/D.

No ROS nodes are started.

This test validates:
- every perception profile exists;
- local_costmap plugin lists are exact and ordered;
- camera/social layer enable flags match the profile contract;
- inflation_layer is always last;
- unsupported extra plugins are rejected.
"""

from pathlib import Path

import yaml


WS = Path.home() / "g1_ws"
PROFILE_DIR = (
    WS
    / "src"
    / "g1_nav2"
    / "params"
    / "perception_profiles"
)


EXPECTED = {
    "A": {
        "plugins": [
            "voxel_layer",
            "inflation_layer",
        ],
        "camera": False,
        "social": False,
    },
    "B": {
        "plugins": [
            "obstacle_layer",
            "voxel_layer",
            "camera_obstacle_layer",
            "inflation_layer",
        ],
        "camera": True,
        "social": False,
    },
    "C": {
        "plugins": [
            "voxel_layer",
            "social_layer",
            "inflation_layer",
        ],
        "camera": False,
        "social": True,
    },
    "D": {
        "plugins": [
            "voxel_layer",
            "camera_obstacle_layer",
            "social_layer",
            "inflation_layer",
        ],
        "camera": True,
        "social": True,
    },
}


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def load_profile(profile):
    path = PROFILE_DIR / f"perception_{profile.lower()}.yaml"

    check(
        path.is_file(),
        f"Profile {profile}: missing {path}",
    )

    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    check(
        isinstance(data, dict),
        f"Profile {profile}: YAML root must be a mapping",
    )

    return path, data


def local_costmap_params(profile, data):
    try:
        params = data["local_costmap"]["local_costmap"]["ros__parameters"]
    except (KeyError, TypeError) as exc:
        raise AssertionError(
            f"Profile {profile}: expected "
            "local_costmap.local_costmap.ros__parameters"
        ) from exc

    check(
        isinstance(params, dict),
        f"Profile {profile}: local costmap parameters must be a mapping",
    )

    return params


def enabled_value(profile, params, layer):
    block = params.get(layer)

    check(
        isinstance(block, dict),
        f"Profile {profile}: missing {layer} configuration block",
    )

    check(
        "enabled" in block,
        f"Profile {profile}: {layer}.enabled is missing",
    )

    value = block["enabled"]

    check(
        isinstance(value, bool),
        (
            f"Profile {profile}: {layer}.enabled must be a YAML "
            f"boolean, got {value!r}"
        ),
    )

    return value


def test_profile(profile, expected):
    path, data = load_profile(profile)
    params = local_costmap_params(profile, data)

    plugins = params.get("plugins")

    check(
        isinstance(plugins, list),
        f"Profile {profile}: plugins must be a list",
    )

    check(
        plugins == expected["plugins"],
        (
            f"Profile {profile}: plugin order mismatch\n"
            f"actual   = {plugins}\n"
            f"expected = {expected['plugins']}"
        ),
    )

    check(
        len(plugins) == len(set(plugins)),
        f"Profile {profile}: duplicate plugins: {plugins}",
    )

    check(
        plugins[-1] == "inflation_layer",
        f"Profile {profile}: inflation_layer must be last",
    )

    camera_enabled = enabled_value(
        profile,
        params,
        "camera_obstacle_layer",
    )

    social_enabled = enabled_value(
        profile,
        params,
        "social_layer",
    )

    check(
        camera_enabled == expected["camera"],
        (
            f"Profile {profile}: camera_obstacle_layer.enabled="
            f"{camera_enabled}, expected {expected['camera']}"
        ),
    )

    check(
        social_enabled == expected["social"],
        (
            f"Profile {profile}: social_layer.enabled="
            f"{social_enabled}, expected {expected['social']}"
        ),
    )

    check(
        ("camera_obstacle_layer" in plugins)
        == expected["camera"],
        (
            f"Profile {profile}: camera layer plugin membership "
            "does not match activation"
        ),
    )

    check(
        ("social_layer" in plugins)
        == expected["social"],
        (
            f"Profile {profile}: social layer plugin membership "
            "does not match activation"
        ),
    )

    print(
        f"[PASS] {profile}: "
        f"plugins={plugins} "
        f"camera={camera_enabled} "
        f"social={social_enabled}"
    )

    return path


def main():
    print("=" * 72)
    print("STEP 11C.1 - NAV2 PERCEPTION PROFILE MATRIX")
    print("=" * 72)

    tested_paths = set()

    for profile, expected in EXPECTED.items():
        path = test_profile(profile, expected)
        tested_paths.add(path.resolve())

    actual_profile_files = {
        path.resolve()
        for path in PROFILE_DIR.glob("perception_*.yaml")
    }

    check(
        actual_profile_files == tested_paths,
        (
            "Unexpected or missing perception profile files.\n"
            f"actual={sorted(str(p) for p in actual_profile_files)}\n"
            f"expected={sorted(str(p) for p in tested_paths)}"
        ),
    )

    print()
    print("=" * 72)
    print("STEP 11C.1 PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()
