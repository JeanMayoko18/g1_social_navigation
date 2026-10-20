import os
from glob import glob

from setuptools import find_packages, setup


package_name = "g1_robot_explainability"


setup(
    name=package_name,
    version="0.0.0",

    packages=find_packages(
        exclude=["test"]
    ),

    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
        (
            os.path.join(
                "share",
                package_name,
                "config",
            ),
            glob("config/*.yaml"),
        ),
        (
            os.path.join(
                "share",
                package_name,
                "launch",
            ),
            glob("launch/*.launch.py"),
        ),
    ],

    install_requires=[
        "setuptools",
    ],

    zip_safe=True,

    maintainer="jean",
    maintainer_email="icmayoko18@gmail.com",

    description=(
        "Explainability and natural-language explanation stack for "
        "deterministic G1 robot navigation decisions."
    ),

    license="Proprietary",

    entry_points={
        "console_scripts": [
            (
                "explanation_node = "
                "g1_robot_explainability.explanation_node:main"
            ),
            (
                "piper_tts = "
                "g1_robot_explainability.tts_node:main"
            ),
            (
                "pc_audio_fallback = "
                "g1_robot_explainability.pc_audio_fallback_node:main"
            ),
        ],
    },
)