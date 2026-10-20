from setuptools import find_packages, setup

package_name = "g1_social_common"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jean",
    maintainer_email="icmayoko18@gmail.com",
    description="Shared semantic contracts for the G1 social navigation stack.",
    license="Proprietary",
)