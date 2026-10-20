from setuptools import find_packages, setup

package_name = 'g1_unitree_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jean',
    maintainer_email='icmayoko18@gmail.com',
    description='Bridge Unitree proprietary video streams to standard ROS 2 image topics.',
    license='Proprietary',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "front_video_bridge = g1_unitree_bridge.front_video_bridge:main",
        ],
    },
)