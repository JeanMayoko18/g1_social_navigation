from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'g1_hri'

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
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jean',
    maintainer_email='icmayoko18@gmail.com',
    description='Human-Robot Interaction management and ROS4HRI integration for the Unitree G1',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'hri_manager_node = g1_hri.hri_manager_node:main',
            'ros4hri_tts_adapter = g1_hri.ros4hri_tts_adapter_node:main',
        ],
    },
)