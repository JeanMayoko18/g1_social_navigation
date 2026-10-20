from glob import glob
import os

from setuptools import find_packages
from setuptools import setup


package_name = 'g1_camera_obstacles'


setup(
    name=package_name,
    version='0.1.0',

    packages=find_packages(
        exclude=['test']
    ),

    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        (
            'share/' + package_name,
            ['package.xml'],
        ),
        (
            os.path.join(
                'share',
                package_name,
                'launch',
            ),
            glob('launch/*.launch.py'),
        ),
        (
            os.path.join(
                'share',
                package_name,
                'config',
            ),
            glob('config/*.yaml'),
        ),
    ],

    install_requires=[
        'setuptools',
        'numpy',
    ],

    zip_safe=True,

    maintainer='jean',
    maintainer_email='jean@todo.todo',

    description=(
        'Optional RealSense depth obstacle perception '
        'for G1 Nav2 navigation.'
    ),

    license='TODO',

    entry_points={
        'console_scripts': [
            (
                'camera_obstacle_node = '
                'g1_camera_obstacles.camera_obstacle_node:main'
            ),
        ],
    },
)