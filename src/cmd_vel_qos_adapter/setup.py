from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'cmd_vel_qos_adapter'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # Launch files
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jean',
    maintainer_email='icmayoko18@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'qos_adapter_node = cmd_vel_qos_adapter.qos_adapter_node:main',
            "cmd_vel_rl_scaler = cmd_vel_qos_adapter.cmd_vel_rl_scaler:main",
            "locomotion_calibration_node = cmd_vel_qos_adapter.locomotion_calibration_node:main"
        ],
    },
)
