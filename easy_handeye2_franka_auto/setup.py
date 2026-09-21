import os
from glob import glob
from setuptools import setup

package_name = 'easy_handeye2_franka_auto'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools', 'numpy', 'torch', 'PyYAML'],
    zip_safe=True,
    maintainer='Local',
    maintainer_email='local@example.com',
    description='Automated eye-on-base hand-eye sampling via pylibfranka',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'handeye_auto_calibrate = easy_handeye2_franka_auto.handeye_auto_calibrate:main',
            'handeye_motion_test = easy_handeye2_franka_auto.handeye_motion_test:main',
        ],
    },
)
