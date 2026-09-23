import os
from glob import glob
from setuptools import setup

package_name = 'easy_handeye2_charuco'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='Local',
    maintainer_email='local@example.com',
    description='ChArUco board TF publisher, visual test tools and eye-on-base bring-up for easy_handeye2',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'charuco_tf_publisher = easy_handeye2_charuco.charuco_tf_publisher:main',
            'charuco_pose_monitor = easy_handeye2_charuco.charuco_pose_monitor:main',
            'generate_board = easy_handeye2_charuco.generate_board:main',
        ],
    },
)
