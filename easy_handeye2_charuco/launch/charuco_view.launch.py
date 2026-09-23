"""RealSense color camera + ChArUco TF publisher, with RViz / image view to check the board axes.

    ros2 launch easy_handeye2_charuco charuco_view.launch.py
    ros2 run easy_handeye2_charuco charuco_pose_monitor     # detection rate + jitter

Also included by eye_on_base_calib.launch.py for the camera/detector half of calibration.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

DEBUG_IMAGE_TOPIC = '/charuco_tf_publisher/debug_image'


def generate_launch_description():
    share = FindPackageShare('easy_handeye2_charuco')
    args = [
        DeclareLaunchArgument('use_camera', default_value='true',
                              description='start realsense2_camera (false if already running)'),
        DeclareLaunchArgument('color_profile', default_value='1280,720,30',
                              description='RealSense rgb_camera.color_profile (w,h,fps)'),
        DeclareLaunchArgument('image_topic', default_value='/camera/camera/color/image_raw'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera/camera/color/camera_info'),
        DeclareLaunchArgument('tracking_marker_frame', default_value='charuco',
                              description='TF child frame published for the board'),
        DeclareLaunchArgument('board_config',
                              default_value=PathJoinSubstitution([share, 'config', 'charuco_board.yaml']),
                              description='board geometry yaml (ros__parameters for charuco_tf_publisher)'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('use_image_view', default_value='false',
                              description='also open rqt_image_view on the debug image'),
    ]

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('realsense2_camera'), 'launch', 'rs_launch.py'])),
        launch_arguments={
            'enable_depth': 'false',
            'rgb_camera.color_profile': LaunchConfiguration('color_profile'),
            'publish_tf': 'true',
        }.items(),
        condition=IfCondition(LaunchConfiguration('use_camera')),
    )

    detector = Node(
        package='easy_handeye2_charuco',
        executable='charuco_tf_publisher',
        name='charuco_tf_publisher',
        output='screen',
        parameters=[
            LaunchConfiguration('board_config'),
            {
                'image_topic': LaunchConfiguration('image_topic'),
                'camera_info_topic': LaunchConfiguration('camera_info_topic'),
                'target_frame': LaunchConfiguration('tracking_marker_frame'),
            },
        ],
    )

    rviz = Node(
        package='rviz2', executable='rviz2', name='charuco_rviz',
        arguments=['-d', PathJoinSubstitution([share, 'config', 'charuco_view.rviz'])],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    image_view = Node(
        package='rqt_image_view', executable='rqt_image_view', name='charuco_image_view',
        arguments=[DEBUG_IMAGE_TOPIC],
        condition=IfCondition(LaunchConfiguration('use_image_view')),
    )

    return LaunchDescription(args + [camera, detector, rviz, image_view])
