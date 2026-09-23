"""Eye-on-base calibration bring-up: RealSense + ChArUco TF + easy_handeye2 handeye_server.

Robot side is NOT started here: handeye_auto_calibrate (easy_handeye2_franka_auto)
talks to the robot through pylibfranka and publishes robot_base_frame ->
robot_effector_frame itself. So:

  * franka_ros2 / franka_bringup must NOT be running (pylibfranka needs FCI).
  * robot_base_frame / robot_effector_frame must match handeye_auto_calibrate's
    --robot-base-frame / --robot-effector-frame.
  * handeye_server only advertises its services once BOTH transforms are in TF,
    so the board must be visible to the camera when handeye_auto_calibrate starts.

    ros2 launch easy_handeye2_charuco eye_on_base_calib.launch.py name:=fr3_eob [debug:=true]
    ros2 run easy_handeye2_franka_auto handeye_auto_calibrate --name fr3_eob \\
        --robot-config <robot.yaml> --robot-base-frame fr3_link0 --robot-effector-frame handeye_ee

tracking_base_frame defaults to camera_link (root of the RealSense TF tree): the
saved result is fr3_link0 -> camera_link, which publish.launch.py can then hang the
whole camera tree from without giving any frame two parents. This is also why the
dummy base->camera static TF from easy_handeye2's calibrate.launch.py is not used.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, OrSubstitution, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    args = [
        DeclareLaunchArgument('name', default_value='fr3_eye_on_base',
                              description='calibration name (file under ~/.ros2/easy_handeye2/calibrations)'),
        DeclareLaunchArgument('robot_base_frame', default_value='fr3_link0'),
        DeclareLaunchArgument('robot_effector_frame', default_value='handeye_ee'),
        DeclareLaunchArgument('tracking_base_frame', default_value='camera_link'),
        DeclareLaunchArgument('tracking_marker_frame', default_value='charuco'),
        DeclareLaunchArgument('debug', default_value='false',
                              description='visualize detection: RViz (camera + charuco TF axes, debug image) '
                                          'and rqt_image_view on the annotated image'),
        DeclareLaunchArgument('use_rviz', default_value='false', description='RViz only (implied by debug)'),
        DeclareLaunchArgument('use_image_view', default_value='false',
                              description='rqt_image_view only (implied by debug)'),
        DeclareLaunchArgument('use_rqt', default_value='false',
                              description='easy_handeye2 rqt calibrator GUI (sample list; do not click take sample)'),
    ]

    camera_and_detector = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('easy_handeye2_charuco'), 'launch', 'charuco_view.launch.py'])),
        launch_arguments={
            'tracking_marker_frame': LaunchConfiguration('tracking_marker_frame'),
            'use_rviz': OrSubstitution(LaunchConfiguration('debug'), LaunchConfiguration('use_rviz')),
            'use_image_view': OrSubstitution(LaunchConfiguration('debug'),
                                             LaunchConfiguration('use_image_view')),
        }.items(),
    )

    handeye_params = [{
        'name': LaunchConfiguration('name'),
        'calibration_type': 'eye_on_base',
        'robot_base_frame': LaunchConfiguration('robot_base_frame'),
        'robot_effector_frame': LaunchConfiguration('robot_effector_frame'),
        'tracking_base_frame': LaunchConfiguration('tracking_base_frame'),
        'tracking_marker_frame': LaunchConfiguration('tracking_marker_frame'),
    }]

    handeye_server = Node(package='easy_handeye2', executable='handeye_server', name='handeye_server',
                          output='screen', parameters=handeye_params)

    rqt_calibrator = Node(package='easy_handeye2', executable='rqt_calibrator.py',
                          name='handeye_rqt_calibrator', parameters=handeye_params,
                          condition=IfCondition(LaunchConfiguration('use_rqt')))

    return LaunchDescription(args + [camera_and_detector, handeye_server, rqt_calibrator])
