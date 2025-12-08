from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('game_sim')
    params_file = os.path.join(pkg_share, 'config', 'params.yaml')

    return LaunchDescription([
        # Node(package='game_sim', executable='game_sim', name='game_sim_node', parameters=[params_file]),
    ])