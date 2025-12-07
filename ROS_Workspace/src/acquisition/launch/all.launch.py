from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='acquisition', executable='acquisition_node', name='acquisition_node'),
        Node(package='inference', executable='inference_node', name='inference_node'),
        Node(package='visualization', executable='visualization_node', name='visualization_node'),
        # Node(package='game_sim', executable='game_sim', name='game_sim_node'),
    ])