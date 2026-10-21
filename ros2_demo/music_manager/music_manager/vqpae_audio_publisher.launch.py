from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="music_manager",
                executable="vqpae_audio_publisher",
                name="vqpae_audio_publisher",
                output="screen",
                parameters=[
                    {
                        # Set at launch time
                        "checkpoint": "",
                        "device": "cuda",
                        "input_device": "default",
                        "topic": "/music/vqpae",
                        "publish_mode": "latest",
                        "batch_size": 1,
                        "verbose": True,
                    }
                ],
            )
        ]
    )
