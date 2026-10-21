from setuptools import find_packages, setup

package_name = "music_manager"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="you",
    maintainer_email="you@example.com",
    description="Real-time system/microphone audio capture and VQPAE feature publishing.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "vqpae_audio_publisher = music_manager.vqpae_audio_publisher:main",
        ],
    },
)
