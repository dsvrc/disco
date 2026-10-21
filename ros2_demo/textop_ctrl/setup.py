from setuptools import setup

package_name = 'textop_ctrl'

setup(
    name=package_name,
    version='0.0.1',
    packages=[],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/msg', ['msg/Gmr.msg']),
        ('share/' + package_name + '/msg', ['msg/Vqpae.msg']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='DiscoDF',
    maintainer_email='todo@example.com',
    description='Custom messages for DiscoDF ROS2 demo.',
    license='MIT',
)
