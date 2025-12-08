from setuptools import find_packages, setup

package_name = 'inference'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # install launch and config files
        ('share/' + package_name + '/launch', ['launch/inference.launch.py']),
        ('share/' + package_name + '/config', ['config/params.yaml']),
    ],
    install_requires=[
        'setuptools',
        'sensor_msgs',
        'std_msgs',
        'custom_msgs',
    ],
    zip_safe=True,
    maintainer='NikosAdamopoulos',
    maintainer_email='nickadamopoulos2004@gmail.com',
    description='Determines player move from raw image',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'inference = inference.inference_node:main',
        ],
    },
)
