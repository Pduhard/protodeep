from setuptools import setup

setup(
    name='Protodeep',
    version='0.1.0',
    author='Pduhard',
    description='homemade deeplearning library for learning purpose',
    keywords='lib',
    packages=[
        'Protodeep',
        'Protodeep.activations',
        'Protodeep.callbacks',
        'Protodeep.initializers',
        'Protodeep.regularizers',
        'Protodeep.layers',
        'Protodeep.layers.connectors',
        'Protodeep.losses',
        'Protodeep.metrics',
        'Protodeep.model',
        'Protodeep.optimizers',
        'Protodeep.utils',
    ],
    long_description=open('README.md').read(),
    install_requires=[
        'numpy==1.19.3',
        'numba==0.43.1'
    ]
)