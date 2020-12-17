import setuptools


with open('requirements.txt', 'r') as f:
    install_requires = f.read()


packages = [
    'transformers_pretraining',
    'transformers_pretraining.bin',
]

entry_points = {
    'console_scripts': [
        'transformers-pretraining=transformers_pretraining.__main__:main'
    ]
}

setuptools.setup(
    name='transformers_pretraining',
    version='0.0.1',
    python_requires='>=3.7',
    install_requires=install_requires,
    packages=packages,
    entry_points=entry_points,
)
