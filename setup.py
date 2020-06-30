#!/usr/bin/env python
# fmt: off
"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Click>=7.0',
    'pytorch-lightning',
    'wandb',
    'omegaconf',
    'Pillow',
    'torchvision'
]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Laksh",
    author_email='lakshaithanii@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="MoCo PyTorch Lightning reimplementation",
    entry_points={
        'console_scripts': [
            'moco_pytorch_lightning=moco_pytorch_lightning.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='moco_pytorch_lightning',
    name='moco_pytorch_lightning',
    packages=find_packages(include=['moco_pytorch_lightning', 'moco_pytorch_lightning.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/aced125/moco_pytorch_lightning',
    version='0.1.0',
    zip_safe=False,
)
