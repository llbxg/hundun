import os
from setuptools import setup, find_packages
import sys


if sys.version_info[:2] < (3, 8):
    raise RuntimeError('Python version >= 3.8 required.')


def _requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()


NAME = 'hundun'

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, NAME, '__version__.py')) as f:
    exec(f.read(), (version := {}))

with open('README.md') as f:
    long_description = f.read()

setup(
    name=NAME,
    version=version['__version__'],
    description='hundun is a python library for the exploration of chaos.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://github.com/llbxg/hundun',
    author='kosh',
    author_email='llllbxg@gmial.com',
    license='MIT',
    packages=find_packages(),
    install_requires=_requirements(),
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.8',
    classifiers=[
        'Topic :: Scientific/Engineering :: Physics'
    ]
    )
