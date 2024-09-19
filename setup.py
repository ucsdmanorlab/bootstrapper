from setuptools import setup, find_packages

from pathlib import Path

this_directory = Path(__file__).parent  

long_description = (this_directory / "README.md").read_text()

setup(
    name='bootstrapper',
    version='0.1.0',
    description='Bootstrap volume segmentations from sparse annotations',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Vijay Venu Thiyagarajan',
    author_email='vvenu@utexas.edu',
    url='https://github.com/ucsdmanorlab/bootstrapper',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'gunpowder>=1.4',
        'daisy>=1.2',
        'numpy',
        'scipy',
        'scikit-image',
        'torch',
        'zarr',
        'waterz',
        'neuroglancer'
        'click'
    ],
    entry_points={
        'console_scripts': [
            'bs=bootstrapper.cli:main',
        ],
    },
)