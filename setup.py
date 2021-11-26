from setuptools import setup

setup(
    name='Spectral Embedding',
    url='https://github.com/iggallagher/Spectral-Embedding',
    author='Ian Gallagher',
    author_email='ian.gallagher@bristol.ac.uk',
    packages=['spectral_embedding'],
    install_requires=['numpy'],
    version='0.1',
    license='MIT',
    description='Python package for spectral embedding of networks',
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
)
