from setuptools import find_packages,setup

setup(
    name='torchservedetr',
    version='0.1',
    package_dir={"":"app"},
    description='Torch Custom Package for DETR',
    author='Phaneendra Gandi',
    author_email='phaneendra.gandi.1999@gmail.com',
    packages=find_packages(where="app"),
    install_requires=['torch','numpy'],
)
