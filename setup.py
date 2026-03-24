from setuptools import setup, find_packages

setup(
    name='HASTE',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # Add any dependencies here, e.g.:
        # 'numpy',
        # 'scipy',
    ],
    author='Joseph Patrick Leonor',
    description='A package for HASTE project with DMD and compute functions',
    include_package_data=True,
)

