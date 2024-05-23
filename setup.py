from setuptools import setup, find_packages
# install anise module for import in notebooks and other code
setup(
    name='anise',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)
