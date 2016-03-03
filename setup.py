# coding: utf-8

from setuptools import setup, find_packages
from pip.req import parse_requirements


VERSION = '20160303'

reqs = [str(req.req) for req in parse_requirements('reqs.txt', session=False)]


setup(
    name='connections',
    version=VERSION,
    url='https://github.com/Linusp/connections',
    author='Linusp',
    author_email='linusp1024@gmail.com',
    description='Implementations of ANN',
    license='MIT',
    packages=find_packages(),
    install_requires=reqs,
    include_package_data=True,
    zip_safe=False,
)
