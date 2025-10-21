from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='preoperativeSAM',
    version='0.1',
    description='preoperative guideline for self-promoting AI in medical imaging',
    author='Angelo Lasala',
    author_email='Lasala.Angelo@santannapisa.it',
    packages=find_packages(),
    install_requires=requirements,
)