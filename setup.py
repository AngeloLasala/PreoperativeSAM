from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='preoperative SAM',
    version='0.0',
    description='preoperative guideline for self-promoting AI in medical imaging',
    author='Angelo Lasala',
    author_email='Lasala.Angelo@santannapisa.it',
    packages=find_packages(),
    install_requires=requirements,
)