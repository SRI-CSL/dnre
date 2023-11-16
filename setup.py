from setuptools import setup, find_packages

description = """Code for implmementing Direct Neural Ratio Estimation.
"""

with open('requirements.txt') as f:
    requirements = f.read()

setup(
    name='dnre',
    version='0.0.1',
    author='Anon',
    description=description,
    install_requires=requirements,
    package_dir={"": "src"},
    platforms='any'
)
