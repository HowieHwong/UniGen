from setuptools import setup


setup(
    name='Unigen',
    version='0.1',
    packages=['unigen'],
    entry_points={
        'console_scripts': [
            'unigen-cli = unigen.cli:main'
        ]
    }
)i