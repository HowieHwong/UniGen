from setuptools import setup, find_packages

setup(
    name='Unigen',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A unified framework for textual dataset generation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/unigen',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'unigen-cli = unigen.cli:main'
        ]
    },
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'openai>1.0.0',
        'protobuf',
        'python-dotenv',
        'PyYAML',
        'replicate',
        'requests',
        'scikit-learn',
        'tenacity',
        'tiktoken',
        'torch',
        'tqdm',
        'transformers',
        'urllib3',
        'wikipedia-api',
        'zhipuai',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)