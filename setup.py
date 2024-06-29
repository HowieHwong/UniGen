from setuptools import setup, find_packages

setup(
    name='Unigen',
    version='0.1.0',
    author='Siyuan Wu & Yue Huang ',
    author_email='nauyisu022@gmail.com',
    description='A unified framework for textual dataset generation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://unigen-framework.github.io/',
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
        'fschat[model_worker]',
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
        'anthropic',
        'google.generativeai',
        'google-api-python-client',
        'google.ai.generativelanguage',
        'zhipuai',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)