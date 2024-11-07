# setup.py

from setuptools import setup, find_packages

setup(
    name="flowgen_embedding",
    version="0.1.0",
    description="A vector embedding service with elasticsearch and langchain",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="aidin",
    author_email="aidin@cube10.io",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "langchain-core",
        "langchain-community",
        "langchain",
        "langchain-openai"
        "pydantic",
        "openai",
        "sentence-transformers",
        "elasticsearch",
        "python-dotenv",
        "psycopg2-binary",
        "tiktoken"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
