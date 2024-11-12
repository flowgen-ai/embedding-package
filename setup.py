# setup.py

from setuptools import setup, find_packages

setup(
    name="flowgen_embedding",
    version="0.1.0",
    packages=find_packages(include=["flowgen_emedding", "flowgen_emedding.utils"]),
    description="A vector embedding service with elasticsearch and langchain",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="aidin",
    author_email="aidin@cube10.io",
    include_package_data=True,
    install_requires=[
        "langchain-core",
        "langchain-community",
        "langchain_elasticsearch",
        "langchain",
        "langchain_openai",
        "openai",
        "elasticsearch",
        "python-dotenv",
        "psycopg2-binary",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
