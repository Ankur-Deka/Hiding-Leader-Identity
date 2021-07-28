import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="grid_search", # Replace with your own username
    version="0.0.1",
    author="Ankur",
    author_email="adeka@andrew.cmu.edu",
    description="Script Invariant Grid Search - SIGS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ankur-Deka/SIGS-Grid-Search",
    packages=['grid_search'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)