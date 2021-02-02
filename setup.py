import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dive",
    version="0.0.1",
    author="Stephan Pribitzer",
    author_email="stephapr@uw.edu",
    description="A toolbox for Bayesian analysis of DEER data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/StollLab/dive",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    python_requires='>=3.8',
)