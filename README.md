# dive

<!-- ![PyPI](https://img.shields.io/pypi/v/dive) -->
<!-- ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dive) -->
<!-- ![Website]() -->
<!-- ![PyPI - Downloads](https://img.shields.io/pypi/dm/dive?color=brightgreen) -->

### About
`dive` is a Python package for Bayesian analysis of dipolar EPR (electron paramagnetic resonance) spectroscopy data.

### Requirements

`dive` is available for Windows, Mac and Linux systems and requires **Python 3.8** and **pymc 3.11.0**.

<!-- All additional dependencies (pymc3, theano, etc.) are automatically downloaded and installed during the setup. -->
 
### Setup

As long as `dive` is in a development state, the following procedure can be used to for installtion:

If using `conda`, install the following packages to the environment of your choice:

    conda install mkl-service libpython m2w64-toolchain scipy matplotlib pandas

In order to get the newest version of `pymc3`, it is necessary to run

    pip install pymc3 

Unfortunately, the `conda-forge` version is usually deprecated.

After successful installation of `pymc3`, navigate to the directory that contains the `dive` source code and run

    python setup.py develop

For drawing graphs of the models, you will need `python-graphviz`, which can be installed with

    conda install -c conda-forge python-graphviz

<!-- A pre-built distribution can be installed using `pip`.

First, ensure that `pip` is up-to-date. From a terminal (preferably with admin privileges) use the following command:

    python -m pip install --upgrade pip

Next, install dive with

    python -m pip install dive -->

### Citation

When you use `dive` in your work, please cite the following publication:

 **Bayesian Probabilistic Analysis of DEER Spectroscopy Data Using Parametric Distance Distribution Models** <br>
Sarah R. Sweger, Stephan Pribitzer, and Stefan Stoll <br>
 *J. Phys. Chem. A* 2020, 124, 30, 6193–6202 <br>
 <a href="https://doi.org/10.1021/acs.jpca.0c05026"> doi.org/10.1021/acs.jpca.0c05026</a>


### License

`dive` is licensed under the [MIT License](LICENSE).

Copyright © 2021: Stephan Pribitzer, Sarah Sweger, Stefan Stoll