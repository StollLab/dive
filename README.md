# dive

### About
`dive` is a Python package for Bayesian analysis of dipolar EPR (electron paramagnetic resonance) spectroscopy data through Markov chain Monte Carlo (MCMC) sampling with the Python package [PyMC](https://www.pymc.io).

### Requirements

`dive` is available for Windows, Mac and Linux systems and requires **Python 3.9** or later and **PyMC 5.0** or later.
 
### Features

`dive`'s features include:
- An output InferenceData object containing many random posterior samples for each parameter
- Full uncertainty quantification for all model parameters, including the distance distribution
- Visualizations for ensembles of fitted signals and residuals
- Visualizations for ensembles of fitted distance distributions
- Histograms for margnialized posteriors of other parameters such as modulation depth and background decay rate

### Setup

You can install `dive` using `pip`. Please note that the installation name is `dive-MCMC`.

    pip install dive-MCMC

`dive` can then be used by importing the package as usual.

    import dive

<!-- As long as `dive` is in a development state, use the following installation procedure:

If using `conda`, install the following packages to the environment of your choice:

    conda install pymc deerlab scipy matplotlib pandas mkl-service h5netcdf pytest

You can also use `pip`.

After successful installation of dependencies, navigate to the directory that contains the `dive` source code and run

    python setup.py develop -->

<!-- A pre-built distribution can be installed using `pip`.

First, ensure that `pip` is up-to-date. From a terminal (preferably with admin privileges) use the following command:

    python -m pip install --upgrade pip

Next, install dive with

    python -m pip install dive -->

### Documentation

See the [documentation](https://stolllab.github.io/dive) for a detailed guide on how to use `dive`. An IPython Notebook guide on using `dive` can also be found under the `examples/` directory.

### Citation

When you use `dive` in your work, please cite the following publication:

 **Bayesian Probabilistic Analysis of DEER Spectroscopy Data Using Parametric Distance Distribution Models** <br>
Sarah R. Sweger, Stephan Pribitzer, and Stefan Stoll <br>
 *J. Phys. Chem. A* 2020, 124, 30, 6193–6202 <br>
 <a href="https://doi.org/10.1021/acs.jpca.0c05026"> doi.org/10.1021/acs.jpca.0c05026</a>


### License

`dive` is licensed under the [MIT License](LICENSE).

Copyright © 2024:  Sarah Sweger, Julian Cheung, Lukas Zha, Stephan Pribitzer, Stefan Stoll
