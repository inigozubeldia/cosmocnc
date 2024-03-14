## cosmocnc

cosmocnc is a Python package for evaluating the number count likelihood of galaxy cluster catalogues in a fast, flexible and accurate way. It is based on the use of Fast Fourier Transform (FFT) convolutions in order to evaluate some of the likelihood integrals. The code was introduced in [Zubeldia & Bolliet (2024)](link), where formalism underlying its likelihoods and its implementation are described in detail. Its main features are the following:

- It supports three types of likelihoods: an unbinned likelihood, a binned likelihood, and an extreme value likelihood.
- It also supports the addition of stacked cluster data (e.g., stacked lensing profiles), which is modelled in a consistent way with the cluster catalogue.
- It links the cluster mass observables (also known as mass proxies, e.g., tSZ signal-to-noise, richness, lensing mass estimate, or X-ray flux) to the cluster mass and redshift through a hierarchical model with an arbitrary number of layers, allowing for correlated scatter between the different mass observables. In each layer, the mass--observable scaling relations and the scatter covariance matrix can be easily defined in a custom way, and can depend on sky location and redshift.
- It incorporates several widely-used halo mass functions.
- The unbinned likelihood is the most general and flexible of the three. It supports an arbitrary number of cluster mass observables for each cluster in the sample and it allows for the set of mass observables to vary from cluster to cluster. It also allows for redshift measurement uncertainties. If some mass observables have completely uncorrelated scatter, \cnc{} takes advantage of this fact, boosting its computational performance significantly.
- The binned and extreme value likelihoods, on the other hand, consider the cluster abundance only across one mass observable and/or redshift, and do not allow for redshift measurement uncertainties.
- cosmocnc can produce mass estimates for each cluster in the sample, which are derived assuming the hierarchical model that is used to model the mass observables.
- It also allows for the generation of synthetic cluster catalogues for a given observational set-up.
- Several of cosmocnc's computations can be accelerated with Python's [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) module.
- The code is interfaced with the Markov chain Monte Carlo (MCMC) code [Cobaya](https://cobaya.readthedocs.io/en/latest/), allowing for easy-to-run MCMC parameter estimation.
- The code is also interfaced with [class_sz](https://github.com/CLASS-SZ/class_sz), allowing a wide range of cosmological models as well as enabling joint analyses with Cosmic Microwave Background (CMB) and Large Scale Structure (LSS) survey data.

### Installation

Download the source code and do 
```
$ pip install -e .
```
You'll then be able to import SZiFi in Python with
```
import cosmocnc
```

##Dependencies

Dependencies: [astropy](https://www.astropy.org) (optional), [class_sz](https://github.com/CLASS-SZ/class_sz) (optional), [hmf](https://hmf.readthedocs.io) (optional), [mcfit](https://github.com/eelregit/mcfit) (optional)

