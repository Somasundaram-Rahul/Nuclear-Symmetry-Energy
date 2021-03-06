# Nuclear-Symmetry-Energy

This repository contains the code used to perform the analysis presented in the paper: Somasundaram, Drischler, Tews, and Margueron, _Constraints on the nuclear symmetry energy from asymmetric-matter calculations with chiral NN and 3N interactions_, [arXiv:2009.04737](https://arxiv.org/abs/2009.04737).

It also contains the analysed data originally presented in the publication: Drischler, Hebeler and Schwenk, _Asymmetric nuclear matter based on chiral two- and three-nucleon interactions_, [Phys. Rev. C 93, 054314 (2016)](https://journals.aps.org/prc/abstract/10.1103/PhysRevC.93.054314).


List of programs:
================

- `EM.py` performs the analysis of the single particle energies and produces the plots stored in the folder `results/effective_mass`.

- `SM_NM.py` performs the analysis of the energy per particle in Symmetric and Neutron matter and produces the plots stored in the folder `results/3_scales`.

- `symmetry_energy.py` calculates the global symmetry energy and creates the corresponding plot stored in the folder `results/esym_esym2`.

- `quadratic_symmetry_energy.py` calculates the quadratic symmetry energy and creates the corresponding plot stored in the folder `results/esym_esym2`.

- `non_quadraticities.py` calculates the non-quadratic contributions to the symmetry energy as well as the final fit residuals and creates the plots stored in the folder `results/non_quadraticities`.

- `crust_core.py` calculates the crust-core transition and produces the plot in the folder `results/crust_core`.

Note: The folder `results/goodness_of_fit` contains the plots presented in the appendix of the paper. 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4012355.svg)](https://doi.org/10.5281/zenodo.4012355)
