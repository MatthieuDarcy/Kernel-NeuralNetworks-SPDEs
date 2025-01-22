![alt text](https://github.com/MatthieuDarcy/StochasticPDEs/blob/main/figures/SPDE.PNG?raw=true)
This repository contains notebooks that run the experiments necessary to reproduce the experiments presented in the paper "_Solving Roughly Forced Nonlinear PDEs via Misspecified Kernel Methods and Neural Networks_"

The repository is organized as follows:

1. Rough spatial: section 4.2.
2. Time-dependent: section 4.3.
3. utils: contains various utilities and implementations for the kernel method.
4. plotting_templates: templates for the plots in the notebooks.

See the subfolders for greater details. 

The requirements indicate the necessary libraries and the version that were used for the numerical experiments.

Note that when using Jax, we have used the GPU version. The CPU version of JAX seems to be problematic with the the convolution operation used for the stochastic heat equatio, but seems to work fine with all other examples.
