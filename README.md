EuroPED-NN

This repository contains the script to run the surrogate model for EuroPED plasma model: EuroPED-NN.

EuroPED-NN.py contains the functions to run EuroPED-NN. The last 4 lines of the script contain an example of how to run EuroPED-NN with some parameters.

The units that the model expects are: Ip[MA], Bt[T], Rmag[m], rminor[m], Ptot[MW], nesep[1e19m^(-3)], ne_ped[1e19m^(-3)], Te_ped[keV], Ped_width[\psi_N].

Based on a Bayesian Neural Network with Noise Contrastive Prior.

Tested on Python=3.10.11 and TensorFlow=2.15.0

Open Access paper available at https://iopscience.iop.org/article/10.1088/1361-6587/ad6707 
