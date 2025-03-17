#EuroPED-NN, surrogate model of EuroPED pedestal model

# CopyRight Notice
# author(s) of code: Alex Panera Alvarez,       DIFFER (c) 2023 - 2024
#                    Aaron Ho,                  DIFFER (c) 2023 - 2024

# CopyRight Disclaimer
# The Dutch Institute For Fundamental Energy Research (DIFFER) hereby disclaims all copyright interest in the model “EuroPED-NN” (plasma pedestal surrogate model)
# written by Alex Panera Alvarez and Aaron Ho during their contract period at DIFFER.
# DIFFER hereby provides permission to license “EuroPED-NN” under the Lesser GPL (GNU LGPL) conditions.

# License Notice
# This file is part of EuroPED-NN.
# EuroPED-NN is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License 
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

# EuroPED-NN is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty 
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with EuroPED-NN. 
# If not, see <https://www.gnu.org/licenses>. 
#
# Reference Journal Articles
# A. Panera Alvarez et al 2024 Plasma Phys. Control. Fusion 66 095012

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Dense, Lambda, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl

#Model architecture
def mean_dist_fn(variational_layer):
    def mean_dist(inputs):
        bias_mean = variational_layer.bias_posterior.mean()

        kernel_mean = variational_layer.kernel_posterior.mean()
        kernel_std = variational_layer.kernel_posterior.stddev()
        

        mu_mean = tf.matmul(inputs, kernel_mean) + bias_mean
        mu_var = tf.matmul(inputs ** 2, kernel_std ** 2)
        mu_std = tf.sqrt(mu_var)
        return tfd.Normal(mu_mean, mu_std)
        

    return mean_dist

def create_model(n_hidden1=20,n_hidden_neped=8,n_hidden_teped=8,n_hidden_delta=10):
    leaky_relu = LeakyReLU(alpha=0.2)
    variational_layer1 = tfpl.DenseReparameterization(1, name='mu')
    variational_layer2 = tfpl.DenseReparameterization(1, name='mu2')
    variational_layer3 = tfpl.DenseReparameterization(1, name='mu3')
    
    input_x1 = Input(shape=(1,))  
    input_x2 = Input(shape=(1,))  
    input_x3 = Input(shape=(1,))
    input_x4 = Input(shape=(1,))
    input_x5 = Input(shape=(1,))
    input_x6 = Input(shape=(1,))
    input_x7 = Input(shape=(1,))
    input_x8 = Input(shape=(1,))
    input_combined = Concatenate(axis=1)([input_x1,input_x2,input_x3,input_x4,input_x5,input_x6,input_x7,input_x8])

    d_combined = Dense(n_hidden1, input_dim=8, activation=leaky_relu)(input_combined)
    d_combined1 = Dense(n_hidden_delta, activation=leaky_relu)(d_combined)
    d_combined2 = Dense(n_hidden_teped, activation=leaky_relu)(d_combined)
    d_combined3 = Dense(n_hidden_neped, activation=leaky_relu)(d_combined)
    s1 = Dense(1, activation='softplus', name='sigma1')(d_combined1)
    s2 = Dense(1, activation='softplus', name='sigma2')(d_combined2)
    s3 = Dense(1, activation='softplus', name='sigma3')(d_combined3)
    m1 = variational_layer1(d_combined1)
    m2 = variational_layer2(d_combined2)
    m3 = variational_layer3(d_combined3)

    mean_dist1 = tfpl.DistributionLambda(mean_dist_fn(variational_layer1), name='output1')(d_combined1)
    mean_dist2 = tfpl.DistributionLambda(mean_dist_fn(variational_layer2), name='output2')(d_combined2)
    mean_dist3 = tfpl.DistributionLambda(mean_dist_fn(variational_layer3), name='output3')(d_combined3)
    ndim_out1 = tfpl.DistributionLambda(lambda p: tfd.Normal(p[0],p[1]))((m1,s1))
    ndim_out2=tfpl.DistributionLambda(lambda p: tfd.Normal(p[0],p[1]))((m2,s2))
    ndim_out3=tfpl.DistributionLambda(lambda p: tfd.Normal(p[0],p[1]))((m3,s3))

    inputs = [input_x1,input_x2,input_x3,input_x4,input_x5,input_x6,input_x7,input_x8]
    #1--> Delta, 2--> Te_ped, 3--> ne_ped
    outputs = [ndim_out1, mean_dist1, ndim_out2, mean_dist2, ndim_out3, mean_dist3]

    return Model(inputs, outputs)


def calculate_epsilon(rminor, rmag):
    return rminor / rmag

def calculate_mu(ip, bt, rminor):
    mu0 = 4 * np.pi * 1e-7
    return (mu0 / (2 * np.pi)) * 1e6 * ip / (bt * rminor)

def normalize(val, mu, std):
    return (val - mu) / std

def unnormalize(val, mu, std):
    return (val * std) + mu


def evaluate_model_dimensionless(model, mu, bt, eps, kappa, delta, ptot, nesep, zeff):

    # Input normalization coefficients used in the training
    mu_coeff = [0.1990419620007955, 0.0202534371777224]
    bt_coeff = [2.2591623632075466, 0.4583746977687184]
    delta_coeff = [0.1970339047169811, 0.0853293272220600]
    eps_coeff = [0.3177485948868631, 0.0065777852588341]
    kappa_coeff = [1.6802375981132076, 0.0291965293006510]
    ptot_coeff = [15.0727117594339646, 5.1823410870804487]
    nesep_coeff = [2.4656216226415091, 1.0250577871739395]
    zeff_coeff = [1.2936152698113208, 0.2548076583053399]

    mu_norm = normalize(mu, mu_coeff[0], mu_coeff[1])
    bt_norm = normalize(bt, bt_coeff[0], bt_coeff[1])
    delta_norm = normalize(delta, delta_coeff[0], delta_coeff[1])
    eps_norm = normalize(eps, eps_coeff[0], eps_coeff[1])
    kappa_norm = normalize(kappa, kappa_coeff[0], kappa_coeff[1])
    ptot_norm = normalize(ptot, ptot_coeff[0], ptot_coeff[1])
    nesep_norm = normalize(nesep, nesep_coeff[0], nesep_coeff[1])
    zeff_norm = normalize(zeff, zeff_coeff[0], zeff_coeff[1])

    ndim_width, mean_width, ndim_teped, mean_teped, ndim_neped, mean_neped = model(
        [tf.constant(mu_norm,dtype=tf.float32),
         tf.constant(bt_norm,dtype=tf.float32),
         tf.constant(eps_norm,dtype=tf.float32),
         tf.constant(kappa_norm,dtype=tf.float32),
         tf.constant(delta_norm,dtype=tf.float32),
         tf.constant(ptot_norm,dtype=tf.float32),
         tf.constant(nesep_norm,dtype=tf.float32),
         tf.constant(zeff_norm,dtype=tf.float32)]
    )
    width_norm = mean_width.mean().numpy()
    teped_norm = mean_teped.mean().numpy()
    neped_norm = mean_neped.mean().numpy()
    width_epi_norm = mean_width.stddev().numpy()
    teped_epi_norm = mean_teped.stddev().numpy()
    neped_epi_norm = mean_neped.stddev().numpy()
    width_alea_norm = ndim_width.stddev().numpy()
    teped_alea_norm = ndim_teped.stddev().numpy()
    neped_alea_norm = ndim_neped.stddev().numpy()

    # Output normalization coefficients used in the training
    width_coeff = [0.0325775584905660, 0.0025140327886658]
    teped_coeff = [0.6280061933962264, 0.2002834884270243]
    neped_coeff = [5.4591276339622636, 1.4684912169918376]
    width_error_coeff = [0.0, 0.0025140327886658]
    teped_error_coeff = [0.0, 0.2002834884270243]
    neped_error_coeff = [0.0, 1.4684912169918376]

    width = unnormalize(width_norm, width_coeff[0], width_coeff[1])
    teped = unnormalize(teped_norm, teped_coeff[0], teped_coeff[1])
    neped = unnormalize(neped_norm, neped_coeff[0], neped_coeff[1])
    width_epi = unnormalize(width_epi_norm, width_error_coeff[0], width_error_coeff[1])
    teped_epi = unnormalize(teped_epi_norm, teped_error_coeff[0], teped_error_coeff[1])
    neped_epi = unnormalize(neped_epi_norm, neped_error_coeff[0], neped_error_coeff[1])
    width_alea = unnormalize(width_alea_norm, width_error_coeff[0], width_error_coeff[1])
    teped_alea = unnormalize(teped_alea_norm, teped_error_coeff[0], teped_error_coeff[1])
    neped_alea = unnormalize(neped_alea_norm, neped_error_coeff[0], neped_error_coeff[1])

    return width, width_epi, width_alea, teped, teped_epi, teped_alea, neped, neped_epi, neped_alea


def evaluate_model(model, ip, bt, delta, rmag, rminor, kappa, ptot, nesep, zeff):
    mu = calculate_mu(ip, bt, rminor)
    eps = calculate_epsilon(rminor, rmag)
    return evaluate_model_dimensionless(model, mu, bt, eps, kappa, delta, ptot, nesep, zeff)

model = create_model()
model.load_weights(r'EuroPED_NN_OOD_14_dec.h5')
#Evaluation example with ip, bt, delta, rmag, rminor, kappa, ptot, nesep, zeff = [1.98,2.1,0.18,2.8,0.92,1.6,11.6,3.4,1.13]
width, width_epi, width_alea, teped, teped_epi, teped_alea, neped, neped_epi, neped_alea = evaluate_model(model,np.array([1.98]),
                            np.array([2.1]),np.array([0.18]),np.array([2.8]),np.array([0.92]),np.array([1.6]),np.array([11.6]),
                            np.array([3.4]),np.array([1.13]))
