import pandas as pd
from typing import Tuple, List, Callable
import torch
import tensorflow as tf
from Ploting.fast_plot_Func import *
import tensorflow_probability as tfp
import edward2 as ed

tfd = eval("tfp.distributions")
tfpl = eval("tfp.layers")
tfb = eval("tfp.bijectors")

gm = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(
        probs=[0.3, 0.7]),
    components_distribution=tfd.Normal(
        loc=[-1., 1],  # One for each component.
        scale=[0.1, 0.5]))  # And same here.

gm2 = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(
        probs=[0.3, 0.7]),
    components_distribution=tfd.MultivariateNormalDiag(
        loc=[[-1., 1],  # component 1
             [1, -1]],  # component 2
        scale_identity_multiplier=[.3, .6]))
samples = gm2.sample(1)

# Initialize a single 2-variate Gaussian.
mvn = tfd.MultivariateNormalDiag(
    loc=[1., -1],
    scale_diag=[1, 2.])
