import pandas as pd
from typing import Tuple, List, Callable
import torch
import tensorflow as tf
from Ploting.fast_plot_Func import *
import tensorflow_probability as tfp
import edward2 as ed
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

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

gm3 = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(
        probs=[0.3, 0.7]),
    components_distribution=tfd.LogitNormal(
        loc=[-1., 1],  # One for each component.
        scale=[0.1, 0.5]))

joint = tfd.JointDistributionSequential([
    tfd.Independent(tfd.Exponential(rate=[100, 120]), 1,
                    name="e"
                    ),  # e
    lambda e: tfd.Gamma(concentration=e[..., 0], rate=e[..., 1],
                        name="g"
                        ),  # g
    tfd.Normal(loc=0, scale=2.,
               # name="my_n"
               ),  # n
    lambda n, g: tfd.Normal(loc=n, scale=g,
                            # name="my_m"
                            ),  # m
    lambda m: tfd.Sample(tfd.Bernoulli(logits=m), 12,
                         name="my_x"
                         )  # x
])
joint.resolve_graph()

X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [12, 4], [10, 7]])
bgm = BayesianGaussianMixture(n_components=10, random_state=42).fit(X)