"""
Example code for a simple GP Characterization of a likelihood.
"""

import os
from gpry.mpi import is_main_process
from gpry.io import create_path
from cobaya import model

# Path for saving plots, make sure it exists!
checkpoint = "output/simple"

# Building the likelihood
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt

# mean = [3, 2]
# cov = [[0.5, 0.4], [0.4, 1.5]]
d = 10
mean = np.ones(d)
cov = np.eye(d)
rv = multivariate_normal(mean, cov)


def lkl(x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9):
    print("not vectorized, this is a dummy that cobaya needs")
    return NotImplemented


def lkl_vec(X):
    print("vectorized")
    print(X.shape)
    return rv.logpdf(X)


#############################################################
# Plotting the likelihood

# if is_main_process:
#     import matplotlib.pyplot as plt
#     from matplotlib.colors import LogNorm

#     a = np.linspace(-10., 10., 200)
#     b = np.linspace(-10., 10., 200)
#     A, B = np.meshgrid(a, b)

#     x = np.stack((A, B), axis=-1)
#     xdim = x.shape
#     x = x.reshape(-1, 2)
#     Y = -1 * lkl(x[:, 0], x[:, 1])
#     Y = Y.reshape(xdim[:-1])

#     # Plot ground truth
#     fig = plt.figure()
#     im = plt.pcolor(A, B, Y, norm=LogNorm())
#     plt.xlabel(r"$x$")
#     plt.ylabel(r"$y$")
#     fig.subplots_adjust(right=0.8)
#     cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
#     cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
#     if is_main_process and checkpoint:
#         create_path(os.path.join(checkpoint, "images/"), verbose=False)
#         plt.savefig(os.path.join(checkpoint, "images/Ground_truth.png"), dpi=300)
#     plt.close()

#############################################################

# Define the model (containing the prior and the likelihood)
from cobaya.model import get_model

info = {"likelihood": {"normal": lkl}}
params = ["x_{}".format(str(i)) for i in range(d)]
# info["params"] = {
#     "x": {"prior": {"min": -10, "max": 10}},
#     "y": {"prior": {"min": -10, "max": 10}}
#     }
info["params"] = {
    param: {"prior": {"min": -10, "max": 10}} for param in params
}

model = get_model(info)
model.__dict__["vectorized"] = lkl_vec

initial_samples=50
points_per_batch=10

options = {
    "max_initial": initial_samples,
    "max_points": 200,
    # "n_initial": n_procs,
    "n_points_per_acq": points_per_batch,
}

from gpry.run import Runner

runner = Runner(
    model, checkpoint=checkpoint, load_checkpoint="resume", options=options
)
# runner = Runner(model, checkpoint=checkpoint, load_checkpoint="overwrite")

# Run the GP
runner.run(vect=True)

# Run the MCMC and extract samples
updated_info, sampler = runner.generate_mc_sample()


# Plotting
# runner.plot_mc(updated_info, sampler)
# runner.plot_distance_distribution(updated_info, sampler)

# Validation
if is_main_process:
    from getdist.gaussian_mixtures import GaussianND
    from getdist.mcsamples import MCSamplesFromCobaya
    import getdist.plots as gdplt
    from gpry.plots import getdist_add_training

    gpr = runner.gpr
    gdsamples_gp = MCSamplesFromCobaya(
        updated_info, sampler.products()["sample"]
    )
    gdsamples_truth = GaussianND(mean, cov, names=list(info["params"]))
    gdplot = gdplt.get_subplot_plotter(width_inch=5)
    gdplot.triangle_plot(
        [gdsamples_truth, gdsamples_gp],
        list(info["params"]),
        filled=[False, True],
        legend_labels=["Truth", "MC from GP"],
    )
    getdist_add_training(gdplot, model, gpr)
    plt.savefig(
        os.path.join(checkpoint, "images/Comparison_triangle.png"), dpi=300
    )
