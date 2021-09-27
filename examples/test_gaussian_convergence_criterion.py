"""
This script tests the goodness of the Gaussian KL convergence criterion, using different
implementations.

In particular, tests the accuracy of the covariance recovered (since for well-recovered
covariance matrix, all criteria would be equivalent).
"""

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, random_correlation
import warnings
from time import time

# GPry things needed for building the model
from gpry.acquisition_functions import Log_exp
from gpry.gpr import GaussianProcessRegressor
from gpry.kernels import RBF, ConstantKernel as C
from gpry.preprocessing import Normalize_y, Normalize_bounds
from gpry.convergence import KL_from_draw, KL_from_MC_training, KL_from_draw_approx, \
    ConvergenceCheckError
from gpry.gp_acquisition import GP_Acquisition
from gpry.tools import kl_norm
from cobaya.model import get_model


dim = 2
z = 0.1

# Number of times that each combination of d and zeta is run with different
# gaussians
n_repeats = 200

# Criteria to be tested: tuple (criterion class, dict of parameters)
criteria = [
    (KL_from_MC_training, {"n_draws": 500, "temperature": 6}),
    (KL_from_draw_approx, {"n_draws": 50000})]

# NB: KL_from_draw does not have self.mean, self.cov, so cannot be used here


######################

data = pd.DataFrame(columns=(["i_run", "n_train"] +
                             [f"crit_{i}" for i in range(len(criteria))] +
                             [f"crit_KL_{i}" for i in range(len(criteria))] +
                             [f"time_{i}" for i in range(len(criteria))] +
                             [f"evals_{i}" for i in range(len(criteria))]))

for n_r in range(n_repeats):
    # Create random gaussian
    std = np.random.uniform(size=dim)
    eigs = np.random.uniform(size=dim)
    eigs = eigs / np.sum(eigs) * dim
    corr = random_correlation.rvs(eigs)
    cov = np.multiply(np.outer(std, std), corr)
    mean = np.zeros_like(std)
    rv = multivariate_normal(mean, cov)
    # Interface with Gpry
    input_params = [f"x_{d}" for d in range(dim)]

    def f(**kwargs):
        X = [kwargs[p] for p in input_params]
        return np.log(rv.pdf(X))

    # Define the likelihood and the prior of the model
    info = {"likelihood": {"f": {"external": f,
                                 "input_params": input_params}}}
    param_dict = {}
    for d, p_d in enumerate(input_params):
        param_dict[p_d] = {
            "prior": {"min": mean[d] - 5 * std[d], "max": mean[d] + 5 * std[d]}}
    info["params"] = param_dict
    # Initialise stuff
    model = get_model(info)
    criteria_inst = [crit(model.prior, params) for crit, params in criteria]
    #############################################################
    # Training part
    bnds = model.prior.bounds(confidence_for_unbounded=0.99995)
    kernel = C(1.0, [0.001, 10000]) \
        * RBF([0.01] * dim, "dynamic", prior_bounds=bnds)
    gp = GaussianProcessRegressor(kernel=kernel,
                                  preprocessing_X=Normalize_bounds(bnds),
                                  preprocessing_y=Normalize_y(),
                                  n_restarts_optimizer=5,
                                  noise_level=1e-3)
    af = Log_exp(zeta=z)
    acquire = GP_Acquisition(bnds,
                             acq_func=af,
                             preprocessing_X=Normalize_bounds(bnds),
                             n_restarts_optimizer=10,
                             )
    init_X = model.prior.sample(dim)
    init_y = np.empty(init_X.shape[0])
    for i, i_X in enumerate(init_X):
        init_y[i] = model.logpost(i_X)
    try:
        gp.append_to_data(init_X, init_y, fit=True)
    except ValueError:
        continue
    # Number of acquired points per step
    n_points = dim
    y_s = init_y
    n_iterations = 10
    for i in range(n_iterations):
        print(f"+++ Iteration {i} (of {n_iterations}) +++++++++")
        new_X, y_lies, acq_vals = acquire.multi_optimization(gp, n_points=n_points)
        if len(new_X) != 0:
            new_y = np.empty(new_X.shape[0])
            for m, X_m in enumerate(new_X):
                new_y[m] = model.logpost(X_m)
            y_s = np.append(y_s, new_y)
            gp.append_to_data(new_X, new_y, fit=True)
        else:
            warnings.warn("No points were added to the GP because the proposed"
                          " points have already been evaluated.")
        # Compute convergence criteria
        row = {"i_run": n_r, "n_train": len(gp.X_train)}
        for i_c, crit in enumerate(criteria_inst):
            try:
                # arreglar esto: mucho tiempo computar 2 veces cada vez
                n_eval_old = gp.n_eval
                start = time()
                val = crit.criterion_value(gp)
                time_eval = time() - start
                # n evaluations is important, but speed is not correlated with it:
                # it's infinitely fastes when predicting for >1 X's at once!
                n_eval = gp.n_eval - n_eval_old
                # Until ConvergenceCheckerror used by the other convergence criteria:
                if np.isnan(val):
                    raise ConvergenceCheckError()
                kl = kl_norm(crit.mean, crit.cov, mean, cov)
            except ConvergenceCheckError:
                val, kl, time_eval, n_eval = np.nan, np.nan, np.nan, np.nan
            row.update({f"crit_{i_c}": val, f"crit_KL_{i_c}": kl,
                        f"time_{i_c}": time_eval, f"evals_{i_c}": n_eval})
        data.loc[len(data.index)] = row
        print(data)
# with open(file_name, 'wb') as f:
#     pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
