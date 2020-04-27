import random
import numpy as np

import stat_utils

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_bayes_linreg(random_seed):

    correct_thresh = 1.e-2
    posterior_var_thresh = 1.e-5

    # Load in ground truth test data.
    # Example adapted from:
    # https://github.com/krasserm/bayesian-machine-learning.git

    GT_params = [0.5, 0.4, -0.3] # intercept is last
    num_features = len(GT_params) - 1

    # Training observations in [-1, 1)
    num_samples = 10000
    train_data = np.random.rand(num_samples, num_features) * 2. - 1.

    # Generate true labels and corrupt with noise to get noisy train labels
    true_response = np.matmul(train_data,
                              np.resize(GT_params[:-1], (num_features,1))) + \
                    GT_params[-1]
    true_response = np.resize(true_response, (num_samples,))

    beta = 100.
    noise_variance = 1./beta

    noisy_response = true_response + \
                     np.random.normal(scale=np.sqrt(noise_variance),
                                      size=num_samples)
    train_labels = noisy_response

    # Initial covariance used for prior
    alpha = 1.
    init_variance = 1./alpha

    # Do Bayesian Linear Regression
    posterior_mean, posterior_cov = \
        stat_utils.bayes_linreg_uninformed_prior(train_data, train_labels,
                                                 noise_variance, init_variance)

    # Test function to ensure correct output
    assert np.allclose(posterior_mean, GT_params,
                       rtol=0., atol=correct_thresh), \
        ("Expected each element of posterior_mean ({}) to be within {} of " + \
         "GT_params ({}), but it is not.").format(posterior_mean,
                                                  correct_thresh, GT_params)
    assert np.all(posterior_cov < posterior_var_thresh), \
        ("Expected all elements of posterior_cov to be less than {}, " + \
         "but at least one element is not.").format(posterior_var_thresh)
