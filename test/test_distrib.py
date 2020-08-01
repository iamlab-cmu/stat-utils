import random

import numpy as np

import stat_utils

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_distrib(random_seed):

    # Scalar ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test that scalar sampling works through a variety of methods
    scalar_config = {"type": "scalar", "value": 5.0}
    scalar_distrib = stat_utils.Distrib(scalar_config)
    assert stat_utils.sample_from_distrib(
        scalar_config
    ) == stat_utils.sample_from_distrib(5.0)
    assert scalar_distrib.sample() == stat_utils.sample_from_distrib(5.0)

    assert stat_utils.sample_from_distrib(5.0) != stat_utils.sample_from_distrib(89.124)

    print(scalar_distrib)
    print(repr(scalar_distrib))

    # Uniform sampling ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    num_sample_uniform = 10000
    uniform_min = 0.0
    uniform_max = 10.0
    uniform_distrib = stat_utils.Distrib(
        {"type": "uniform", "range": [uniform_min, uniform_max]}
    )

    print(uniform_distrib)
    print(repr(uniform_distrib))

    uniform_samples = []
    for _ in range(num_sample_uniform):
        uniform_samples.append(uniform_distrib.sample())
    uniform_samples = np.array(uniform_samples)

    # Test that all values are within uniform range
    assert np.logical_and(
        uniform_min <= uniform_samples, uniform_samples <= uniform_max
    ).all(), (
        "Expected all samples from uniform_distrib to be within "
        + "the specified range [{}, {}], but at least one sample is not.".format(
            uniform_min, uniform_max
        )
    )

    # Test that uniform sampling mean is close to expectation
    uniform_mean = np.mean(uniform_samples)
    uniform_mean_GT = 5.0
    correct_thresh = 0.1
    assert np.allclose(uniform_mean, uniform_mean_GT, rtol=0.0, atol=correct_thresh), (
        "Expected uniform_mean ({}) to be within {} of {}, " + "but it is not."
    ).format(uniform_mean, correct_thresh, uniform_mean_GT)

    # Test that sampling "None" returns None
    assert (
        stat_utils.sample_from_distrib(None) is None
    ), "Expected that 'sample_from_distrib' for None is None, but it is not."

    # Test that strings evaluate correctly
    e_scalar = stat_utils.Distrib({"type": "scalar", "value": "2*np.e"})
    print(e_scalar)
    print(repr(e_scalar))
    sample_from_e_scalar = e_scalar.sample()
    print(sample_from_e_scalar)
    assert np.isclose(
        sample_from_e_scalar, 2.0 * np.e
    ), "Expected scalar string handling to work, but it does not."

    pi_uniform_distrib = stat_utils.Distrib(
        {"type": "uniform", "range": ["-np.pi", "np.pi"]}
    )
    print(pi_uniform_distrib)
    print(repr(pi_uniform_distrib))
    sample_from_pi_uniform_distrib = pi_uniform_distrib.sample()
    print(sample_from_pi_uniform_distrib)
    assert (
        -np.pi <= sample_from_pi_uniform_distrib
        and sample_from_pi_uniform_distrib <= np.pi
    ), "Expected uniform string handling to work, but it does not."
