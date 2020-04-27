import random

import numpy as np

import stat_utils

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_distrib(random_seed):

    # Scalar ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test that scalar sampling works through a variety of methods
    scalar_config = {'type': 'scalar', 'value': 5.}
    scalar_distrib = stat_utils.Distrib(scalar_config)
    assert stat_utils.sample_from_distrib(scalar_config) == \
           stat_utils.sample_from_distrib(5.)
    assert scalar_distrib.sample() == stat_utils.sample_from_distrib(5.)

    assert stat_utils.sample_from_distrib(5.) != \
           stat_utils.sample_from_distrib(89.124)

    print(scalar_distrib)
    print(repr(scalar_distrib))

    # Uniform sampling ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    num_sample_uniform = 10000
    uniform_min = 0.
    uniform_max = 10.
    uniform_distrib = stat_utils.Distrib({'type': 'uniform',
                                          'range': [uniform_min, uniform_max]})

    print(uniform_distrib)
    print(repr(uniform_distrib))

    uniform_samples = []
    for _ in range(num_sample_uniform):
        uniform_samples.append(uniform_distrib.sample())
    uniform_samples = np.array(uniform_samples)

    # Test that all values are within uniform range
    assert np.logical_and(uniform_min <= uniform_samples,
                          uniform_samples <= uniform_max).all(), \
        "Expected all samples from uniform_distrib to be within " + \
        "the specified range [{}, {}], but at least one sample is not.".format(
            uniform_min, uniform_max)

    # Test that uniform sampling mean is close to expectation
    uniform_mean = np.mean(uniform_samples)
    uniform_mean_GT = 5.
    correct_thresh = 0.1
    assert np.allclose(uniform_mean, uniform_mean_GT,
                       rtol=0., atol=correct_thresh), \
        ("Expected uniform_mean ({}) to be within {} of {}, " + \
         "but it is not.").format(uniform_mean, correct_thresh,
                                  uniform_mean_GT)
