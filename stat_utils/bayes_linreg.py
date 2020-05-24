import numpy as np

# ---------------------------------------------------------


def bayes_linreg(
    train_data, train_labels, prior_mean, prior_covariance, noise_variance
):
    """
    Bayes linear regression. Assumes the probability distribution is Gaussian.
    This model includes an intercept term, so num_params = num_features + 1,
    where num_params is the number of models parameters and num_features
    is the number of features of the training data.

    Args:
        train_data (nested float list or np.ndarray):
            An array of training data. The shape should be
            [num_samples x num_features]
        train_labels (list of float or np.ndarray):
            An array of labels for the training data with length num_samples.
        prior_mean (list of float or np.ndarray):
            An array representing the mean for the Gaussian prior
            with length num_params.
        prior_covariance (nested float list or np.ndarray):
            The covariance for the Gaussian prior with shape
            [num_params x num_params].
        noise_variance (float):
            A parameter specifying the noise variance in the training data.
            Equivalent to the inverse of the 'beta' term seen in ML literature.

    Returns:
        mean (np.ndarray, shape=(num_params,)):
            Linear regression coefficients ("model parameters"). The intercept
            parameter is the LAST TERM in the array. The order is consistent
            with train_data and train_labels.
        covariance (np.ndarray, shape=(num_params, num_params)):
            Covariance with respect to the model parameters. The order
            is consistent with train_data, train_labels, and mean.
    """

    # Input argument handling
    num_samples = train_data.shape[0]
    assert num_samples == len(
        train_labels
    ), "Expected train_labels to have length {}, but has length {}.".format(
        num_samples, len(train_labels)
    )

    num_features = train_data.shape[1]
    num_params = num_features + 1
    assert num_params == len(
        prior_mean
    ), "Expected prior_mean to have length {}, but it has length {}.".format(
        num_params, len(prior_mean)
    )

    prior_covariance = np.array(prior_covariance)
    assert (
        num_params,
        num_params,
    ) == prior_covariance.shape, "Expected prior_covariance to have shape {}, but it has shape {}.".format(
        (num_params, num_params), prior_covariance.shape
    )

    assert isinstance(
        noise_variance, float
    ), 'Expected "noise_variance" to be a float, but it is {}.'.format(
        type(noise_variance)
    )
    noise_precision = 1.0 / noise_variance

    # Calculation of new covariance
    design_matrix = np.hstack((train_data, np.ones((num_samples, 1))))
    prior_precision = np.linalg.inv(prior_covariance)
    precision = prior_precision + noise_precision * np.matmul(
        np.transpose(design_matrix), design_matrix
    )
    covariance = np.linalg.inv(precision)

    # Calculation of new mean
    mean = np.matmul(
        np.matmul(covariance, prior_precision), prior_mean
    ) + noise_precision * np.matmul(
        covariance, np.matmul(np.transpose(design_matrix), train_labels)
    )

    return mean, covariance


def bayes_linreg_uninformed_prior(
    train_data, train_labels, noise_variance, init_variance=1.0
):

    """
    Bayes linear regression with an uninformed prior. Assumes the probability
    distribution is Gaussian. Specifically, the uninformed prior is
    a Gaussian distribution with zero mean and isotropic covariance,
    specificied by covariance = init_variance * eye(num_params).
    This model includes an intercept term, so num_params = num_features + 1,
    where num_params is the number of models parameters and num_features
    is the number of features of the training data.

    Args:
        train_data (nested float list or np.ndarray):
            An array of training data. The shape should be
            [num_samples x num_features].
        train_labels (list of float or np.ndarray):
            An array of labels for the training data with length num_samples.
        noise_variance (float):
            A parameter specifying the noise variance in the training data.
            Equivalent to the inverse of the 'beta' term seen in ML literature.
        init_variance (float, defaults to 1.):
            A parameter specifying the scale of the isotropic covariance.
            Equivalent to the inverse of the 'alpha' term seen in ML literature.

    Returns:
        mean (np.ndarray, shape=(num_params,)):
            Linear regression coefficients ("model parameters"). The intercept
            parameter is the LAST TERM in the array. The order is consistent
            with train_data and train_labels.
        covariance (np.ndarray, shape=(num_params, num_params)):
            Covariance with respect to the model parameters. The order
            is consistent with train_data, train_labels, and mean.
    """

    # Input argument handling
    assert isinstance(
        init_variance, float
    ), 'Expected "init_variance" to be a float, but it is {}.'.format(
        type(init_variance)
    )

    num_features = train_data.shape[1]
    num_params = num_features + 1
    prior_mean = np.zeros(num_params)
    prior_covariance = init_variance * np.eye(num_params)
    mean, covariance = bayes_linreg(
        train_data, train_labels, prior_mean, prior_covariance, noise_variance
    )
    return mean, covariance
