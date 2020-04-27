import numpy as np

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

KNOWN_DISTRIB_TYPES = ['scalar',
                       'uniform']

class Distrib:

    def __init__(self, distrib_config):

        # Input argument handling
        assert isinstance(distrib_config, dict), \
            "Expected \"distrib_config\" to be type dict, " + \
            "but it is type \"{}\".".format(type(distrib_config))

        assert 'type' in distrib_config, \
            "Expected key \"type\" to exist in distribution config, " + \
            "but it does not."

        distrib_type = distrib_config['type']

        assert distrib_type in KNOWN_DISTRIB_TYPES, \
            "Distribution type \"{}\" not recognized.".format(distrib_type)

        self._distrib_config = distrib_config

        # TBD - error checking of expected config keys based on distribution?

    def __repr__(self):
        return str(self._distrib_config)

    def __str__(self):
        if self.type() == 'scalar':
            distrib_str = 'scalar: {}'.format(self._distrib_config['value'])

        elif self.type() == 'uniform':
            uniform_range = self._distrib_config['range']
            distrib_str = 'Uniform({}, {})'.format(uniform_range[0],
                                                   uniform_range[1])

        else:
            raise NotImplementedError

        return distrib_str


    def type(self):
        return self._distrib_config['type']

    def sample(self):

        if self.type() == 'scalar':
            assert 'value' in self._distrib_config, \
                "Expected key \"value\" to exist in distribution config, " + \
                "but it does not."
            sample = self._distrib_config['value']

        elif self.type() == 'uniform':
            assert 'range' in self._distrib_config, \
                "Expected key \"range\" to exist in distribution config, " + \
                "but it does not."
            uniform_range = self._distrib_config['range']

            uniform_range_len = len(uniform_range)
            assert uniform_range_len == 2, \
                "Expected range of uniform distribution to be of length 2, " + \
                "but it has length {}.".format(uniform_range_len)

            min_value = uniform_range[0]
            max_value = uniform_range[1]

            assert min_value <= max_value, \
                ("Expected min_value ({}) to be less than or equal to max_value ({}), " + \
                 "but it is not.").format(min_value, max_value)

            sample = np.random.uniform(low=min_value, high=max_value)

        else:
            raise NotImplementedError

        return sample

def sample_from_distrib(distrib_input):

    if isinstance(distrib_input, Distrib):
        sample = distrib_input.sample()

    elif isinstance(distrib_input, dict):
        # Assume it's a dict that defines a distribution
        distrib = Distrib(distrib_input)
        sample = distrib.sample()

    elif isinstance(distrib_input, float):
        # Assume it's a scalar
        sample = distrib_input

    else:
        raise NotImplementedError

    return sample