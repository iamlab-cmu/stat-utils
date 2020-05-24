import random

import numpy as np
import pytest

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.fixture
def random_seed():

    # Set random seed. This ensures that the unit test won't fail improbably.
    # This assumes the tests are expected to be robust to random seed choice.
    # When developing a new unit test, test that it works first for a number
    # of different random seed choices.
    random_seed = 7
    random.seed(random_seed)
    np.random.seed(random_seed)
