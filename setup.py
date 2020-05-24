"""
stat-utils
"""

import io
import os
import re
from setuptools import setup

requirements = ["black", "numpy", "pytest", "scipy"]


def package_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file_path = os.path.join(current_dir, "stat_utils", "__init__.py")
    with io.open(version_file_path, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


setup(
    name="stat-utils",
    version=package_version(),
    description="A library for various statistical learning and inference methods.",
    author="Timothy Lee",
    author_email="timothyelee@cmu.edu",
    packages=["stat_utils"],
    package_dir={"": "."},
    install_requires=requirements,
)
