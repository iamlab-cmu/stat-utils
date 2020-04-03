"""
stat-utils
"""

from setuptools import setup

requirements = [
    'numpy',
    'pytest',
    'scipy'
]

setup(name='stat-utils',
      version='0.1.0',
      description='A library for various statistical learning and inference methods.',
      author='Timothy Lee',
      author_email='timothyelee@cmu.edu',
      packages=['stat_utils'],
      package_dir={'': '.'},
      install_requires = requirements
      )
