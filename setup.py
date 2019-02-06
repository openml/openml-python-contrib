# -*- coding: utf-8 -*-

import setuptools
import sys

with open("openmlcontrib/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

dependency_links = []

try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)

try:
    import scipy
except ImportError:
    print('scipy is required during installation')
    sys.exit(1)


setuptools.setup(name="openml-contrib",
                 author="Jan N. van Rijn",
                 description="Convenience functions for integrating openml package with several other libraries",
                 license="BSD 3-clause",
                 url="https://www.openml.org/",
                 version=version,
                 packages=setuptools.find_packages(),
                 package_data={'': ['*.txt', '*.md']},
                 install_requires=[
                     'ConfigSpace',
                     'numpy',
                     'scipy',
                     'openml',
                     'pandas>=0.24.0',
                 ],
                 extras_require={
                     'test': [
                         'nbconvert',
                         'jupyter_client',
                         'matplotlib'
                     ]
                 },
                 test_suite="pytest",
                 classifiers=['Intended Audience :: Science/Research',
                              'Intended Audience :: Developers',
                              'License :: OSI Approved :: BSD License',
                              'Programming Language :: Python',
                              'Topic :: Software Development',
                              'Topic :: Scientific/Engineering',
                              'Operating System :: POSIX',
                              'Operating System :: Unix',
                              'Operating System :: MacOS',
                              'Programming Language :: Python :: 2',
                              'Programming Language :: Python :: 2.7',
                              'Programming Language :: Python :: 3',
                              'Programming Language :: Python :: 3.4',
                              'Programming Language :: Python :: 3.5',
                              'Programming Language :: Python :: 3.6'])
