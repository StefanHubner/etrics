#!/usr/bin/python3

from setuptools import setup

setup(name='Etrics',
	version='1.0',
	description='Econometrics Toolkit for Regression, Integration and Computational Statistics',
	author='Stefan Hubner',
	author_email='s.hubner@tilburguniversity.edu',
	url='http://git.hubner.co.at/etrics',
	packages=['etrics'],
	install_requires = [ 'scipy' ],
	classifiers = [
		'Programming Language :: Python',
		'Programming Language :: Python :: 3',
		'License :: OSI Approved :: GNU General Public License (GPL)',
		'Operating System :: OS Independent',
		'Environment :: Console',
		'Intended Audience :: Science/Research',
		'Topic :: Scientific/Engineering' ]
)
