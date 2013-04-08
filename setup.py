from setuptools import setup, find_packages
import sys, os

version = '0.1'

setup(name='polyglot',
      version=version,
      description="polyglot is a tool for detecting multilingual documents and identifying the languages therein.",
      long_description= open("README").read(),
      classifiers=[], # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
      keywords=['language detection', 'multilingual documents', 'text classification'],
      author='Marco Lui',
      author_email='saffsd@gmail.com',
      url='https://github.com/saffsd/polyglot',
      license='BSD',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      zip_safe=False,
      install_requires=[
          # -*- Extra requirements: -*-
          'numpy',
      ],
      entry_points= {
        'console_scripts': [
          'polyglot = polyglot.detect:main',
        ],
      },
      )
