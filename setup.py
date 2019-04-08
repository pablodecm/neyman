from setuptools import setup

__version__='0.0.1'

setup(
  name='neyman',
  version=__version__,
  description='A modern library for classical statistical inference',
  author='Pablo de Castro',
  author_email='pablodecm@gmail.com',
  packages=['neyman','neyman.models','neyman.inferences'],
  install_requires=['numpy>=1.7','tensorflow-probability>=0.6'],
  tests_require=['pytest', 'pytest-pep8'],
)


