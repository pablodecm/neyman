from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import tensorflow as tf

from neyman.models import Parameter 

def asimov_dataset(observed_vars):
  """ Returns a dict with Asimov dataset tensors.
  
      observed_vars: list of variables that are observed (data).
  
  """

  asimov = OrderedDict() 

  for obs_var in observed_vars:
    if isinstance(obs_var, Parameter):
      # mean will match the expectation for most distributions
      # gradient is stopped because it will be used as pseudo-data
      asimov[obs_var] = tf.stop_gradient(obs_var.mean())
    else:
      raise TypeError("Observed variables have to be Parameters")
  
  return asimov

