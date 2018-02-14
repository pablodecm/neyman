from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from neyman.models import RandomVariable 


def log_likelihood(data):
  """ Returns a dictionary of log_prob of data. """

  log_likelihood = {}
  parameters = set()

  for (key, value) in six.iteritems(data):
    if isinstance(key, RandomVariable):
      data_value = tf.convert_to_tensor(value)
      log_likelihood[key] = key.log_prob(data_value)
      parameters.update(set(key.get_ancestors()))
    else:
      raise TypeError("The keys of data have to be Parameters")
  
  return log_likelihood, list(parameters)

def log_likelihood_with_constraints(data, pois, constraints=None):
  """ Returns a dictionary of log_prob of data.

  The likelihood is mutiplied by the constraints of all parameters
  that are not POIs. Will allow the specification of user defined
  parameter constraints in the future.
  
  """

  ll_const = {}

  ll, pars = log_likelihood(data)
  ll_const.update(ll)

  if constraints:
    raise NotImplementedError("Use of user-defined constraints not \
        implemented yet")

  # remove parameters of interest
  pars_const = set(pars.copy())
  pars_const.difference_update(set(pois))

  for par in pars_const:
    if not par.get_ancestors(): 
      ll_const[par] = key.log_prob(par)
    else:
      raise ValueError("Non-POI cannot have ancestors")

  return ll_const, list(pars)

