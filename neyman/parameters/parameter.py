from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Parameter(object):
  """ Base class for model parameters. 

  A `Parameter` is a thin wrapper over a `tf.Distribution` which
  simplifes the generation, update and bookkeeping of each
  individual model parameters.

  The dimension of the parameter is determined by the `tf.Distribution`
  `event_shape`. In order to simplify parameter sampling the input
  `batch_shape` as to be [] (scalar). Please use
  `tf.contrib.distributions.Independent` or a multi-variate `tf.Distribution`
  to create multi-dimensional Parameters.

  """

  def __init__(self, *args, **kwargs):
    """ Create a new model parameter.

    """
    
    # use same scope 
    name = kwargs.get('name', type(self).__name__)
    with tf.name_scope(name) as ns:
      kwargs['name'] = ns

    super(Parameter, self).__init__(*args, **kwargs)

