from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import tensorflow as tf
from edward.models import RandomVariable
from edward.models.random_variable import _RANDOM_VARIABLE_COLLECTION

class Parameter(RandomVariable):
  """ Base class for model parameters. 

  A `Parameter` is a wrapper over a `tf.Distribution`, which
  simplifes the generation, update and bookkeeping of each
  individual model parameter.

  By default it also wraps a placeholder which will represent
  the parameters in the computation graph and can be feeded
  with a Tensor handles for efficiency.

  All relevant Tensor methods will act over the placeholder, which
  is actually as Tensor. The placeholder Tensor can be accesed
  directly with value(). From a technical standpoint, it inherits from Edward's
  RandomVariable in order to maximize compatibility and code reuse.

  The dimension of the parameter is determined by the `tf.Distribution`
  `event_shape`.  Please use `neyman.parameters.Independent`
  or a multi-variate `neyman.parameter` to create multi-dimensional
  Parameters.

  """

  def __init__(self, *args, **kwargs):
    """ Create a new parameter variable.

    """
    
    # use same scope 
    name = kwargs.get('name', type(self).__name__)
    with tf.name_scope(name) as ns:
      kwargs['name'] = ns

    # pop and store RandomVariable-specific parameters in _kwargs
    sample_shape = kwargs.pop('sample_shape', (None,))
    value = kwargs.pop('value', None)
    collections = kwargs.pop('collections', ["random_variables"])

    # store args, kwargs for easy graph copying
    self._args = args
    self._kwargs = kwargs.copy()

    if sample_shape != ():
      self._kwargs['sample_shape'] = sample_shape
    if value is not None:
      self._kwargs['value'] = value
    if collections != ["random_variables"]:
      self._kwargs['collections'] = collections

    super(RandomVariable, self).__init__(*args, **kwargs)

    self._placeholder_shape = tf.TensorShape(sample_shape) \
        .concatenate(self.batch_shape).concatenate(self.event_shape)

    with tf.name_scope(name) as ns:
      self._value = tf.placeholder_with_default(
          self.sample(tf.TensorShape((1,))),
          shape=self._placeholder_shape, name="placeholder")

    for collection in collections:
      if collection == "random_variables":
        collection = _RANDOM_VARIABLE_COLLECTION
      collection[tf.get_default_graph()].append(self)
  
