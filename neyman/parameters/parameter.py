from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

try:
    from tensorflow.python.client.session import \
              register_session_run_conversion_functions
except Exception as e:
    raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


class Parameter(object):
  """ Base class for model parameters. 

  A `Parameter` is a wrapper over a `tf.Distribution`, which
  simplifes the generation, update and bookkeeping of each
  individual model parameter.

  By default it also wraps a placeholder which will represent
  the parameters in the computation graph and can be feeded
  with a Tensor handles for efficiency.

  All relevant Tensor methods will act over the placeholder, which
  is actually as Tensor. The placeholder Tensor can be accesed
  directly with value(). This idea and the relevant code have
  been taken from Edward library. 

  The dimension of the parameter is determined by the `tf.Distribution`
  `event_shape`.  Please use `tf.contrib.distributions.Independent`
  or a multi-variate `tf.Distribution` to create multi-dimensional
  Parameters.

  """

  def __init__(self, *args, **kwargs):
    """ Create a parameter.

    """
    
    # use same scope 
    name = kwargs.get('name', type(self).__name__)
    with tf.name_scope(name) as ns:
      kwargs['name'] = ns


    super(Parameter, self).__init__(*args, **kwargs)

    with tf.name_scope(name) as ns:
      self._value = tf.placeholder(shape=(None,), dtype=self.dtype,
                                   name="placeholder")


  def value(self):
    """Get tensor that the random variable corresponds to."""
    return self._value

  @staticmethod
  def _overload_all_operators():
    """Register overloads for all operators."""
    for operator in tf.Tensor.OVERLOADABLE_OPERATORS:
      Parameter._overload_operator(operator)

  @staticmethod
  def _overload_operator(operator):
    """Defer an operator overload to `tf.Tensor`.
    We pull the operator out of tf.Tensor dynamically to avoid ordering issues.
    Args:
      operator: string. The operator name.
    """
    def _run_op(a, *args):
      return getattr(tf.Tensor, operator)(a.value(), *args)
    # Propagate __doc__ to wrapper
    try:
      _run_op.__doc__ = getattr(tf.Tensor, operator).__doc__
    except AttributeError:
      pass

    setattr(Parameter, operator, _run_op)  

  # "This enables the Variable's overloaded "right" binary operators to
  # run when the left operand is an ndarray, because it accords the
  # Variable class higher priority than an ndarray, or a numpy matrix."
  # Taken from implementation of tf.Tensor.
  __array_priority__ = 100

  @staticmethod
  def _session_run_conversion_fetch_function(tensor):
    return ([tensor.value()], lambda val: val[0])

  @staticmethod
  def _session_run_conversion_feed_function(feed, feed_val):
    return [(feed.value(), feed_val)]

  @staticmethod
  def _session_run_conversion_feed_function_for_partial_run(feed):
    return [feed.value()]

  @staticmethod
  def _tensor_conversion_function(v, dtype=None, name=None, as_ref=False):
    _ = name, as_ref
    if dtype and not dtype.is_compatible_with(v.dtype):
      raise ValueError(
          "Incompatible type conversion requested to type '%s' for variable "
          "of type '%s'" % (dtype.name, v.dtype.name))
    return v.value()

Parameter._overload_all_operators()

register_session_run_conversion_functions(
    Parameter,
    Parameter._session_run_conversion_fetch_function,
    Parameter._session_run_conversion_feed_function,
    Parameter._session_run_conversion_feed_function_for_partial_run)

tf.register_tensor_conversion_function(
    Parameter, Parameter._tensor_conversion_function)

