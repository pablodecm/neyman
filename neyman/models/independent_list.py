from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops.distributions import distribution as distribution_lib

class IndependentList(distribution_lib.Distribution):
  """Independent distribution from a list of 1D distributions.
  """

  def __init__(
      self, distributions,
      validate_args=False, name=None):
    """  Construct a `IndependentList` distribution.
    """
    parameters = locals()
    name = name or "IndependentList"
    self._distributions = distributions

    super(IndependentList, self).__init__(
        dtype=self._distributions[0].dtype,
        reparameterization_type=self._distributions[0].reparameterization_type,
        validate_args=validate_args,
        allow_nan_stats=self._distributions[0].allow_nan_stats,
        parameters=parameters,
        graph_parents=(
            sum([d._graph_parents for d in self._distributions],[])
        ),  # pylint: disable=protected-access
        name=name)

  @property
  def distributions(self):
    return self._distributions

  def _batch_shape_tensor(self):
    batch_shape = self.distributions[0].batch_shape_tensor()
    return batch_shape

  def _batch_shape(self):
    batch_shape = self.distributions[0].batch_shape
    return batch_shape

  def _event_shape_tensor(self):
    event_shape = tf.convert_to_tensor([len(self.distributions)], dtype=tf.float32)
    return event_shape

  def _event_shape(self):
    event_shape = tf.TensorShape([len(self.distributions)])
    return event_shape

  def _sample_n(self, n, seed):
    samples = [ d.sample(n) for d in self.distributions]
    return tf.stack(samples, axis=-1)

  def _log_prob(self, x):
    splitted_x = tf.split(x, len(self.distributions), axis=-1)
    log_probs = [d.log_prob(x_i) for x_i, d
                    in zip(splitted_x, self.distributions)]
    return tf.reduce_sum(log_probs, axis=0)

