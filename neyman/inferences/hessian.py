from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def batch_gradient(ys, pars, name="batched_gradients",
    colocate_gradients_with_ops=False, gate_gradients=False,
    aggregation_method=None):
  """ Variation of `tf.gradients` for batched gradient computation. """

  kwargs = {
      'colocate_gradients_with_ops': colocate_gradients_with_ops,
      'gate_gradients': gate_gradients,
      'aggregation_method': aggregation_method
  }

  _gradients = tf.gradients(ys, pars, **kwargs)
  # None gradients to zero
  grad_gradients = [tf.zeros_like(pars[j]) if grad is None else grad
                      for j, grad in enumerate(_gradients)]
  # expand dimensions if scalar-dim parameter
  _gradients = [tf.expand_dims(g,1) if len(g.shape)==1 else g for g
                in _gradients]
  # reshape to deal with scalars 
  grad_shape = pars[0].shape.concatenate(tf.TensorShape([len(pars)]))

  return tf.reshape(tf.stack(_gradients,axis=-1), grad_shape)
  
def batch_hessian(ys, pars, name="batched_hessian",
    colocate_gradients_with_ops=False, gate_gradients=False,
    aggregation_method=None):
  """ Variation of `tf.hessians` for batched hessian computation. """

  kwargs = {
      'colocate_gradients_with_ops': colocate_gradients_with_ops,
      'gate_gradients': gate_gradients,
      'aggregation_method': aggregation_method
  }

  batch_hessian = []

  # Compute first-order derivatives
  _batch_gradients = batch_gradient(ys, pars, **kwargs)

  for i, _gradient in enumerate(tf.unstack(_batch_gradients, axis=-1)):
    grad_gradients = tf.gradients(_gradient, pars, **kwargs)
    # zero also when second derivative is None
    grad_gradients = [tf.zeros_like(pars[j]) if g_grad is None else g_grad
                          for j, g_grad in enumerate(grad_gradients)]
    # expand dimensions if scalar-dim parameter
    grad_gradients = [tf.expand_dims(g,1) if len(g.shape)==1 else g for g
                      in grad_gradients]
    # reshape to deal with scalars 
    grad_shape = pars[i].shape.concatenate(tf.TensorShape([len(pars)]))

    batch_hessian.append(tf.reshape(tf.stack(grad_gradients,axis=-1), grad_shape))

  return tf.stack(batch_hessian, axis=-1), _batch_gradients
    
