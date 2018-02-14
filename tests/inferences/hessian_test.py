from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from neyman.inferences import batch_hessian 

class test_batch_hessian(tf.test.TestCase):

  def _test_hessian_parametrized(self, a_value, b_value):

    a = tf.convert_to_tensor(a_value)
    b = tf.convert_to_tensor(b_value)
    f = a**5.+b**3.+a*b
    b_hess, b_grad = batch_hessian(f, [a,b])
    t_grad = tf.stack([5*a**4.+b,3.*b**2.+a], axis=-1)
    t_hess = tf.stack([tf.stack([20*a**3.,tf.ones_like(a)], axis=-1),
                      tf.stack([tf.ones_like(b),6.*b], axis=-1)], axis=-1)

    with self.test_session() as sess:
      b_grad_val, t_grad_val = sess.run([b_grad, t_grad]) 
      b_hess_val, t_hess_val = sess.run([b_hess, t_hess]) 

    self.assertAllClose(b_grad_val,t_grad_val)
    self.assertAllClose(b_hess_val,t_hess_val)

  def test_hessian(self):
    self._test_hessian_parametrized([1.],[2.])
    self._test_hessian_parametrized([1.,4.],[2.,6.])
    self._test_hessian_parametrized(1.,2.)
