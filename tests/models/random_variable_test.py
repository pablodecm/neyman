from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from neyman.models import ed

class test_parameter_class(tf.test.TestCase):

  def _test_parameter_constructor(self, param_class, *args, **kwargs):
    param = param_class(*args, **kwargs) 

  def test_parameter_constructor(self):
    with self.test_session():
      self._test_parameter_constructor(ed.Normal, loc=0.0, scale=1.0)
      self._test_parameter_constructor(ed.Poisson, rate=10.)

if __name__ == '__main__':
  tf.test.main()

