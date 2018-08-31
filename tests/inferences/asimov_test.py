from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from neyman.inferences import asimov_dataset
from neyman.models import ed

ds = tf.contrib.distributions

class test_asimov_dataset(tf.test.TestCase):

  def test_asimov_one_poisson(self):
    rate = np.array(7.)
    pois = ed.Poisson(rate=rate)
    asimov_dict = asimov_dataset([pois])

    with self.test_session() as sess:
      asimov_dict_val = sess.run(asimov_dict) 
    
    self.assertAllClose(asimov_dict_val[pois], rate)

  def test_asimov_two_poisson(self):
    rates = np.array([7.,3.])
    poiss = [ed.Poisson(rate=rate) for rate in rates]
    asimov_dict = asimov_dataset(poiss)

    with self.test_session() as sess:
      asimov_dict_val = sess.run(asimov_dict) 
    
    for rate, pois in zip(rates,poiss):
      self.assertAllClose(asimov_dict_val[pois], rate)

  def test_asimov_ind_poisson(self):
    rates = np.array([7.,3.])
    ind_pois = ed.Independent(distribution=ds.Poisson(rates),
        reinterpreted_batch_ndims=1)
    asimov_dict = asimov_dataset([ind_pois])

    with self.test_session() as sess:
      asimov_dict_val = sess.run(asimov_dict) 

    self.assertAllClose(asimov_dict_val[ind_pois], rates)
