from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Model(object):
  """ Base class for statistical models. 

  Provides basic methods for building likelihoods from
  data and from sampled parameter/model toys.

  """

  def __init__(self,
               model_fn, 
               model_params):
    """ Creates a statistical model.
    

    Args:

      model_fn: A function which allows the signature model_fn(**parameters)
      where parameters would be a dictionary of Tensors generated for each
      for each 'Parameter' in model_params. The function has to return a
      `tf.Distribution` type object (with `sample` and `log_prob` methods)
      or a dictionary of them, there the keys will represent different
      independent modelled datasets.

      model_params: A dictionary of Parameters, with str keys that match
      `model_fn` function call. The compatibility of the model_fn call with
      the keys in model_params will be checked at construction time.

    """

    self._model_fn = model_fn 
    self._model_params = model_params

  def log_prob(parameters, data):  
    """ Returns a the log_likelihood of the data obtained given  
    the parameters.  

    Args:

      data: a single Tensor (or convertible) or a dict of Tensors
      with keys matching output from `model_fn` when several
      independent datasets are used.
    
    Returns:
      log_likelihood: Tensor or dict of Tensors (keys being the
      different independent datasets).
       
    """

    model = model_fn(**parameters) 

    if isinstance(data, dict) and isinstance(model, dict):
      log_likelihood = dict()
      for k,v in model:
        log_likelihood[k] = model[k].log_prob(data[k])
    elif isinstance(data, dict) or isinstance(model, dict):
      raise ValueError("Only model_fn or data are a dict")
    else:
      log_likelihood = model.log_prob(data)

    return log_likelihood
    
