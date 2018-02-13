from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect as _inspect

from neyman.models.parameter import Parameter as _Parameter
from tensorflow.contrib import distributions as _distributions

# Automatically generate parameter classes from classes in
# tf.contrib.distributions. Idea/code taken from Edward library.
_globals = globals()
for _name in sorted(dir(_distributions)):
  _candidate = getattr(_distributions, _name)
  if (_inspect.isclass(_candidate) and
          _candidate != _distributions.Distribution and
          issubclass(_candidate, _distributions.Distribution)):

    # to use _candidate's docstring, must write a new __init__ method
    def __init__(self, *args, **kwargs):
      _Parameter.__init__(self, *args, **kwargs)
    __init__.__doc__ = _candidate.__init__.__doc__
    _params = {'__doc__': _candidate.__doc__,
               '__init__': __init__}
    _globals[_name] = type(_name, (_Parameter, _candidate), _params)

    del _candidate

del absolute_import
del division
del print_function
