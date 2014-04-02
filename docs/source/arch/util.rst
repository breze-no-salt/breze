Utilities
=========

.. autoclass:: breze.arch.util.Model
   :members: __init__, function, var_exp_for_gpu,

.. autoclass:: breze.arch.util.ParameterSet
   :members: __init__, __contains__, __getitem__, __setitem__

   
Nested Lists for Theano, etc.
---

.. autofunction:: breze.arch.util.flatten
.. autofunction:: breze.arch.util.unflatten
.. autofunction:: breze.arch.util.theano_function_with_nested_exprs
.. autofunction:: breze.arch.util.theano_expr_bfs
.. autofunction:: breze.arch.util.tell_deterministic


GPU related utilities
---

.. autofunction:: breze.arch.util.cpu_tensor_to_gpu
.. autofunction:: breze.arch.util.cpu_tensor_to_gpu_nested
.. autofunction:: breze.arch.util.cpu_expr_to_gpu
.. autofunction:: breze.arch.util.cpu_expr_to_gpu_nested
.. autofunction:: breze.arch.util.garray_to_cudandarray
.. autofunction:: breze.arch.util.garray_to_cudandarray_nested
.. autofunction:: breze.arch.util.gnumpy_func_wrap


Other
---

.. autofunction:: breze.arch.util.get_named_variables
.. autofunction:: breze.arch.util.lookup
.. autofunction:: breze.arch.util.lookup_some_key
