.. breze documentation master file, created by
   sphinx-quickstart on Thu Apr 26 12:04:40 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to breze's documentation!
======================================

Basics
------

.. toctree::
   specify


Models and Algorithms
---------------------

Learning representations, clustering:

.. toctree::
   pca
   xca
   sparsefiltering
   rica
   cca
   sfa
   kmeans
   rim
   sgvb
   :maxdepth: 1

Denoising:

.. toctree::
   lde 
   :maxdepth: 1

Supervised Learning

.. toctree::
   rnn
   mlp
   :maxdepth: 1


Sampling

.. toctree::
   sampling/hmc


Trainers

.. toctree::
   learn/trainer/trainer


Helpers, convenience functions and tools
----------------------------------------

.. toctree::
   feature
   data
   utils
   display
   :maxdepth: 1


Architectures, Components
-------------------------

.. toctree::    
   arch/component/norm
   arch/component/transfer
   arch/component/loss
   arch/component/corrupt
   arch/component/misc
   arch/component/layer
   arch/component/common
   arch/component/distributions/normal
   arch/component/distributions/mvn
   arch/util

For variance propagation:

.. toctree::
   arch/component/varprop/common


Implementation Notes
--------------------

.. toctree::
   varprop


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

