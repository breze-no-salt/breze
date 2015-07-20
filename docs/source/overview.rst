Overview of breze, climin and brummlearn
========================================

The machine learning software developed at BRML is currently divided into
three packages:

 - *climin*, hosted at http://github.com/BRML/climin
 - *breze*, hosted at http://github.com/BRML/breze
 - *brummlearn*, hosted at http://github.com/BRML/brummlearn.

The division into three packages might seem unnecessary at first and the
aim of this document is to line out the reasoning behind it. The packages
*climin* and *breze* follow the UNIX ideology to do one thing and do it
well. *Brummlearn* on the other hand plans to tie things together and
provide easy to use functionalities. Climin focuses entirely on 
optimization, breze is a package build around Theano to ease 
the most common use cases and provide code for various models and their
components. Brummlearn aims to bring several things together.

What distinguishes these packages from the popular sklearn is that
cutting-edge algorithms are welcome, as are dependencies. All the packages
are mainly aimed at researchers.


Climin
------

climin is  meant for optimization of machine learning problems.
Currently, optimization in machine learning is mostly done either with a 
home grown stochastic gradient (or something
slightly more complicated) or scipy's ``optimize`` module. The research 
community is somewhat fragmented, with various implementations with different
APIs lounging around the net.
Scipy 
is aimed towards general optimization, which differs from optimization
for machine learning in one critical point: optimization in machine learning
does not stop when it has converged, but when some other, external criteria
is satisfied. Monitoring intermediate statistics during
optimization is crucial to the user: validation errors, parameters, etc.
Especially during long running experiments, the user wants to get access
to a wide range of values to guide the development.

Due to this, climin is build around the concept of Python iterators, which
do that exact thing: they provide a framework for iteration. In the case
of climin, each parameter update is considered as one iteration::

   for info in my_optimizer:
       print info['loss']
       if info['loss'] < 0.1:
          break

Instead of defining a callable (making it necessary to scroll and pass around
local variables), the user can write what she is doing in every iteration
right in the block of a simple for loop. It is straightforward to perform
actions such as writing parameters to disk, printing out the current loss,
checking the validation error or sending an email to your mobile once you've
beat the current state of the art in your field.

Furthermore, the idea is to have a consistent API for a lot of optimizers.
Different optimizers are good for different problems and maybe you just have
not tried the right optimizer. Thus, climin comes with a wide range of
standard optimizers such as stochastic gradient, rprop, lbfgs or more modern
ones such as rmsprop and stochastic meta descent.


Breze
-----

Theano is a very impressive library which offers three unique features:
automatic differentiation, optimization of expressions and transparent
CPU/GPU usage. Nevertheless, only few models are of openly available.
Furthermore, some tasks are still rather repetitive. Breze jumps in in the
following two ways:

 - Quite some models are implemented: neural networks,
   several feature extractors and recurrent networks.
 - Breze makes it easier to do the bookkeeping: holding groups of expressions,
   accessing parameter variables and their values, compiling functions
   with less key strokes.

That being said, Breze is opinionated in the following two ways:

 - Optimizers are not being written in Theano; instead, that part is
   being taken care of by making use of climin. Breze is designed with
   climin in mind.
 - GPU usage is not properly tested at the time of writing.
 - We do something that Theano is not explicitly designed for, we
   steal memory regions. Up until now, this works fine, though.


Brummlearn
----------

This is the place where everything comes together. There are
modules for classic algorithms such as PCA or the more recent
developments such as XCA, filtering time series or extract
features from EMG data or sampling. Some of these depend on
Breze, some on climin, and some only on scipy. Most of these
algorithms feature out-of-the-box usage patterns.
