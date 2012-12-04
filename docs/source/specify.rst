Specifiying losses, norms, transfer functions etc.
==================================================

Many models within brummlearn make use of Breze as its building block.
To maintain flexibility and conciseness, configuring those models can
be achieved twofold: either by using a string or by using a function
that follows the specific API.

Using the builtin loss functions
--------------------------------

Let us start with an example. To instantiate a linear model, we can make
use of the following notation::

   from brummlearn.glm import Linear
   model = Linear(5, 1, loss='squared')

In this case, we specify the sum of squares loss as string. The logic behind
this aims to be straight forward: for losses, a lookup is done in the module
``breze.components.distance``. Thus, the function
``breze.component.distance.squared`` is used as a loss. This function follows
a simple protocol. In the case of an supervised model, it is called with the
target as its first argument and the output of the model as its second argument.
However, both are required to be Theano variables. In the case of an
unsupervised model, the output of the model is the only argument passed on to
the loss.

A list of supervised losses can be found by checking the contents of the
``breze.components.distance`` module::

   >>> from breze.components import distance
   >>> dir(distance)
    ['T',
     '__builtins__',
     '__doc__',
     '__file__',
     '__name__',
     '__package__',
     'absolute',
     'bernoulli_kl',
     'bernoulli_neg_cross_entropy',
     'discrete_entropy',
     'distance_matrix',
     'lookup',
     'nca',
     'neg_cross_entropy',
     'nominal_neg_cross_entropy',
     'norm',
     'squared']

Some of these are just global variable of course.


Using custom loss functions
---------------------------

Using your own loss function comes down to implementing it following the above
protocol and working on Theano variables. We can thus define the sum of squares
loss ourself as follows::

    def squared(target, output):
        d = target - output
        return (d**2).sum()

We can also use more complicated loss functions. The Huber loss for example is
a mix of the absolute error and the squared error, depending on the size of the
error. It depends on an additional threshold parameter and is defined as follow:

.. math::
    L_\delta (a) & = & \frac{a^2}{2} \qquad \qquad & \text{if  } |a| \le \delta , \\
    L_\delta (a) & = & \delta (|a| - \frac{\delta}{2} ), \qquad &\text{else}.

We can implement this as follows::

    import theano.tensor as T
    delta = 0.1
    def huber(target, output):
        d = target - output
        a = .5 * d**2
        b = delta * (abs(d) - delta / 2.)
        l = T.switch(abs(d) <= delta, a, b)
        return l.sum()

Unfortunately, we will have to set a global variable for this. The most elegant
solution is to use a function template::

    import theano.tensor as T
    def make_huber(delta):
        def inner(target, output):
            d = target - output
            a = .5 * d**2
            b = delta * (abs(d) - delta / 2.)
            l = T.switch(abs(d) <= delta, a, b)
            return l.sum()
        return inner

    my_huber = make_huber(0.1)

This way we can create wild loss functions.


Using norms and transfer functions
----------------------------------

The story is similar when using norms and loss functions. In the former
case, the module of interest is ``breze.component.norm``. The protocol 
is that a single argument, a Theano variable, is given. The result is
expected to be a Theano variable of the same shape. This is also
the case for transfer functions, except that the module in question is
``breze.component.transfer``.
