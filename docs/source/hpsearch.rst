Automated Hyperparameter Search
===============================

The process of optimizing of hyper parameters can be tedious and
time consuming. Several methods have been proposed to automate
this and leverage machine learning methods to guide the search
for good hyper parameters.

The idea of this package is to aggregate the knowledge on HP search
gained from several publications, such as Bergstra1, Bergstra2, Snoek1.

There are two main ways to search for hyper parameters. One is to ignore all the
knowledge we have gained form previous experiments. The other is to model the
landscape of the cost function (which maps hyper parameters to something like
a validation loss) and use that model to find promising new candidate solutions.

In the former case, random search and grid search are popular methods. In the
latter case, Bayesian optimization is used for search.

The principal ideas of this module are to easily integrate the methods available
into existing scripts and frameworks. Thus, the functionality is rather low
level. We begin by defining a _search space_ which is helpful to encode prior
knowledge on the hyper parameters. Blablabla.
