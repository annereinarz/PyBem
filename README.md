PyBem
=====

BEM code for solving the heat equation.

More precisely, we solve the Dirichlet problem in a smooth domain Omega with boundary
Gamma and a time interval I = (0,4)
Find  u: Omega x I --> R satisfying:
  (partial_t-Delta)u = 0,          in Omega x I
  u = 0,                           at Omega x {t=0}
  u|_{Gamma x I} = g,              in Gamma x I,


Prerequisites
-------------

* Python 2.7
* numpy      (tested with 1.8.2)
* scipy      (tested with 0.13.3)
* matplotlib (tested with 1.3.1)

How to use
----------

Running `nosetests` in the root directory will run all the unit tests.

If you want to see the library in action, you can run:

    PYTHONPATH=.:test python -i test/functional.py

This will run a longer calculation and keep a plot of the result open.
