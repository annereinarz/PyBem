PyBem
=====

BEM code for solving the heat equation.

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
