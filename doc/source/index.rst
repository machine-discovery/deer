.. deerx documentation master file, created by
   sphinx-quickstart on Wed May 29 10:48:34 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DEER: parallelizing sequential models
=====================================

DEER is a library that allows you to evaluate sequential models in parallel, like ODE solver and RNN.
The parallelization is done over the time axis, where sequential for-loop is usually used.
This enables fast evaluations and trainings with modern deep learning hardware, such as GPU.

.. toctree::
   :maxdepth: 2
   :caption: Modules

   api/deer/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
