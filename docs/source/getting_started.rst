===============
Getting started
===============

.. autosummary::
   :toctree: generated
   
.. _installation:

Installation
------------

To use GWDA, first install it using pip:

.. code-block:: console

   (.venv) $ pip install xxx

Modules
----------------

To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

For example:

>>> import gwda
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']