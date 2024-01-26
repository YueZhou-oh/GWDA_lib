===============
Getting started
===============

.. autosummary::
   :toctree: generated

.. _install:

Install
-------

To use GWDA, we strongly recommend using the latest release of `astropre container <https://hub.docker.com/r/zzhopezhou/astropre/tags>`_ with GPU nodes.

You can launch an instance of the `astropre container <https://hub.docker.com/r/zzhopezhou/astropre/tags>`_ container and 
mount `GWDA <https://github.com/YueZhou-oh/GWDA_lib>`_ as well as your dataset with the following Docker commands.

.. code-block:: console

   $ docker pull zzhopezhou/astropre:xxxx
   $ docker run --gpus all -itd -v /path/to/GWDA_lib:/workspace/GWDA_lib -v /path/to/dataset:/workspace/dataset zzhopezhou/astropre:xxxx

In the container, two environments of different python version are provided.
Specifically, the ``base`` environment is mainly used for model training and the ``waveform`` environment is for data generation.

.. code-block:: console

   (base) root@93b17a314f9d:/workspace# which python
   /opt/conda/bin/python
   (base) root@93b17a314f9d:/workspace# conda activate few_env
   (waveform) root@93b17a314f9d:/workspace# which python
   /opt/conda/envs/waveform/bin/python


If you can't use this for some reason, use the latest pytorch, cuda, nccl, NVIDIA `APEX <https://github.com/NVIDIA/apex#quick-start>`_ and
make sure that the following required `python packages <https://github.com/YueZhou-oh/GWDA_lib/blob/main/requirements.txt>`_ are successfully installed.

.. code-block:: console
   :linenos:

   astropy
   deepspeed
   fastdtw
   few
   fftw
   gwdatafind
   gwpy
   gwsurrogate
   lalsuite
   librosa
   ligotimegps
   lisaorbits
   numpy
   matplotlib
   pandas
   pillow
   PyCBC
   scikit-learn
   scipy
   speechbrain
   statsmodels
   torchaudio
   transformers
   

.. _modules:

Modules
----------------

To retrieve a list of random ingredients,
you can use the ``gwda.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

For example:

>>> import gwda
>>> gwda.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']