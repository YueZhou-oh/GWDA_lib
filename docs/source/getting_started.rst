===============
Getting started
===============

.. autosummary::
   :toctree: generated

.. _install:

Install
-------

To use GWDA, we strongly recommend using the release of `GWDA container <https://hub.docker.com/layers/zzhopezhou/astropre/gwda/images/sha256-38d1ce5842c31632cb96da1e542d0e10ab1732fb29696e9798a984af0fb2cdd8?context=repo>`_ with GPU nodes.

You can launch an instance of the `astropre container <https://hub.docker.com/layers/zzhopezhou/astropre/gwda/images/sha256-38d1ce5842c31632cb96da1e542d0e10ab1732fb29696e9798a984af0fb2cdd8?context=repo>`_ container and 
mount `GWDA <https://github.com/YueZhou-oh/GWDA_lib>`_ as well as your dataset with the following Docker commands.

.. code-block:: console

   $ docker pull zzhopezhou/astropre:gwda
   $ docker run --gpus all -itd -v /path/to/GWDA_lib:/workspace/GWDA_lib -v /path/to/dataset:/workspace/dataset zzhopezhou/astropre:gwda

In the container, two environments of different python version are provided.
Specifically, the ``base`` environment is mainly used for model training and the ``waveform`` environment is for data generation.

.. code-block:: console

   (base) root@93b17a314f9d:/workspace# which python
   /opt/conda/bin/python
   (base) root@93b17a314f9d:/workspace# conda activate waveform
   (waveform) root@93b17a314f9d:/workspace# which python
   /opt/conda/envs/waveform/bin/python


If you can't use this for some reason, use the latest pytorch, cuda, nccl, NVIDIA `APEX <https://github.com/NVIDIA/apex#quick-start>`_ and
make sure that the following required `python packages <https://github.com/YueZhou-oh/GWDA_lib/blob/main/requirements.txt>`_ are successfully installed.

.. code-block:: console
   :linenos:

   astropy
   corner
   deepspeed
   esbonio
   fastdtw
   few
   fftw
   gwdatafind
   gwpy
   gwsurrogate
   hydra-core
   imbalanced-learn==0.11.0
   lalsimulation
   lalsuite
   librosa
   ligotimegps
   lisaorbits
   nflows
   numpy
   matplotlib
   omegaconf
   pandas
   pillow
   pybind11
   PyCBC
   rich
   scikit-learn
   scipy
   speechbrain
   statsmodels
   tensorrt
   torch
   torchsummary
   torchtext
   torchvision
   transformers
   wandb
   
