###################################################
Training examples of AI-centered model
###################################################

.. autosummary::
   :toctree: generated


===========================================
Signal Classification
===========================================

TBA

==============================================
Data Denoising
==============================================

Firstly, downloading demo dataset (``train_data, valid_data, test_data``) from `this repository <https://github.com/AI-HPC-Research-Team/LIGO_noise_suppression>`_.
and put it under `dataset/denoise <https://github.com/YueZhou-oh/GWDA_lib/tree/main/dataset/denoise>`_ folder.
By running `denoise_demo.sh <https://github.com/YueZhou-oh/GWDA_lib/blob/main/demo/denoise_demo.sh>`_ script, your own denoising model can be trained. 

You can modify configurations in `denoise_demo.sh <https://github.com/YueZhou-oh/GWDA_lib/blob/main/demo/denoise_demo.sh>`_ to build your own model with different model size.

.. code-block:: console
    :linenos:

    $ conda activate base
    $ cd /workspace/GWDA_lib/demo
    $ bash denoise_demo.sh

The output log can be seen as follows.

.. code-block:: shell
    :linenos:

      using world size: 2, data-parallel-size: 2, tensor-model-parallel size: 1, pipeline-model-parallel size: 1 
      setting global batch size to 16
      using torch.float16 for parameters ...
      ------------------------ arguments ------------------------
      accumulate_allreduce_grads_in_fp32 .............. False
      adam_beta1 ...................................... 0.9
      xxxxxxx
      -------------------- end of arguments ---------------------
      setting number of micro-batches to constant 1
      > initializing torch distributed ...
      > initializing tensor model parallel with size 1
      > initializing pipeline model parallel with size 1
      > setting random seeds to 1234 ...
      > initializing model parallel cuda seeds on global rank 0, model parallel rank 0, and data parallel rank 0 with model parallel seed: 3952 and data parallel seed: 1234
      > compiling and loading fused kernels ...
      Detected CUDA files, patching ldflags
      Emitting ninja build file /workspace/zhouy/GWDA_lib/demo/../src/model/denoising/fused_kernels/build/build.ninja...
      Building extension module scaled_upper_triang_masked_softmax_cuda...
      Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
      ninja: no work to do.
      Loading extension module scaled_upper_triang_masked_softmax_cuda...
      Detected CUDA files, patching ldflags
      Emitting ninja build file /workspace/zhouy/GWDA_lib/demo/../src/model/denoising/fused_kernels/build/build.ninja...
      Building extension module scaled_masked_softmax_cuda...
      Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
      ninja: no work to do.
      Loading extension module scaled_masked_softmax_cuda...
      Detected CUDA files, patching ldflags
      Emitting ninja build file /workspace/zhouy/GWDA_lib/demo/../src/model/denoising/fused_kernels/build/build.ninja...
      Building extension module fused_mix_prec_layer_norm_cuda...
      Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
      ninja: no work to do.
      Loading extension module fused_mix_prec_layer_norm_cuda...
      >>> done with compiling and loading fused kernels. Compilation time: 3.274 seconds
      time to initialize megatron (seconds): 41.829
      [after megatron is initialized] datetime: 2024-02-02 15:50:01 
      building WaveFormer model ...
      > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 220058673
      > learning rate decay style: linear
      WARNING: could not find the metadata file demo/latest_checkpointed_iteration.txt 
         will not load any checkpoints and will start from random
      time (ms) | load-checkpoint: 0.16
      [after model, optimizer, and learning rate scheduler are built] datetime: 2024-02-02 15:50:01 
      > building train, validation, and test datasets ...
      > building train, validation, and test datasets for BERT ...
      > finished creating BERT datasets ...
      [after dataloaders are built] datetime: 2024-02-02 15:50:06 
      done with setup ...time (ms) | model-and-optimizer-setup: 111.39 | train/valid/test-data-iterators-setup: 4415.50

      training ...
      [before the start of training step] datetime: 2024-02-02 15:50:06 
      iteration        1/   30000 | current time: 1706860208.35 | consumed samples:           16 | elapsed time per iteration (ms): 1996.1 | learning rate: 0.000E+00 | global batch size:    16 | loss scale: 4294967296.0 | number of skipped iterations:   1 | number of nan iterations:   0 |
      time (ms) | backward-compute: 138.46 | backward-params-all-reduce: 32.71 | backward-embedding-all-reduce: 0.04 | optimizer-copy-to-main-grad: 3.17 | optimizer-unscale-and-check-inf: 42.67 | optimizer: 45.94 | batch-generator: 263.80
      ----------------------------------------------------------------------------------------------------
      validation loss at iteration 1 | lm loss value: 4.280033E-01 | lm loss PPL: 1.534191E+00 | 
      --------------------------------------------------------------------------------------------
      iteration        2/   30000 | current time: 1706860208.78 | consumed samples:           32 | elapsed time per iteration (ms): 429.4 | learning rate: 0.000E+00 | global batch size:    16 | loss scale: 2147483648.0 | number of skipped iterations:   1 | number of nan iterations:   0 |
      time (ms) | backward-compute: 31.50 | backward-params-all-reduce: 35.43 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 2.87 | optimizer-unscale-and-check-inf: 12.14 | optimizer: 15.32 | batch-generator: 274.37
      ----------------------------------------------------------------------------------------------------
      validation loss at iteration 2 | lm loss value: 4.258614E-01 | lm loss PPL: 1.530909E+00 | 
      --------------------------------------------------------------------------------------------


==============================================
Signal Detection
==============================================

Firstly, activating ``waveform`` environment.
Then, by running `train_se_mlp.py <https://github.com/YueZhou-oh/GWDA_lib/blob/main/src/model/detection/train_se_mlp.py>`_ script, your own detection model can be trained. 

.. code-block:: console
    :linenos:

    $ conda activate waveform
    $ cd cd /workspace/GWDA_lib/src/model/detection/
    $ python train_se_mlp.py se-mlp.yaml

The output log can be seen as follows.

.. code-block:: shell
    :linenos:

      speechbrain.core - Beginning experiment!
      speechbrain.core - Experiment folder: results/detection_demo22/1607
      speechbrain.core - Info: test_only arg overridden by command line input to: False
      speechbrain.core - Info: auto_mix_prec arg from hparam file is used
      speechbrain.core - 5.6M trainable parameters in Separation
      speechbrain.utils.checkpoints - Would load a checkpoint here, but none found yet.
      speechbrain.utils.epoch_loop - Going into epoch 1
      100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:02<00:00,  5.45it/s, loss1=6.18, loss2=0.693, train_loss=6.18]
      100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.50it/s]
      speechbrain.utils.train_logger - epoch: 1, lr: 5.00e-04 - train si-snr: 6.18, train loss1: 6.18, train loss2: 6.93e-01 - valid si-snr: -6.32e-01, valid loss1: -6.32e-01, valid loss2: 6.96e-01
      speechbrain.utils.checkpoints - Saved an end-of-epoch checkpoint in results/detection_demo22/1607/save/CKPT+2024-02-02+15-55-58+00
      speechbrain.utils.epoch_loop - Going into epoch 2
      100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:02<00:00,  5.72it/s, loss1=-2.26, loss2=0.693, train_loss=-2.26]
      100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.47it/s]
      speechbrain.utils.train_logger - epoch: 2, lr: 5.00e-04 - train si-snr: -2.26e+00, train loss1: -2.26e+00, train loss2: 6.93e-01 - valid si-snr: -2.13e+00, valid loss1: -2.13e+00, valid loss2: 6.97e-01
      speechbrain.utils.checkpoints - Saved an end-of-epoch checkpoint in results/detection_demo22/1607/save/CKPT+2024-02-02+15-56-01+00
      speechbrain.utils.checkpoints - Deleted checkpoint in results/detection_demo22/1607/save/CKPT+2024-02-02+15-55-58+00
      speechbrain.utils.epoch_loop - Going into epoch 3

==============================================
Parameter Estimation
==============================================

Firstly, activating ``waveform`` environment.
Then, downloading waveform data ``GW150914_downsampled_posterior_samples.dat`` from `Zenodo <https://zenodo.org/uploads/10608761>`_.
and put it under `dataset/pe/downsampled_posterior_samples_v1.0.0 <https://github.com/YueZhou-oh/GWDA_lib/tree/main/dataset/pe/downsampled_posterior_samples_v1.0.0>`_ folder.
Finally, by running `run.sh <https://github.com/YueZhou-oh/GWDA_lib/blob/main/src/model/nflow/run.sh>`_ script, your own parameter estimation model can be trained. 

.. code-block:: console
    :linenos:

    $ conda activate waveform
    $ cd cd /workspace/GWDA_lib/src/model/nflow/
    $ bash run.sh

The output log can be seen as follows.

.. code-block:: shell
    :linenos:

      Nestedspace(activation='elu', apply_unconditional_transform=False, base_transform_type='rq-coupling', basis_dir='../../../dataset/pe/GW150914_sample_prior_basis/', batch_norm=True, batch_size=2048, bw_dstar=None, cuda=True, data_dir='../../../dataset/pe/GW150914_sample_prior_basis/', detectors=None, distance_prior=None, distance_prior_fn='uniform_distance', dont_sample_extrinsic_only=False, dropout_probability=0.0, epochs=10000, flow_lr=None, hidden_dims=512, kl_annealing=True, lr=0.0001, lr_anneal_method='cosine', lr_annealing=True, mixed_alpha=0.0, mode='train', model_dir='models/GW170104_sample_uniform_100basis_all_uniform_prior/', model_source='new', model_type='nde', nbins=8, nflows=15, nsample=100000, nsamples_target_event=50000, num_transform_blocks=10, output_freq=10, sampling_from='uniform', save=True, save_aux_filename='waveforms_supplementary.hdf5', save_model_name='model.pt', snr_annealing=False, snr_threshold=None, steplr_gamma=0.5, steplr_step_size=80, tail_bound=1.0, transfer_epochs=0, truncate_basis=100)
      Waveform directory ../../../dataset/pe/GW150914_sample_prior_basis/
      Model directory models/GW170104_sample_uniform_100basis_all_uniform_prior/
      Device cuda
      Loading dataset
      Sampling 100000 sets of parameters from uniform prior.
      init training...
      init relative whitening...
      Truncating reduced basis from 600 to 100 elements.
      initialidze reduced basis aux...
      Building time translation matrices.
      100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1001/1001 [00:19<00:00, 52.63it/s]
      calculate threshold standardizatison...
      Generating 90000 detector waveforms
      Setting extrinsic parameters to fiducial values.
      100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 90000/90000 [05:46<00:00, 259.74it/s]
      Calculating new standardization factors.
      Loading load_all_bilby_samples...
      100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.03s/it]
      Loading load_all_event_strain...
      0%|                                                                                                                                                                                                            | 0/1 [00:00<?, ?it/s]init relative whitening...
      Building time translation matrices.
      100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1001/1001 [00:18<00:00, 53.28it/s]
      100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:19<00:00, 19.34s/it]
      Detectors: ['H1', 'L1']

      Constructing model of type nde

      Initial learning rate 0.0001
      Using cosine LR annealing.

      Model hyperparameters:
      input_dim        15
      num_flow_steps   15
      context_dim      400
      base_transform_kwargs
               hidden_dim      512
               num_transform_blocks    10
               activation      elu
               dropout_probability     0.0
               batch_norm      True
               num_bins        8
               tail_bound      1.0
               apply_unconditional_transform   False
               base_transform_type     rq-coupling

      Training for 10000 epochs
      Starting timer
      Learning rate: 0.0001
      Re-generating waveforms for uniform prior.
      Setting extrinsic parameters to fiducial values.
      Train Epoch: 1 [0/90000 (0%)]   Loss: 27.4751   Cost: 12.08s
      Train Epoch: 1 [20480/90000 (23%)]      Loss: 22.0006   Cost: 4.84s
      Train Epoch: 1 [40960/90000 (45%)]      Loss: 21.8823   Cost: 5.22s
      Train Epoch: 1 [61440/90000 (68%)]      Loss: 21.4447   Cost: 4.57s
      Train Epoch: 1 [81920/90000 (91%)]      Loss: 21.3464   Cost: 5.40s
      Train Epoch: 1  Average Loss: 22.2120

