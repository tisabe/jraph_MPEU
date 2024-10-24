"""Test suite for parsing profiling experiments."""
import csv
from pathlib import Path
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
print(os.getcwd())
from jraph_MPEU import parse_longer_experiments as ppe


sample_err_file = """
I0922 19:38:42.859733 22772591851328 train.py:77] LOG Message: Recompiling!
I0922 19:38:45.130296 22772591851328 train.py:77] LOG Message: Recompiling!
I0922 19:38:48.776723 22772591851328 train.py:77] LOG Message: Recompiling!
I0922 19:38:51.163419 22772591851328 train.py:77] LOG Message: Recompiling!
I0922 19:38:58.136900 22772591851328 train.py:77] LOG Message: Recompiling!
I0922 19:44:37.514216 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_0100000.pkl
I0922 19:44:37.549383 22772591851328 train.py:646] Step 100000 train loss: 0.020037677139043808
I0922 19:44:49.728827 22772591851328 train.py:367] RMSE/MAE train: [0.20275778 0.08972112]
I0922 19:44:49.729081 22772591851328 train.py:367] RMSE/MAE validation: [0.1509765  0.09208725]
I0922 19:44:49.729232 22772591851328 train.py:367] RMSE/MAE test: [0.15761761 0.09259332]
I0922 19:50:32.564531 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_0200000.pkl
I0922 19:50:32.598335 22772591851328 train.py:646] Step 200000 train loss: 0.008511696942150593
I0922 19:50:38.821388 22772591851328 train.py:367] RMSE/MAE train: [0.12920047 0.07171371]
I0922 19:50:38.821630 22772591851328 train.py:367] RMSE/MAE validation: [0.12058905 0.07399455]
I0922 19:50:38.821775 22772591851328 train.py:367] RMSE/MAE test: [0.12795346 0.07371643]
I0922 19:56:17.253071 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_0300000.pkl
I0922 19:56:17.289658 22772591851328 train.py:646] Step 300000 train loss: 0.013656564056873322
I0922 19:56:23.783148 22772591851328 train.py:367] RMSE/MAE train: [0.2079944  0.06098242]
I0922 19:56:23.783575 22772591851328 train.py:367] RMSE/MAE validation: [0.10287144 0.0636196 ]
I0922 19:56:23.783723 22772591851328 train.py:367] RMSE/MAE test: [0.11564735 0.06401897]
I0922 20:02:02.694253 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_0400000.pkl
I0922 20:02:02.729142 22772591851328 train.py:646] Step 400000 train loss: 0.003360104514285922
I0922 20:02:08.788712 22772591851328 train.py:367] RMSE/MAE train: [0.13138653 0.05367709]
I0922 20:02:08.788946 22772591851328 train.py:367] RMSE/MAE validation: [0.09845737 0.05725281]
I0922 20:02:08.789089 22772591851328 train.py:367] RMSE/MAE test: [0.1072979  0.05752484]
I0922 20:07:48.825165 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_0500000.pkl
I0922 20:07:48.860656 22772591851328 train.py:646] Step 500000 train loss: 0.0037836572155356407
I0922 20:07:55.016988 22772591851328 train.py:367] RMSE/MAE train: [0.09631481 0.05228592]
I0922 20:07:55.017230 22772591851328 train.py:367] RMSE/MAE validation: [0.09596273 0.05597962]
I0922 20:07:55.017373 22772591851328 train.py:367] RMSE/MAE test: [0.10519367 0.05631791]
I0922 20:13:34.807075 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_0600000.pkl
I0922 20:13:34.843732 22772591851328 train.py:646] Step 600000 train loss: 0.005303317215293646
I0922 20:13:40.927053 22772591851328 train.py:367] RMSE/MAE train: [0.10897066 0.04724551]
I0922 20:13:40.927317 22772591851328 train.py:367] RMSE/MAE validation: [0.09366015 0.05153499]
I0922 20:13:40.927462 22772591851328 train.py:367] RMSE/MAE test: [0.09955601 0.0521435 ]
I0922 20:19:21.567133 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_0700000.pkl
I0922 20:19:21.571768 22772591851328 train.py:646] Step 700000 train loss: 0.0024990160018205643
I0922 20:19:27.701822 22772591851328 train.py:367] RMSE/MAE train: [0.24602357 0.04577076]
I0922 20:19:27.702062 22772591851328 train.py:367] RMSE/MAE validation: [0.10015737 0.04990895]
I0922 20:19:27.702208 22772591851328 train.py:367] RMSE/MAE test: [0.09932088 0.05091943]
I0922 20:25:06.500418 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_0800000.pkl
I0922 20:25:06.536822 22772591851328 train.py:646] Step 800000 train loss: 0.003160941880196333
I0922 20:25:12.737468 22772591851328 train.py:367] RMSE/MAE train: [0.12060375 0.04212299]
I0922 20:25:12.737710 22772591851328 train.py:367] RMSE/MAE validation: [0.09219223 0.04659678]
I0922 20:25:12.737853 22772591851328 train.py:367] RMSE/MAE test: [0.09700008 0.04752998]
I0922 20:28:31.572394 22772591851328 train.py:77] LOG Message: Recompiling!
I0922 20:30:55.473704 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_0900000.pkl
I0922 20:30:55.478268 22772591851328 train.py:646] Step 900000 train loss: 0.0026200469583272934
I0922 20:31:01.863673 22772591851328 train.py:367] RMSE/MAE train: [0.14779199 0.0422107 ]
I0922 20:31:01.863908 22772591851328 train.py:367] RMSE/MAE validation: [0.09590055 0.0466175 ]
I0922 20:31:01.864052 22772591851328 train.py:367] RMSE/MAE test: [0.09708929 0.04771617]
I0922 20:36:38.992646 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_1000000.pkl
I0922 20:36:38.997025 22772591851328 train.py:646] Step 1000000 train loss: 0.004634609911590815
I0922 20:36:45.118834 22772591851328 train.py:367] RMSE/MAE train: [0.06904935 0.03926954]
I0922 20:36:45.119240 22772591851328 train.py:367] RMSE/MAE validation: [0.09040674 0.04481585]
I0922 20:36:45.119395 22772591851328 train.py:367] RMSE/MAE test: [0.09211571 0.04554995]
I0922 20:42:25.172928 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_1100000.pkl
I0922 20:42:25.177401 22772591851328 train.py:646] Step 1100000 train loss: 0.0028198808431625366
I0922 20:42:31.297907 22772591851328 train.py:367] RMSE/MAE train: [0.14949136 0.0463004 ]
I0922 20:42:31.298149 22772591851328 train.py:367] RMSE/MAE validation: [0.09058431 0.04979084]
I0922 20:42:31.298291 22772591851328 train.py:367] RMSE/MAE test: [0.102557   0.05175615]
I0922 20:48:12.978005 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_1200000.pkl
I0922 20:48:13.014300 22772591851328 train.py:646] Step 1200000 train loss: 0.004747063387185335
I0922 20:48:19.133619 22772591851328 train.py:367] RMSE/MAE train: [0.09153273 0.03562463]
I0922 20:48:19.133857 22772591851328 train.py:367] RMSE/MAE validation: [0.08139673 0.04091057]
I0922 20:48:19.134002 22772591851328 train.py:367] RMSE/MAE test: [0.09301611 0.04215152]
I0922 20:53:58.162662 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_1300000.pkl
I0922 20:53:58.199018 22772591851328 train.py:646] Step 1300000 train loss: 0.004920187871903181
I0922 20:54:04.260295 22772591851328 train.py:367] RMSE/MAE train: [0.08647772 0.03385804]
I0922 20:54:04.260532 22772591851328 train.py:367] RMSE/MAE validation: [0.08163706 0.03981383]
I0922 20:54:04.260676 22772591851328 train.py:367] RMSE/MAE test: [0.09263884 0.04036092]
I0922 20:59:41.995592 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_1400000.pkl
I0922 20:59:41.999883 22772591851328 train.py:646] Step 1400000 train loss: 0.002028561197221279
I0922 20:59:48.005961 22772591851328 train.py:367] RMSE/MAE train: [0.10341713 0.03342566]
I0922 20:59:48.006198 22772591851328 train.py:367] RMSE/MAE validation: [0.08039026 0.03904671]
I0922 20:59:48.006340 22772591851328 train.py:367] RMSE/MAE test: [0.09422753 0.04016866]
I0922 21:05:25.749386 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_1500000.pkl
I0922 21:05:25.782956 22772591851328 train.py:646] Step 1500000 train loss: 0.001935144537128508
I0922 21:05:31.841035 22772591851328 train.py:367] RMSE/MAE train: [0.0898233  0.03082593]
I0922 21:05:31.841274 22772591851328 train.py:367] RMSE/MAE validation: [0.08040138 0.03745404]
I0922 21:05:31.841418 22772591851328 train.py:367] RMSE/MAE test: [0.09101384 0.03778027]
I0922 21:11:09.256847 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_1600000.pkl
I0922 21:11:09.261456 22772591851328 train.py:646] Step 1600000 train loss: 0.013698373921215534
I0922 21:11:15.522304 22772591851328 train.py:367] RMSE/MAE train: [0.06472723 0.02980887]
I0922 21:11:15.522550 22772591851328 train.py:367] RMSE/MAE validation: [0.07890424 0.03675493]
I0922 21:11:15.522698 22772591851328 train.py:367] RMSE/MAE test: [0.09363729 0.03736945]
I0922 21:16:48.912192 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_1700000.pkl
I0922 21:16:48.916626 22772591851328 train.py:646] Step 1700000 train loss: 0.0019305669702589512
I0922 21:16:54.972350 22772591851328 train.py:367] RMSE/MAE train: [0.08795491 0.02890161]
I0922 21:16:54.972585 22772591851328 train.py:367] RMSE/MAE validation: [0.07891241 0.03599951]
I0922 21:16:54.972727 22772591851328 train.py:367] RMSE/MAE test: [0.09068796 0.03652508]
I0922 21:22:30.790741 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_1800000.pkl
I0922 21:22:30.795112 22772591851328 train.py:646] Step 1800000 train loss: 0.002203097566962242
I0922 21:22:36.705464 22772591851328 train.py:367] RMSE/MAE train: [0.06778511 0.02836205]
I0922 21:22:36.705696 22772591851328 train.py:367] RMSE/MAE validation: [0.07512736 0.03550514]
I0922 21:22:36.705840 22772591851328 train.py:367] RMSE/MAE test: [0.09027727 0.03594178]
I0922 21:28:10.303036 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_1900000.pkl
I0922 21:28:10.306377 22772591851328 train.py:646] Step 1900000 train loss: 0.004515858832746744
I0922 21:28:16.143673 22772591851328 train.py:367] RMSE/MAE train: [0.08369535 0.02662788]
I0922 21:28:16.143906 22772591851328 train.py:367] RMSE/MAE validation: [0.07553749 0.03409659]
I0922 21:28:16.144049 22772591851328 train.py:367] RMSE/MAE test: [0.08832357 0.03476478]
I0922 21:33:42.134622 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_2000000.pkl
I0922 21:33:42.137916 22772591851328 train.py:646] Step 2000000 train loss: 0.0008505643345415592
I0922 21:33:47.819770 22772591851328 train.py:367] RMSE/MAE train: [0.08793343 0.02943161]
I0922 21:33:47.820001 22772591851328 train.py:367] RMSE/MAE validation: [0.07778961 0.03666263]
I0922 21:33:47.820141 22772591851328 train.py:367] RMSE/MAE test: [0.08889424 0.03734236]
I0922 21:33:47.825908 22772591851328 train.py:674] Reached maximum number of steps without early stopping.
I0922 21:33:47.826324 22772591851328 train.py:681] Lowest validation loss: 0.07512736408546457
I0922 21:33:47.941115 22772591851328 train.py:684] Median batching time: 0.0006923675537109375
I0922 21:33:48.016448 22772591851328 train.py:687] Mean batching time: 0.0007143152430057525
I0922 21:33:48.124960 22772591851328 train.py:690] Median update time: 0.002786874771118164
I0922 21:33:48.200265 22772591851328 train.py:693] Mean update time: 0.002795692672967911
"""

sample_full_err_file = """
/u/dansp/.bashrc: line 52: /: Is a directory
2024-09-22 19:36:46.912216: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Error.  nthreads cannot be larger than environment variable "NUMEXPR_MAX_THREADS" (64)I0922 19:37:30.965628 22772591851328 main.py:51] JAX host: 0 / 1
I0922 19:37:30.965753 22772591851328 main.py:52] JAX local devices: [cuda(id=0), cuda(id=1), cuda(id=2), cuda(id=3)]
I0922 19:37:30.965973 22772591851328 train.py:553] Loading datasets.
I0922 19:37:30.966045 22772591851328 input_pipeline.py:581] Did not find split file at /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/splits.json. Pulling data.
I0922 19:37:35.266318 22772591851328 input_pipeline.py:275] Number of entries selected: 102569
I0922 19:38:34.715420 22772591851328 input_pipeline.py:653] Mean: -0.14989074089271043, Std: 0.8802420655757094
I0922 19:38:35.361307 22772591851328 train.py:555] Number of node classes: 74
I0922 19:38:35.361955 22772591851328 train.py:565] Because of compute device: gpu_a100 we set compile batching to: True
I0922 19:38:35.362034 22772591851328 train.py:569] Config dynamic batch: False
I0922 19:38:35.362076 22772591851328 train.py:570] Config dynamic batch is True?: False
I0922 19:38:35.746380 22772591851328 train.py:478] Initializing network.
I0922 19:38:38.357593 22772591851328 train.py:594] 177568 params, size: 0.71MB
I0922 19:38:38.374848 22772591851328 train.py:610] Starting training.
I0922 19:38:38.421775 22772591851328 train.py:77] LOG Message: Recompiling!
2024-09-22 19:38:41.140252: W external/xla/xla/service/gpu/buffer_comparator.cc:1054] INTERNAL: ptxas exited with non-zero error code 65280, output: ptxas /tmp/tempfile-ravg1170-528e6f61-1909-622b8be3ab01d, line 10; fatal   : Unsupported .version 7.8; current version is '7.4'
ptxas fatal   : Ptx assembly aborted due to errors

Relying on driver to perform ptx compilation. 
Setting XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda  or modifying $PATH can be used to set the location of ptxas
This message will only be logged once.
I0922 19:38:42.859733 22772591851328 train.py:77] LOG Message: Recompiling!
I0922 19:38:45.130296 22772591851328 train.py:77] LOG Message: Recompiling!
I0922 19:38:48.776723 22772591851328 train.py:77] LOG Message: Recompiling!
I0922 19:38:51.163419 22772591851328 train.py:77] LOG Message: Recompiling!
I0922 19:38:58.136900 22772591851328 train.py:77] LOG Message: Recompiling!
I0922 19:44:37.514216 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_0100000.pkl
I0922 19:44:37.549383 22772591851328 train.py:646] Step 100000 train loss: 0.020037677139043808
I0922 19:44:49.728827 22772591851328 train.py:367] RMSE/MAE train: [0.20275778 0.08972112]
I0922 19:44:49.729081 22772591851328 train.py:367] RMSE/MAE validation: [0.1509765  0.09208725]
I0922 19:44:49.729232 22772591851328 train.py:367] RMSE/MAE test: [0.15761761 0.09259332]
I0922 19:50:32.564531 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_0200000.pkl
I0922 19:50:32.598335 22772591851328 train.py:646] Step 200000 train loss: 0.008511696942150593
I0922 19:50:38.821388 22772591851328 train.py:367] RMSE/MAE train: [0.12920047 0.07171371]
I0922 19:50:38.821630 22772591851328 train.py:367] RMSE/MAE validation: [0.12058905 0.07399455]
I0922 19:50:38.821775 22772591851328 train.py:367] RMSE/MAE test: [0.12795346 0.07371643]
I0922 19:56:17.253071 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_0300000.pkl
I0922 19:56:17.289658 22772591851328 train.py:646] Step 300000 train loss: 0.013656564056873322
I0922 19:56:23.783148 22772591851328 train.py:367] RMSE/MAE train: [0.2079944  0.06098242]
I0922 19:56:23.783575 22772591851328 train.py:367] RMSE/MAE validation: [0.10287144 0.0636196 ]
I0922 19:56:23.783723 22772591851328 train.py:367] RMSE/MAE test: [0.11564735 0.06401897]
I0922 20:02:02.694253 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_0400000.pkl
I0922 20:02:02.729142 22772591851328 train.py:646] Step 400000 train loss: 0.003360104514285922
I0922 20:02:08.788712 22772591851328 train.py:367] RMSE/MAE train: [0.13138653 0.05367709]
I0922 20:02:08.788946 22772591851328 train.py:367] RMSE/MAE validation: [0.09845737 0.05725281]
I0922 20:02:08.789089 22772591851328 train.py:367] RMSE/MAE test: [0.1072979  0.05752484]
I0922 20:07:48.825165 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_0500000.pkl
I0922 20:07:48.860656 22772591851328 train.py:646] Step 500000 train loss: 0.0037836572155356407
I0922 20:07:55.016988 22772591851328 train.py:367] RMSE/MAE train: [0.09631481 0.05228592]
I0922 20:07:55.017230 22772591851328 train.py:367] RMSE/MAE validation: [0.09596273 0.05597962]
I0922 20:07:55.017373 22772591851328 train.py:367] RMSE/MAE test: [0.10519367 0.05631791]
I0922 20:13:34.807075 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_0600000.pkl
I0922 20:13:34.843732 22772591851328 train.py:646] Step 600000 train loss: 0.005303317215293646
I0922 20:13:40.927053 22772591851328 train.py:367] RMSE/MAE train: [0.10897066 0.04724551]
I0922 20:13:40.927317 22772591851328 train.py:367] RMSE/MAE validation: [0.09366015 0.05153499]
I0922 20:13:40.927462 22772591851328 train.py:367] RMSE/MAE test: [0.09955601 0.0521435 ]
I0922 20:19:21.567133 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_0700000.pkl
I0922 20:19:21.571768 22772591851328 train.py:646] Step 700000 train loss: 0.0024990160018205643
I0922 20:19:27.701822 22772591851328 train.py:367] RMSE/MAE train: [0.24602357 0.04577076]
I0922 20:19:27.702062 22772591851328 train.py:367] RMSE/MAE validation: [0.10015737 0.04990895]
I0922 20:19:27.702208 22772591851328 train.py:367] RMSE/MAE test: [0.09932088 0.05091943]
I0922 20:25:06.500418 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_0800000.pkl
I0922 20:25:06.536822 22772591851328 train.py:646] Step 800000 train loss: 0.003160941880196333
I0922 20:25:12.737468 22772591851328 train.py:367] RMSE/MAE train: [0.12060375 0.04212299]
I0922 20:25:12.737710 22772591851328 train.py:367] RMSE/MAE validation: [0.09219223 0.04659678]
I0922 20:25:12.737853 22772591851328 train.py:367] RMSE/MAE test: [0.09700008 0.04752998]
I0922 20:28:31.572394 22772591851328 train.py:77] LOG Message: Recompiling!
I0922 20:30:55.473704 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_0900000.pkl
I0922 20:30:55.478268 22772591851328 train.py:646] Step 900000 train loss: 0.0026200469583272934
I0922 20:31:01.863673 22772591851328 train.py:367] RMSE/MAE train: [0.14779199 0.0422107 ]
I0922 20:31:01.863908 22772591851328 train.py:367] RMSE/MAE validation: [0.09590055 0.0466175 ]
I0922 20:31:01.864052 22772591851328 train.py:367] RMSE/MAE test: [0.09708929 0.04771617]
I0922 20:36:38.992646 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_1000000.pkl
I0922 20:36:38.997025 22772591851328 train.py:646] Step 1000000 train loss: 0.004634609911590815
I0922 20:36:45.118834 22772591851328 train.py:367] RMSE/MAE train: [0.06904935 0.03926954]
I0922 20:36:45.119240 22772591851328 train.py:367] RMSE/MAE validation: [0.09040674 0.04481585]
I0922 20:36:45.119395 22772591851328 train.py:367] RMSE/MAE test: [0.09211571 0.04554995]
I0922 20:42:25.172928 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_1100000.pkl
I0922 20:42:25.177401 22772591851328 train.py:646] Step 1100000 train loss: 0.0028198808431625366
I0922 20:42:31.297907 22772591851328 train.py:367] RMSE/MAE train: [0.14949136 0.0463004 ]
I0922 20:42:31.298149 22772591851328 train.py:367] RMSE/MAE validation: [0.09058431 0.04979084]
I0922 20:42:31.298291 22772591851328 train.py:367] RMSE/MAE test: [0.102557   0.05175615]
I0922 20:48:12.978005 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_1200000.pkl
I0922 20:48:13.014300 22772591851328 train.py:646] Step 1200000 train loss: 0.004747063387185335
I0922 20:48:19.133619 22772591851328 train.py:367] RMSE/MAE train: [0.09153273 0.03562463]
I0922 20:48:19.133857 22772591851328 train.py:367] RMSE/MAE validation: [0.08139673 0.04091057]
I0922 20:48:19.134002 22772591851328 train.py:367] RMSE/MAE test: [0.09301611 0.04215152]
I0922 20:53:58.162662 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_1300000.pkl
I0922 20:53:58.199018 22772591851328 train.py:646] Step 1300000 train loss: 0.004920187871903181
I0922 20:54:04.260295 22772591851328 train.py:367] RMSE/MAE train: [0.08647772 0.03385804]
I0922 20:54:04.260532 22772591851328 train.py:367] RMSE/MAE validation: [0.08163706 0.03981383]
I0922 20:54:04.260676 22772591851328 train.py:367] RMSE/MAE test: [0.09263884 0.04036092]
I0922 20:59:41.995592 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_1400000.pkl
I0922 20:59:41.999883 22772591851328 train.py:646] Step 1400000 train loss: 0.002028561197221279
I0922 20:59:48.005961 22772591851328 train.py:367] RMSE/MAE train: [0.10341713 0.03342566]
I0922 20:59:48.006198 22772591851328 train.py:367] RMSE/MAE validation: [0.08039026 0.03904671]
I0922 20:59:48.006340 22772591851328 train.py:367] RMSE/MAE test: [0.09422753 0.04016866]
I0922 21:05:25.749386 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_1500000.pkl
I0922 21:05:25.782956 22772591851328 train.py:646] Step 1500000 train loss: 0.001935144537128508
I0922 21:05:31.841035 22772591851328 train.py:367] RMSE/MAE train: [0.0898233  0.03082593]
I0922 21:05:31.841274 22772591851328 train.py:367] RMSE/MAE validation: [0.08040138 0.03745404]
I0922 21:05:31.841418 22772591851328 train.py:367] RMSE/MAE test: [0.09101384 0.03778027]
I0922 21:11:09.256847 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_1600000.pkl
I0922 21:11:09.261456 22772591851328 train.py:646] Step 1600000 train loss: 0.013698373921215534
I0922 21:11:15.522304 22772591851328 train.py:367] RMSE/MAE train: [0.06472723 0.02980887]
I0922 21:11:15.522550 22772591851328 train.py:367] RMSE/MAE validation: [0.07890424 0.03675493]
I0922 21:11:15.522698 22772591851328 train.py:367] RMSE/MAE test: [0.09363729 0.03736945]
I0922 21:16:48.912192 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_1700000.pkl
I0922 21:16:48.916626 22772591851328 train.py:646] Step 1700000 train loss: 0.0019305669702589512
I0922 21:16:54.972350 22772591851328 train.py:367] RMSE/MAE train: [0.08795491 0.02890161]
I0922 21:16:54.972585 22772591851328 train.py:367] RMSE/MAE validation: [0.07891241 0.03599951]
I0922 21:16:54.972727 22772591851328 train.py:367] RMSE/MAE test: [0.09068796 0.03652508]
I0922 21:22:30.790741 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_1800000.pkl
I0922 21:22:30.795112 22772591851328 train.py:646] Step 1800000 train loss: 0.002203097566962242
I0922 21:22:36.705464 22772591851328 train.py:367] RMSE/MAE train: [0.06778511 0.02836205]
I0922 21:22:36.705696 22772591851328 train.py:367] RMSE/MAE validation: [0.07512736 0.03550514]
I0922 21:22:36.705840 22772591851328 train.py:367] RMSE/MAE test: [0.09027727 0.03594178]
I0922 21:28:10.303036 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_1900000.pkl
I0922 21:28:10.306377 22772591851328 train.py:646] Step 1900000 train loss: 0.004515858832746744
I0922 21:28:16.143673 22772591851328 train.py:367] RMSE/MAE train: [0.08369535 0.02662788]
I0922 21:28:16.143906 22772591851328 train.py:367] RMSE/MAE validation: [0.07553749 0.03409659]
I0922 21:28:16.144049 22772591851328 train.py:367] RMSE/MAE test: [0.08832357 0.03476478]
I0922 21:33:42.134622 22772591851328 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/aflow/static/round_False/32/gpu_a100/iteration_0/checkpoints/checkpoint_2000000.pkl
I0922 21:33:42.137916 22772591851328 train.py:646] Step 2000000 train loss: 0.0008505643345415592
I0922 21:33:47.819770 22772591851328 train.py:367] RMSE/MAE train: [0.08793343 0.02943161]
I0922 21:33:47.820001 22772591851328 train.py:367] RMSE/MAE validation: [0.07778961 0.03666263]
I0922 21:33:47.820141 22772591851328 train.py:367] RMSE/MAE test: [0.08889424 0.03734236]
I0922 21:33:47.825908 22772591851328 train.py:674] Reached maximum number of steps without early stopping.
I0922 21:33:47.826324 22772591851328 train.py:681] Lowest validation loss: 0.07512736408546457
I0922 21:33:47.941115 22772591851328 train.py:684] Median batching time: 0.0006923675537109375
I0922 21:33:48.016448 22772591851328 train.py:687] Mean batching time: 0.0007143152430057525
I0922 21:33:48.124960 22772591851328 train.py:690] Median update time: 0.002786874771118164
I0922 21:33:48.200265 22772591851328 train.py:693] Mean update time: 0.002795692672967911
"""

profiling_err_file_failing = """
/u/dansp/.bashrc: line 52: /: Is a directory
2024-09-24 17:38:58.470240: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Error.  nthreads cannot be larger than environment variable "NUMEXPR_MAX_THREADS" (64)I0924 17:39:03.870156 22758399235904 main.py:51] JAX host: 0 / 1
I0924 17:39:03.870271 22758399235904 main.py:52] JAX local devices: [cuda(id=0), cuda(id=1), cuda(id=2), cuda(id=3)]
I0924 17:39:03.870506 22758399235904 train.py:553] Loading datasets.
I0924 17:39:03.870580 22758399235904 input_pipeline.py:581] Did not find split file at /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/schnet/aflow/static/round_False/64/gpu_a100/iteration_9/splits.json. Pulling data.
I0924 17:39:03.877534 22758399235904 input_pipeline.py:275] Number of entries selected: 102569
I0924 17:39:58.410973 22758399235904 input_pipeline.py:653] Mean: -0.14989074089271043, Std: 0.8802420655757094
I0924 17:39:59.046295 22758399235904 train.py:555] Number of node classes: 74
I0924 17:39:59.047116 22758399235904 train.py:565] Because of compute device: gpu_a100 we set compile batching to: True
I0924 17:39:59.047192 22758399235904 train.py:569] Config dynamic batch: False
I0924 17:39:59.047235 22758399235904 train.py:570] Config dynamic batch is True?: False
I0924 17:39:59.225993 22758399235904 train.py:478] Initializing network.
I0924 17:40:01.255933 22758399235904 train.py:594] 84768 params, size: 0.34MB
I0924 17:40:01.256367 22758399235904 train.py:610] Starting training.
I0924 17:40:01.259335 22758399235904 train.py:77] LOG Message: Recompiling!
2024-09-24 17:40:03.484802: W external/xla/xla/service/gpu/buffer_comparator.cc:1054] INTERNAL: ptxas exited with non-zero error code 65280, output: ptxas /tmp/tempfile-ravg1034-ebe0bf0a-120911-622df51aaa28d, line 10; fatal   : Unsupported .version 7.8; current version is '7.4'
ptxas fatal   : Ptx assembly aborted due to errors

Relying on driver to perform ptx compilation. 
Setting XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda  or modifying $PATH can be used to set the location of ptxas
This message will only be logged once.
I0924 17:40:04.901338 22758399235904 train.py:77] LOG Message: Recompiling!
I0924 17:40:07.994270 22758399235904 train.py:77] LOG Message: Recompiling!
I0924 17:40:10.220393 22758399235904 train.py:77] LOG Message: Recompiling!
I0924 17:40:17.344727 22758399235904 train.py:77] LOG Message: Recompiling!
I0924 17:45:44.525483 22758399235904 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/schnet/aflow/static/round_False/64/gpu_a100/iteration_9/checkpoints/checkpoint_0100000.pkl
I0924 17:45:44.529567 22758399235904 train.py:646] Step 100000 train loss: 0.05842500552535057
I0924 17:45:52.163456 22758399235904 train.py:367] RMSE/MAE train: [0.21237597 0.11413605]
I0924 17:45:52.163751 22758399235904 train.py:367] RMSE/MAE validation: [0.22647893 0.11766672]
I0924 17:45:52.163903 22758399235904 train.py:367] RMSE/MAE test: [0.21606945 0.11935771]
I0924 17:51:21.042421 22758399235904 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/schnet/aflow/static/round_False/64/gpu_a100/iteration_9/checkpoints/checkpoint_0200000.pkl
I0924 17:51:21.045928 22758399235904 train.py:646] Step 200000 train loss: 0.02536071091890335
I0924 17:51:24.830087 22758399235904 train.py:367] RMSE/MAE train: [0.15101543 0.09066857]
I0924 17:51:24.830338 22758399235904 train.py:367] RMSE/MAE validation: [0.16744364 0.09691827]
I0924 17:51:24.830483 22758399235904 train.py:367] RMSE/MAE test: [0.33255186 0.10088759]
I0924 17:56:50.923702 22758399235904 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/schnet/aflow/static/round_False/64/gpu_a100/iteration_9/checkpoints/checkpoint_0300000.pkl
I0924 17:56:50.927639 22758399235904 train.py:646] Step 300000 train loss: 0.006440181750804186
I0924 17:56:54.840701 22758399235904 train.py:367] RMSE/MAE train: [0.16949118 0.08516319]
I0924 17:56:54.840936 22758399235904 train.py:367] RMSE/MAE validation: [0.15188606 0.08946227]
I0924 17:56:54.841089 22758399235904 train.py:367] RMSE/MAE test: [0.18006016 0.09206266]
I0924 18:02:21.583600 22758399235904 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/schnet/aflow/static/round_False/64/gpu_a100/iteration_9/checkpoints/checkpoint_0400000.pkl
I0924 18:02:21.605945 22758399235904 train.py:646] Step 400000 train loss: 0.027614042162895203
I0924 18:02:25.260513 22758399235904 train.py:367] RMSE/MAE train: [0.12943486 0.07110915]
I0924 18:02:25.260749 22758399235904 train.py:367] RMSE/MAE validation: [0.13939094 0.07732406]
I0924 18:02:25.260893 22758399235904 train.py:367] RMSE/MAE test: [0.1949203  0.07983868]
I0924 18:07:51.351306 22758399235904 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/schnet/aflow/static/round_False/64/gpu_a100/iteration_9/checkpoints/checkpoint_0500000.pkl
I0924 18:07:51.354978 22758399235904 train.py:646] Step 500000 train loss: 0.018727794289588928
I0924 18:07:54.990169 22758399235904 train.py:367] RMSE/MAE train: [0.12237335 0.06590556]
I0924 18:07:54.990408 22758399235904 train.py:367] RMSE/MAE validation: [0.13804108 0.07311054]
I0924 18:07:54.990552 22758399235904 train.py:367] RMSE/MAE test: [0.21854894 0.07517116]
I0924 18:13:22.557905 22758399235904 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/schnet/aflow/static/round_False/64/gpu_a100/iteration_9/checkpoints/checkpoint_0600000.pkl
I0924 18:13:22.561559 22758399235904 train.py:646] Step 600000 train loss: 0.01701252907514572
I0924 18:13:26.285543 22758399235904 train.py:367] RMSE/MAE train: [0.11190766 0.06087917]
I0924 18:13:26.285787 22758399235904 train.py:367] RMSE/MAE validation: [0.13212401 0.06857336]
I0924 18:13:26.285960 22758399235904 train.py:367] RMSE/MAE test: [0.20018461 0.07099614]
I0924 18:18:53.966493 22758399235904 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/schnet/aflow/static/round_False/64/gpu_a100/iteration_9/checkpoints/checkpoint_0700000.pkl
I0924 18:18:53.969971 22758399235904 train.py:646] Step 700000 train loss: 0.006466029677540064
I0924 18:18:57.694610 22758399235904 train.py:367] RMSE/MAE train: [0.10547164 0.05812341]
I0924 18:18:57.694849 22758399235904 train.py:367] RMSE/MAE validation: [0.1321057  0.06595314]
I0924 18:18:57.694989 22758399235904 train.py:367] RMSE/MAE test: [0.14463823 0.06757485]
I0924 18:24:24.064136 22758399235904 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/schnet/aflow/static/round_False/64/gpu_a100/iteration_9/checkpoints/checkpoint_0800000.pkl
I0924 18:24:24.067719 22758399235904 train.py:646] Step 800000 train loss: 0.0068345810286700726
I0924 18:24:27.735902 22758399235904 train.py:367] RMSE/MAE train: [0.09990552 0.05348239]
I0924 18:24:27.736142 22758399235904 train.py:367] RMSE/MAE validation: [0.11473587 0.06137076]
I0924 18:24:27.736289 22758399235904 train.py:367] RMSE/MAE test: [0.13562776 0.06293421]
I0924 18:29:54.067908 22758399235904 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/schnet/aflow/static/round_False/64/gpu_a100/iteration_9/checkpoints/checkpoint_0900000.pkl
I0924 18:29:54.071306 22758399235904 train.py:646] Step 900000 train loss: 0.0046090479008853436
I0924 18:29:57.810379 22758399235904 train.py:367] RMSE/MAE train: [0.09801675 0.05205359]
I0924 18:29:57.810614 22758399235904 train.py:367] RMSE/MAE validation: [0.1093366  0.05991875]
I0924 18:29:57.810757 22758399235904 train.py:367] RMSE/MAE test: [0.13568153 0.0614305 ]
I0924 18:35:26.976723 22758399235904 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/schnet/aflow/static/round_False/64/gpu_a100/iteration_9/checkpoints/checkpoint_1000000.pkl
I0924 18:35:27.066455 22758399235904 train.py:646] Step 1000000 train loss: 0.0038400664925575256
I0924 18:35:30.720299 22758399235904 train.py:367] RMSE/MAE train: [0.08746072 0.04847789]
I0924 18:35:30.720542 22758399235904 train.py:367] RMSE/MAE validation: [0.10852153 0.05651071]
I0924 18:35:30.720689 22758399235904 train.py:367] RMSE/MAE test: [0.13380356 0.05836148]
I0924 18:40:58.702487 22758399235904 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/schnet/aflow/static/round_False/64/gpu_a100/iteration_9/checkpoints/checkpoint_1100000.pkl
I0924 18:40:58.705847 22758399235904 train.py:646] Step 1100000 train loss: 0.009619319811463356
I0924 18:41:02.341280 22758399235904 train.py:367] RMSE/MAE train: [0.08076213 0.0458471 ]
I0924 18:41:02.341521 22758399235904 train.py:367] RMSE/MAE validation: [0.10369791 0.05425759]
I0924 18:41:02.341665 22758399235904 train.py:367] RMSE/MAE test: [0.13687102 0.05640327]
I0924 18:46:30.874583 22758399235904 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/schnet/aflow/static/round_False/64/gpu_a100/iteration_9/checkpoints/checkpoint_1200000.pkl
I0924 18:46:30.878109 22758399235904 train.py:646] Step 1200000 train loss: 0.005527402739971876
I0924 18:46:34.515245 22758399235904 train.py:367] RMSE/MAE train: [0.08336239 0.04967458]
I0924 18:46:34.515486 22758399235904 train.py:367] RMSE/MAE validation: [0.10784623 0.05842353]
I0924 18:46:34.515659 22758399235904 train.py:367] RMSE/MAE test: [0.13934406 0.0605933 ]
I0924 18:52:02.985765 22758399235904 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/schnet/aflow/static/round_False/64/gpu_a100/iteration_9/checkpoints/checkpoint_1300000.pkl
I0924 18:52:02.989212 22758399235904 train.py:646] Step 1300000 train loss: 0.004508291371166706
I0924 18:52:06.696719 22758399235904 train.py:367] RMSE/MAE train: [0.08342784 0.04415218]
I0924 18:52:06.696966 22758399235904 train.py:367] RMSE/MAE validation: [0.10698289 0.05290947]
I0924 18:52:06.697119 22758399235904 train.py:367] RMSE/MAE test: [0.13458772 0.05463742]
I0924 18:52:26.834381 22758399235904 train.py:77] LOG Message: Recompiling!
I0924 18:57:39.671085 22758399235904 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/schnet/aflow/static/round_False/64/gpu_a100/iteration_9/checkpoints/checkpoint_1400000.pkl
I0924 18:57:39.674584 22758399235904 train.py:646] Step 1400000 train loss: 0.004646456800401211
I0924 18:57:43.295721 22758399235904 train.py:367] RMSE/MAE train: [0.06975051 0.04157296]
I0924 18:57:43.296143 22758399235904 train.py:367] RMSE/MAE validation: [0.10538365 0.05096531]
I0924 18:57:43.296293 22758399235904 train.py:367] RMSE/MAE test: [0.13490838 0.05292116]
I0924 19:03:11.963994 22758399235904 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/schnet/aflow/static/round_False/64/gpu_a100/iteration_9/checkpoints/checkpoint_1500000.pkl
I0924 19:03:11.967476 22758399235904 train.py:646] Step 1500000 train loss: 0.004669209476560354
I0924 19:03:15.659356 22758399235904 train.py:367] RMSE/MAE train: [0.07187499 0.04126127]
I0924 19:03:15.659620 22758399235904 train.py:367] RMSE/MAE validation: [0.10997754 0.05128954]
I0924 19:03:15.659773 22758399235904 train.py:367] RMSE/MAE test: [0.13283571 0.05235443]
I0924 19:08:45.293798 22758399235904 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/schnet/aflow/static/round_False/64/gpu_a100/iteration_9/checkpoints/checkpoint_1600000.pkl
I0924 19:08:45.297337 22758399235904 train.py:646] Step 1600000 train loss: 0.005343337077647448
I0924 19:08:49.033897 22758399235904 train.py:367] RMSE/MAE train: [0.073963   0.04112255]
I0924 19:08:49.034139 22758399235904 train.py:367] RMSE/MAE validation: [0.10556793 0.05092652]
I0924 19:08:49.034292 22758399235904 train.py:367] RMSE/MAE test: [0.1308994  0.05230932]
I0924 19:14:17.661101 22758399235904 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/schnet/aflow/static/round_False/64/gpu_a100/iteration_9/checkpoints/checkpoint_1700000.pkl
I0924 19:14:17.664489 22758399235904 train.py:646] Step 1700000 train loss: 0.0027835459914058447
I0924 19:14:21.692784 22758399235904 train.py:367] RMSE/MAE train: [0.07081964 0.03969293]
I0924 19:14:21.693022 22758399235904 train.py:367] RMSE/MAE validation: [0.10531375 0.04953643]
I0924 19:14:21.693166 22758399235904 train.py:367] RMSE/MAE test: [0.13210397 0.05102902]
I0924 19:19:50.310861 22758399235904 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/schnet/aflow/static/round_False/64/gpu_a100/iteration_9/checkpoints/checkpoint_1800000.pkl
I0924 19:19:50.314261 22758399235904 train.py:646] Step 1800000 train loss: 0.003999321721494198
I0924 19:19:54.007967 22758399235904 train.py:367] RMSE/MAE train: [0.07038012 0.04060399]
I0924 19:19:54.008211 22758399235904 train.py:367] RMSE/MAE validation: [0.1081848  0.05057584]
I0924 19:19:54.008361 22758399235904 train.py:367] RMSE/MAE test: [0.13369079 0.05195197]
I0924 19:25:21.050896 22758399235904 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/schnet/aflow/static/round_False/64/gpu_a100/iteration_9/checkpoints/checkpoint_1900000.pkl
I0924 19:25:21.054158 22758399235904 train.py:646] Step 1900000 train loss: 0.004979283548891544
I0924 19:25:24.639035 22758399235904 train.py:367] RMSE/MAE train: [0.06350924 0.03861469]
I0924 19:25:24.639283 22758399235904 train.py:367] RMSE/MAE validation: [0.11398429 0.04950108]
I0924 19:25:24.639428 22758399235904 train.py:367] RMSE/MAE test: [0.13860971 0.05027738]
I0924 19:30:51.229342 22758399235904 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/schnet/aflow/static/round_False/64/gpu_a100/iteration_9/checkpoints/checkpoint_2000000.pkl
I0924 19:30:51.232682 22758399235904 train.py:646] Step 2000000 train loss: 0.0026564206928014755
I0924 19:30:54.844288 22758399235904 train.py:367] RMSE/MAE train: [0.06234671 0.03682689]
I0924 19:30:54.844531 22758399235904 train.py:367] RMSE/MAE validation: [0.11384458 0.04758696]
I0924 19:30:54.844682 22758399235904 train.py:367] RMSE/MAE test: [0.13701246 0.04903284]
I0924 19:30:54.851101 22758399235904 train.py:674] Reached maximum number of steps without early stopping.
I0924 19:30:54.851681 22758399235904 train.py:681] Lowest validation loss: 0.10369790991891552
I0924 19:30:54.977513 22758399235904 train.py:684] Median batching time: 0.001110076904296875
I0924 19:30:55.052170 22758399235904 train.py:687] Mean batching time: 0.0011481321103572846
I0924 19:30:55.162486 22758399235904 train.py:690] Median update time: 0.0026547908782958984
I0924 19:30:55.237533 22758399235904 train.py:693] Mean update time: 0.0026973051978349686
"""

convert_string_to_float_file = """
2024-09-21 07:22:08.597080: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Error.  nthreads cannot be larger than environment variable "NUMEXPR_MAX_THREADS" (64)I0921 07:22:30.495136 22514370438976 main.py:51] JAX host: 0 / 1
I0921 07:22:30.495256 22514370438976 main.py:52] JAX local devices: [cuda(id=0), cuda(id=1), cuda(id=2), cuda(id=3)]
I0921 07:22:30.495486 22514370438976 train.py:553] Loading datasets.
I0921 07:22:30.495563 22514370438976 input_pipeline.py:581] Did not find split file at /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/qm9/dynamic/round_False/32/gpu_a100/iteration_7/splits.json. Pulling data.
I0921 07:22:31.487912 22514370438976 input_pipeline.py:275] Number of entries selected: 133885
I0921 07:23:44.065457 22514370438976 input_pipeline.py:653] Mean: -75.9140390541746, Std: 10.36986183441548
I0921 07:23:45.060635 22514370438976 train.py:555] Number of node classes: 5
I0921 07:23:45.061464 22514370438976 train.py:565] Because of compute device: gpu_a100 we set compile batching to: True
I0921 07:23:45.061539 22514370438976 train.py:569] Config dynamic batch: True
I0921 07:23:45.061582 22514370438976 train.py:570] Config dynamic batch is True?: True
I0921 07:23:45.266489 22514370438976 train.py:478] Initializing network.
I0921 07:23:47.659484 22514370438976 train.py:594] 173152 params, size: 0.69MB
I0921 07:23:47.659948 22514370438976 train.py:610] Starting training.
I0921 07:23:47.663232 22514370438976 train.py:77] LOG Message: Recompiling!
2024-09-21 07:23:51.420173: W external/xla/xla/service/gpu/buffer_comparator.cc:1054] INTERNAL: ptxas exited with non-zero error code 65280, output: ptxas /tmp/tempfile-ravg1120-9057b1d2-28532-6229a5c6f9468, line 10; fatal   : Unsupported .version 7.8; current version is '7.4'
ptxas fatal   : Ptx assembly aborted due to errors

Relying on driver to perform ptx compilation. 
Setting XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda  or modifying $PATH can be used to set the location of ptxas
This message will only be logged once.
I0921 07:30:01.609405 22514370438976 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/qm9/dynamic/round_False/32/gpu_a100/iteration_7/checkpoints/checkpoint_0100000.pkl
I0921 07:30:01.615419 22514370438976 train.py:646] Step 100000 train loss: 0.0025825551711022854
I0921 07:30:14.274028 22514370438976 train.py:367] RMSE/MAE train: [0.63762646 0.56198494]
I0921 07:30:14.274304 22514370438976 train.py:367] RMSE/MAE validation: [0.64053979 0.56614005]
I0921 07:30:14.274457 22514370438976 train.py:367] RMSE/MAE test: [0.63644425 0.56342979]
I0921 07:36:23.180192 22514370438976 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/qm9/dynamic/round_False/32/gpu_a100/iteration_7/checkpoints/checkpoint_0200000.pkl
I0921 07:36:23.185059 22514370438976 train.py:646] Step 200000 train loss: 0.0012534132692962885
I0921 07:36:33.223848 22514370438976 train.py:367] RMSE/MAE train: [0.22917887 0.16128595]
I0921 07:36:33.224080 22514370438976 train.py:367] RMSE/MAE validation: [0.23004881 0.15912314]
I0921 07:36:33.224224 22514370438976 train.py:367] RMSE/MAE test: [0.2250047 0.1604264]
I0921 07:42:43.112245 22514370438976 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/qm9/dynamic/round_False/32/gpu_a100/iteration_7/checkpoints/checkpoint_0300000.pkl
I0921 07:42:43.116894 22514370438976 train.py:646] Step 300000 train loss: 0.0003567503299564123
I0921 07:42:53.181013 22514370438976 train.py:367] RMSE/MAE train: [0.1812213  0.14005429]
I0921 07:42:53.181252 22514370438976 train.py:367] RMSE/MAE validation: [0.17847037 0.13838503]
I0921 07:42:53.181404 22514370438976 train.py:367] RMSE/MAE test: [0.18507901 0.14135047]
I0921 07:49:02.103027 22514370438976 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/qm9/dynamic/round_False/32/gpu_a100/iteration_7/checkpoints/checkpoint_0400000.pkl
I0921 07:49:02.107646 22514370438976 train.py:646] Step 400000 train loss: 0.00015636355965398252
I0921 07:49:11.819210 22514370438976 train.py:367] RMSE/MAE train: [0.1243     0.09017694]
I0921 07:49:11.819446 22514370438976 train.py:367] RMSE/MAE validation: [0.12357951 0.09069529]
I0921 07:49:11.819596 22514370438976 train.py:367] RMSE/MAE test: [0.12825794 0.09103447]
I0921 07:55:20.288949 22514370438976 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/qm9/dynamic/round_False/32/gpu_a100/iteration_7/checkpoints/checkpoint_0500000.pkl
I0921 07:55:20.293587 22514370438976 train.py:646] Step 500000 train loss: 7.989697769517079e-05
I0921 07:55:30.067422 22514370438976 train.py:367] RMSE/MAE train: [0.11459652 0.08087971]
I0921 07:55:30.067661 22514370438976 train.py:367] RMSE/MAE validation: [0.11934893 0.08216458]
I0921 07:55:30.067810 22514370438976 train.py:367] RMSE/MAE test: [0.11386451 0.08208607]
I0921 08:01:39.139464 22514370438976 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/qm9/dynamic/round_False/32/gpu_a100/iteration_7/checkpoints/checkpoint_0600000.pkl
I0921 08:01:39.144009 22514370438976 train.py:646] Step 600000 train loss: 0.0001237685646628961
I0921 08:01:48.810663 22514370438976 train.py:367] RMSE/MAE train: [0.15309076 0.12301692]
I0921 08:01:48.810899 22514370438976 train.py:367] RMSE/MAE validation: [0.15296898 0.12343316]
I0921 08:01:48.811051 22514370438976 train.py:367] RMSE/MAE test: [0.15476382 0.12413499]
I0921 08:07:56.416045 22514370438976 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/qm9/dynamic/round_False/32/gpu_a100/iteration_7/checkpoints/checkpoint_0700000.pkl
I0921 08:07:56.420655 22514370438976 train.py:646] Step 700000 train loss: 0.0001710331125650555
I0921 08:08:06.375671 22514370438976 train.py:367] RMSE/MAE train: [0.09724305 0.07112052]
I0921 08:08:06.375907 22514370438976 train.py:367] RMSE/MAE validation: [0.10051427 0.07216019]
I0921 08:08:06.376056 22514370438976 train.py:367] RMSE/MAE test: [0.10070716 0.07242802]
I0921 08:14:14.920322 22514370438976 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/qm9/dynamic/round_False/32/gpu_a100/iteration_7/checkpoints/checkpoint_0800000.pkl
I0921 08:14:14.924881 22514370438976 train.py:646] Step 800000 train loss: 9.332817717222497e-05
I0921 08:14:24.628510 22514370438976 train.py:367] RMSE/MAE train: [0.10526463 0.07417917]
I0921 08:14:24.628751 22514370438976 train.py:367] RMSE/MAE validation: [0.09714349 0.07477885]
I0921 08:14:24.628902 22514370438976 train.py:367] RMSE/MAE test: [0.10362583 0.0759442 ]
I0921 08:20:33.078607 22514370438976 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/qm9/dynamic/round_False/32/gpu_a100/iteration_7/checkpoints/checkpoint_0900000.pkl
I0921 08:20:33.083292 22514370438976 train.py:646] Step 900000 train loss: 4.312652163207531e-05
I0921 08:20:42.788543 22514370438976 train.py:367] RMSE/MAE train: [0.08619641 0.06452338]
I0921 08:20:42.788779 22514370438976 train.py:367] RMSE/MAE validation: [0.09631777 0.06615742]
I0921 08:20:42.788928 22514370438976 train.py:367] RMSE/MAE test: [0.09176402 0.06596294]
I0921 08:26:49.325927 22514370438976 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/qm9/dynamic/round_False/32/gpu_a100/iteration_7/checkpoints/checkpoint_1000000.pkl
I0921 08:26:49.330620 22514370438976 train.py:646] Step 1000000 train loss: 4.207703022984788e-05
I0921 08:26:58.952258 22514370438976 train.py:367] RMSE/MAE train: [0.06884504 0.04885343]
I0921 08:26:58.952494 22514370438976 train.py:367] RMSE/MAE validation: [0.09523428 0.05094998]
I0921 08:26:58.952640 22514370438976 train.py:367] RMSE/MAE test: [0.07267908 0.05076584]
I0921 08:33:05.361518 22514370438976 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/qm9/dynamic/round_False/32/gpu_a100/iteration_7/checkpoints/checkpoint_1100000.pkl
I0921 08:33:05.366066 22514370438976 train.py:646] Step 1100000 train loss: 4.020088817924261e-05
I0921 08:33:14.886878 22514370438976 train.py:367] RMSE/MAE train: [0.07085029 0.05352306]
I0921 08:33:14.887110 22514370438976 train.py:367] RMSE/MAE validation: [0.0800021  0.05494436]
I0921 08:33:14.887259 22514370438976 train.py:367] RMSE/MAE test: [0.0773113  0.05588198]
I0921 08:39:20.505896 22514370438976 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/qm9/dynamic/round_False/32/gpu_a100/iteration_7/checkpoints/checkpoint_1200000.pkl
I0921 08:39:20.510442 22514370438976 train.py:646] Step 1200000 train loss: 2.5999226636486128e-05
I0921 08:39:30.311761 22514370438976 train.py:367] RMSE/MAE train: [0.05222315 0.03802177]
I0921 08:39:30.311997 22514370438976 train.py:367] RMSE/MAE validation: [0.06575698 0.0404972 ]
I0921 08:39:30.312145 22514370438976 train.py:367] RMSE/MAE test: [0.06023241 0.04034673]
I0921 08:45:35.474556 22514370438976 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/qm9/dynamic/round_False/32/gpu_a100/iteration_7/checkpoints/checkpoint_1300000.pkl
I0921 08:45:35.478927 22514370438976 train.py:646] Step 1300000 train loss: 1.3793463949696161e-05
I0921 08:45:45.014043 22514370438976 train.py:367] RMSE/MAE train: [0.08595129 0.07485353]
I0921 08:45:45.014283 22514370438976 train.py:367] RMSE/MAE validation: [0.09203123 0.07639384]
I0921 08:45:45.014434 22514370438976 train.py:367] RMSE/MAE test: [0.09162185 0.07642386]
I0921 08:51:49.250559 22514370438976 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/qm9/dynamic/round_False/32/gpu_a100/iteration_7/checkpoints/checkpoint_1400000.pkl
I0921 08:51:49.254976 22514370438976 train.py:646] Step 1400000 train loss: 2.812404818541836e-05
I0921 08:51:58.783100 22514370438976 train.py:367] RMSE/MAE train: [0.04826016 0.03565576]
I0921 08:51:58.783335 22514370438976 train.py:367] RMSE/MAE validation: [0.06070637 0.0378385 ]
I0921 08:51:58.783484 22514370438976 train.py:367] RMSE/MAE test: [0.05729091 0.03781344]
I0921 08:58:04.354798 22514370438976 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/qm9/dynamic/round_False/32/gpu_a100/iteration_7/checkpoints/checkpoint_1500000.pkl
I0921 08:58:04.359186 22514370438976 train.py:646] Step 1500000 train loss: 2.2484884539153427e-05
I0921 08:58:13.871082 22514370438976 train.py:367] RMSE/MAE train: [0.05039525 0.03880818]
I0921 08:58:13.871315 22514370438976 train.py:367] RMSE/MAE validation: [0.06177463 0.0411522 ]
I0921 08:58:13.871464 22514370438976 train.py:367] RMSE/MAE test: [0.05942395 0.04133885]
I0921 09:04:18.085724 22514370438976 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/qm9/dynamic/round_False/32/gpu_a100/iteration_7/checkpoints/checkpoint_1600000.pkl
I0921 09:04:18.090103 22514370438976 train.py:646] Step 1600000 train loss: 3.5359073081053793e-05
I0921 09:04:28.064980 22514370438976 train.py:367] RMSE/MAE train: [0.06367842 0.05231116]
I0921 09:04:28.065217 22514370438976 train.py:367] RMSE/MAE validation: [0.0725041  0.05510169]
I0921 09:04:28.065369 22514370438976 train.py:367] RMSE/MAE test: [0.07272359 0.05505082]
I0921 09:10:33.489484 22514370438976 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/qm9/dynamic/round_False/32/gpu_a100/iteration_7/checkpoints/checkpoint_1700000.pkl
I0921 09:10:33.493985 22514370438976 train.py:646] Step 1700000 train loss: 1.8517826902098022e-05
I0921 09:10:43.085015 22514370438976 train.py:367] RMSE/MAE train: [0.04856026 0.03693727]
I0921 09:10:43.085393 22514370438976 train.py:367] RMSE/MAE validation: [0.05777337 0.03919467]
I0921 09:10:43.085545 22514370438976 train.py:367] RMSE/MAE test: [0.06161602 0.03969593]
I0921 09:16:49.385122 22514370438976 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/qm9/dynamic/round_False/32/gpu_a100/iteration_7/checkpoints/checkpoint_1800000.pkl
I0921 09:16:49.389605 22514370438976 train.py:646] Step 1800000 train loss: 1.81268151209224e-05
I0921 09:16:58.963572 22514370438976 train.py:367] RMSE/MAE train: [0.04259785 0.03222654]
I0921 09:16:58.963806 22514370438976 train.py:367] RMSE/MAE validation: [0.05474784 0.03499521]
I0921 09:16:58.963953 22514370438976 train.py:367] RMSE/MAE test: [0.05678338 0.03520006]
I0921 09:23:07.636128 22514370438976 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/qm9/dynamic/round_False/32/gpu_a100/iteration_7/checkpoints/checkpoint_1900000.pkl
I0921 09:23:07.640480 22514370438976 train.py:646] Step 1900000 train loss: 1.0581983588053845e-05
I0921 09:23:17.191517 22514370438976 train.py:367] RMSE/MAE train: [0.03762809 0.02813114]
I0921 09:23:17.191755 22514370438976 train.py:367] RMSE/MAE validation: [0.05094656 0.0310673 ]
I0921 09:23:17.191905 22514370438976 train.py:367] RMSE/MAE test: [0.05345152 0.03106724]
I0921 09:29:21.729304 22514370438976 train.py:150] Serializing experiment state to /u/dansp/batching_2_000_000_steps_20_9_2024//profiling_experiments/MPEU/qm9/dynamic/round_False/32/gpu_a100/iteration_7/checkpoints/checkpoint_2000000.pkl
I0921 09:29:21.733754 22514370438976 train.py:646] Step 2000000 train loss: 4.694749804912135e-05
I0921 09:29:31.270494 22514370438976 train.py:367] RMSE/MAE train: [0.03707845 0.02783663]
I0921 09:29:31.270727 22514370438976 train.py:367] RMSE/MAE validation: [0.05057768 0.03068886]
I0921 09:29:31.270873 22514370438976 train.py:367] RMSE/MAE test: [0.05613954 0.03114205]
I0921 09:29:31.275461 22514370438976 train.py:674] Reached maximum number of steps without early stopping.
I0921 09:29:31.275787 22514370438976 train.py:681] Lowest validation loss: 0.050577678924907624
I0921 09:29:31.395223 22514370438976 train.py:684] Median batching time: 0.0010075569152832031
I0921 09:29:31.470323 22514370438976 train.py:687] Mean batching time: 0.0010253586618900298
I0921 09:29:31.569359 22514370438976 train.py:690] Median update time: 0.0030689239501953125
I0921 09:29:31.644672 22514370438976 train.py:693] Mean update time: 0.0030888630081415177
"""

sample_err_file_cancelled = sample_err_file + '\nCANCELLED DUE TO TIME LIMIT'

sample_err_file_cancelled_nothing_else = '\nCANCELLED DUE TO TIME LIMIT'


PATHS_TEXT_FILE = '/home/dts/Documents/hu/jraph_MPEU/tests/data/fake_profile_paths.txt'


class ParseProfileExperiments(unittest.TestCase):
    """Unit and integration test functions in models.py."""

    def setUp(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir_for_err_file = os.path.join(
                tmp_dir,
                'tests')
            os.makedirs(temp_dir_for_err_file)
            save_directory=os.path.join(tmp_dir,'tests')
            paths_txt_file = PATHS_TEXT_FILE
            csv_filename = None
            self.profiling_parser_object = ppe.LongerParser(
                paths_txt_file,
                csv_filename,
                save_directory
            )

    #TODO(dts): figure out what happens to the CSV writer if some fields
    # are missing ideally, I would like to write nan to those fields.
    def test_get_recompile_and_timing_info(self):
        # Write the sample text to file:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_file_name = os.path.join(tmp_dir, 'sample_err_file.err')
            with open(temp_file_name, 'w') as fd:
                fd.write(sample_err_file)
            # Now test reading the file.
            data_dict = {}
            data_dict = self.profiling_parser_object.get_recompile_and_timing_info(
                temp_file_name, data_dict)
            self.assertEqual(data_dict['recompilation_counter'], 6)
            self.assertEqual(data_dict['experiment_completed'], 1)
            # Test that we were able to get RMSE info
            self.assertEqual(data_dict['step_2_000_000_train_rmse'], 0.08793343)
            self.assertEqual(data_dict['step_2_000_000_val_rmse'], 0.07778961)
            self.assertEqual(data_dict['step_2_000_000_test_rmse'], 0.08889424)
            self.assertEqual(data_dict['step_2_000_000_batching_time_mean'], 0.0007143152430057525)
            self.assertEqual(data_dict['step_2_000_000_update_time_mean'], 0.002795692672967911)


    def test_get_recompile_and_timing_info_full(self):
        # Write the sample text to file:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_file_name = os.path.join(tmp_dir, 'sample_err_file_full.err')
            with open(temp_file_name, 'w') as fd:
                fd.write(sample_full_err_file)
            # Now test reading the file.
            data_dict = {}
            data_dict = self.profiling_parser_object.get_recompile_and_timing_info(
                temp_file_name, data_dict)
            self.assertEqual(data_dict['recompilation_counter'], 7)
            self.assertEqual(data_dict['experiment_completed'], 1)
            # Test that we were able to get MAE/RMSE info
            self.assertEqual(data_dict['step_100_000_train_rmse'], 0.20275778)
            self.assertEqual(data_dict['step_100_000_val_rmse'], 0.1509765)
            self.assertEqual(data_dict['step_2_000_000_batching_time_mean'], 0.0007143152430057525)
            self.assertEqual(data_dict['step_2_000_000_update_time_mean'], 0.002795692672967911)

    def test_update_dict_with_batching_method_size(self):
        """Test updating the dict with a batching method.
        
        
        schnet/qm9/static/round_True/16/gpu_a100/iteration_5
        """
        parent_path = 'tests/data/mpnn/aflow/dynamic/round_True/64/gpu_a100/iteration_5'
        data_dict = {}
        data_dict = self.profiling_parser_object.update_dict_with_batching_method_size(
            parent_path, data_dict)
        self.assertEqual(data_dict['batching_type'], 'dynamic')
        self.assertEqual(data_dict['iteration'], 5)
        self.assertEqual(data_dict['batching_round_to_64'], 'True')
        self.assertEqual(data_dict['computing_type'], 'gpu_a100')
        self.assertEqual(data_dict['batch_size'], 64)
        self.assertEqual(data_dict['model'], 'mpnn')
        self.assertEqual(data_dict['dataset'], 'aflow')
    

    def test_check_calc_finished(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_file_name = os.path.join(tmp_dir, 'sample_err_file.err')
            with open(temp_file_name, 'w') as fd:
                fd.write(sample_err_file)
            parent_path = Path(temp_file_name).parent.absolute()
            calc_ran_bool, most_recent_error_file = self.profiling_parser_object.check_experiment_ran(
                temp_file_name, parent_path)
            self.assertEqual(calc_ran_bool, True)
            self.assertEqual(most_recent_error_file, temp_file_name)


    def test_check_sim_time_lim(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_file_name_cancelled = os.path.join(tmp_dir, 'sample_err_file_cancelled.err')
            with open(temp_file_name_cancelled, 'w') as fd:
                fd.write(sample_err_file_cancelled)
            calc_expired = self.profiling_parser_object.check_sim_time_lim(
                temp_file_name_cancelled)
            self.assertEqual(calc_expired, True)

            temp_file_name = os.path.join(tmp_dir, 'sample_err_file.err')
            with open(temp_file_name, 'w') as fd:
                fd.write(sample_err_file)
            calc_expired = self.profiling_parser_object.check_sim_time_lim(
                temp_file_name)
            self.assertEqual(calc_expired, False)


    def test_gather_all_path_data_expired(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir_for_err_file = os.path.join(
                tmp_dir,
                'tests/data/mpnn/aflow/dynamic/round_False/64/gpu_a100/'
                'iteration_5')
            os.makedirs(temp_dir_for_err_file)
            temp_file_name_cancelled = os.path.join(
                temp_dir_for_err_file,
                'sample_err_file_cancelled_nothing_else.err')
            with open(temp_file_name_cancelled, 'w') as fd:
                fd.write(sample_err_file_cancelled_nothing_else)
            
            paths_to_resubmit = os.path.join(
                tmp_dir,
                'paths_to_resubmit.txt')

            profiling_parser_object = ppe.LongerParser(
                paths_txt_file=PATHS_TEXT_FILE,
                csv_filename=None,
                save_directory=tmp_dir)

            data_dict = profiling_parser_object.gather_all_path_data(
                temp_file_name_cancelled)
            self.assertEqual(
                data_dict['submission_path'],
                temp_file_name_cancelled)
            self.assertEqual(
                data_dict['iteration'], 5)
            self.assertEqual(
                data_dict['batching_round_to_64'], 'False')

    def test_create_header(self):
        """Test writing the header to the CSV."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir_for_err_file = os.path.join(
                tmp_dir,
                'tests/data/mpnn/aflow/dynamic/64/gpu_a100/'
                'iteration_5')
            os.makedirs(temp_dir_for_err_file)
            temp_file_name_cancelled = os.path.join(
                temp_dir_for_err_file,
                'sample_err_file.err')
            with open(temp_file_name_cancelled, 'w') as fd:
                fd.write(sample_err_file)
            
            paths_to_resubmit = os.path.join(
                tmp_dir,
                'paths_to_resubmit.txt')

            csv_file_name = os.path.join(
                temp_dir_for_err_file,
                'output.csv')
            profiling_parser_object = ppe.LongerParser(
                paths_txt_file=PATHS_TEXT_FILE,
                csv_filename=csv_file_name,
                save_directory=tmp_dir)
            profiling_parser_object.create_header()

            # Now open the CSV and count how many lines are there.
            df = pd.read_csv(csv_file_name)

            self.assertEqual(len(df.columns), 76)

    def test_write_all_path_data(self):
        """Test writing data for a submission path as a row to csv and db."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir_for_err_file = os.path.join(
                tmp_dir,
                'tests/data/mpnn/aflow/dynamic/round_False/64/gpu_a100/'
                'iteration_5')
            os.makedirs(temp_dir_for_err_file)
            temp_file_name_cancelled = os.path.join(
                temp_dir_for_err_file,
                'sample_err_file.err')
            with open(temp_file_name_cancelled, 'w') as fd:
                fd.write(sample_full_err_file)
            
            paths_to_resubmit = os.path.join(
                tmp_dir,
                'paths_to_resubmit.txt')

            csv_file_name = os.path.join(
                tmp_dir,
                'output.csv')

            paths_text_file = os.path.join(tmp_dir, 'path_text_file.csv')
            # Create a paths text file:
            with open(paths_text_file, 'w') as txt_file:
                txt_file.write(temp_file_name_cancelled)

            with open(paths_text_file, 'w') as txt_file:
                txt_file.write(temp_file_name_cancelled)

            profiling_parser_object = ppe.LongerParser(
                paths_txt_file=paths_text_file,
                csv_filename=csv_file_name,
                save_directory=tmp_dir)


            if not os.path.isfile(csv_file_name):
                profiling_parser_object.create_header()
            with open(csv_file_name, 'a') as csv_file:
                dict_writer = csv.DictWriter(
                    csv_file, fieldnames=profiling_parser_object.csv_columns, extrasaction='ignore')
                profiling_parser_object.write_all_path_data(dict_writer)
            df = pd.read_csv(csv_file_name)
            print(df)
            self.assertEqual(
                df['path'].values[0],
                temp_file_name_cancelled)
            self.assertEqual(
                df['experiment_completed'].values[0],
                True)
            self.assertEqual(
                df['recompilation_counter'].values[0],
                7)
            print(df)
            print(df.columns)

            self.assertEqual(
                df['step_100_000_train_rmse'].values[0],
                0.20275778)

    def test_get_recompile_and_timing_info_failed(self):
        # Write the sample text to file:
        with tempfile.TemporaryDirectory() as tmp_dir:
            failing_err_file = os.path.join(tmp_dir, 'failing_err_file.err')
            with open(failing_err_file, 'w') as fd:
                fd.write(profiling_err_file_failing)

            # Now test reading the file.
            data_dict = {}
            data_dict = self.profiling_parser_object.get_recompile_and_timing_info(
                failing_err_file, data_dict)
            print(data_dict)
            self.assertEqual(data_dict['recompilation_counter'], 6)
            self.assertEqual(data_dict['experiment_completed'], True)
            # Test that we were able to get MAE/RMSE info
            self.assertEqual(data_dict['step_100_000_train_rmse'], 0.21237597)
            self.assertEqual(data_dict['step_100_000_val_rmse'], 0.22647893)

            self.assertEqual(data_dict['step_100_000_test_rmse'], 0.21606945)
            self.assertEqual(
                data_dict['step_2_000_000_batching_time_mean'], 0.0011481321103572846)
            self.assertEqual(
                data_dict['step_2_000_000_update_time_mean'], 0.0026973051978349686)

    def test_get_convert_string_to_float_file(self):
        # Write the sample text to file:
        with tempfile.TemporaryDirectory() as tmp_dir:
            failing_err_file = os.path.join(tmp_dir, 'convert_string_to_float_file.err')
            with open(failing_err_file, 'w') as fd:
                fd.write(convert_string_to_float_file)

            # Now test reading the file.
            data_dict = {}
            data_dict = self.profiling_parser_object.get_recompile_and_timing_info(
                failing_err_file, data_dict)
            print(data_dict)
            self.assertEqual(data_dict['recompilation_counter'], 1)
            self.assertEqual(data_dict['experiment_completed'], True)
            # Test that we were able to get MAE/RMSE info
            self.assertEqual(data_dict['step_100_000_train_rmse'], 0.63762646)
            self.assertEqual(data_dict['step_400_000_val_rmse'], 0.12357951)

            self.assertEqual(data_dict['step_2_000_000_test_rmse'], 0.05613954)
            self.assertEqual(
                data_dict['step_2_000_000_batching_time_mean'], 0.0010253586618900298)
            self.assertEqual(
                data_dict['step_2_000_000_update_time_mean'], 0.0030888630081415177)

    def test_check_experiment_ran_empty_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir_for_err_file = os.path.join(
                tmp_dir,
                'tests/data/mpnn/aflow/dynamic/round_Talse/64/gpu_a100/'
                'iteration_5')
            os.makedirs(temp_dir_for_err_file)
            save_directory=os.path.join(tmp_dir,'tests')

            profiling_parser_object = ppe.LongerParser(
                paths_txt_file=PATHS_TEXT_FILE,
                csv_filename=None,
                save_directory=save_directory)
            # Now do not create a file in the folder and see what happens when we parse it.
            # temp_file_name_cancelled = os.path.join(
            #     temp_dir_for_err_file,
            #     'sample_err_file.err')
            # with open(temp_file_name_cancelled, 'w') as fd:
            #     fd.write(sample_full_err_file)
            with self.assertRaises(ValueError):
                profiling_parser_object.check_experiment_ran(
                    os.path.join(temp_dir_for_err_file, 'submission_MgO.sh'), temp_dir_for_err_file)
            with open(
                    profiling_parser_object.paths_resubmit_from_scratch, 'r') as fo:
                    parsed_resubmit_paths = fo.readlines()
                    print(parsed_resubmit_paths)
                    self.assertEqual(len(parsed_resubmit_paths), 1)
                    self.assertEqual(parsed_resubmit_paths[0], os.path.join(temp_dir_for_err_file, 'submission_MgO.sh\n'))

    def test_parsing_empty_folder(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir_for_err_file = os.path.join(
                tmp_dir,
                'tests/data/mpnn/aflow/dynamic/round_Talse/64/gpu_a100/'
                'iteration_5')
            os.makedirs(temp_dir_for_err_file)
            save_directory=os.path.join(tmp_dir,'tests')

            profiling_parser_object = ppe.LongerParser(
                paths_txt_file=PATHS_TEXT_FILE,
                csv_filename=None,
                save_directory=save_directory)
            # Now do not create a file in the folder and see what happens when we parse it.
            # temp_file_name_cancelled = os.path.join(
            #     temp_dir_for_err_file,
            #     'sample_err_file.err')
            # with open(temp_file_name_cancelled, 'w') as fd:
            #     fd.write(sample_full_err_file)
            with self.assertRaises(ValueError):
                profiling_parser_object.gather_all_path_data(
                    os.path.join(temp_dir_for_err_file, 'submission_MgO.sh'))

            with open(
                    profiling_parser_object.paths_resubmit_from_scratch, 'r') as fo:
                parsed_resubmit_paths = fo.readlines()
                print(parsed_resubmit_paths)
                self.assertEqual(len(parsed_resubmit_paths), 1)
                self.assertEqual(parsed_resubmit_paths[0], os.path.join(temp_dir_for_err_file, 'submission_MgO.sh\n'))


if __name__ == '__main__':
    unittest.main()
