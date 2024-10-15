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

sample_err_file_cancelled = sample_err_file + '\nCANCELLED DUE TO TIME LIMIT'

sample_err_file_cancelled_nothing_else = '\nCANCELLED DUE TO TIME LIMIT'


PATHS_TEXT_FILE = '/home/dts/Documents/hu/jraph_MPEU/tests/data/fake_profile_paths.txt'


class ParseProfileExperiments(unittest.TestCase):
    """Unit and integration test functions in models.py."""

    def setUp(self):
        paths_text_file = PATHS_TEXT_FILE
        csv_filename = None
        paths_to_resubmit = None
        paths_misbehaving = None
        self.profiling_parser_object = ppe.LongerParser(
            paths_text_file,
            csv_filename,
            paths_to_resubmit,
            paths_misbehaving
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
                parent_path)
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
                paths_to_resubmit=paths_to_resubmit,
                paths_misbehaving=None)

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
                paths_to_resubmit=paths_to_resubmit,
                paths_misbehaving=None)
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
                paths_to_resubmit=paths_to_resubmit,
                paths_misbehaving=None)


            if not os.path.isfile(csv_file_name):
                profiling_parser_object.create_header()
            with open(csv_file_name, 'a') as csv_file:
                dict_writer = csv.DictWriter(
                    csv_file, fieldnames=profiling_parser_object.csv_columns, extrasaction='ignore')
                profiling_parser_object.write_all_path_data(dict_writer)
            df = pd.read_csv(csv_file_name)
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



if __name__ == '__main__':
    unittest.main()
