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
        self.profiling_parser_object = ppe.ProfilingParser(
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
            self.assertEqual(data_dict['recompilation_counter'], 1)
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
            self.assertEqual(data_dict['recompilation_counter'], 6)
            self.assertEqual(data_dict['experiment_completed'], 1)
            # Test that we were able to get MAE/RMSE info
            self.assertEqual(data_dict['step_100000_train_rmse'], 0.0042509)
            self.assertEqual(data_dict['step_100000_val_rmse'], 0.08130376)
            self.assertEqual(data_dict['step_1000000_batching_time_mean'], 0.0009820856218338012)
            self.assertEqual(data_dict['step_1000000_update_time_mean'], 0.003449098623037338)

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

            profiling_parser_object = ppe.ProfilingParser(
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
            profiling_parser_object = ppe.ProfilingParser(
                paths_txt_file=PATHS_TEXT_FILE,
                csv_filename=csv_file_name,
                paths_to_resubmit=paths_to_resubmit,
                paths_misbehaving=None)
            profiling_parser_object.create_header()

            # Now open the CSV and count how many lines are there.
            df = pd.read_csv(csv_file_name)

            self.assertEqual(len(df.columns), 19)

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

            profiling_parser_object = ppe.ProfilingParser(
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
                1)

            self.assertEqual(
                df['step_100000_train_rmse'].values[0],
                0.0042509)


    def test_get_recompile_and_timing_info_failed(self):
        # Write the sample text to file:
        failing_err_file = 'tests/data/profiling_err_file_failing.err'

        # Now test reading the file.
        data_dict = {}
        data_dict = self.profiling_parser_object.get_recompile_and_timing_info(
            failing_err_file, data_dict)
        self.assertEqual(data_dict['recompilation_counter'], 60)
        self.assertEqual(data_dict['experiment_completed'], True)
        # Test that we were able to get MAE/RMSE info
        self.assertEqual(data_dict['step_100000_train_rmse'], 0.47112376)
        self.assertEqual(data_dict['step_100000_val_rmse'], 0.466972)

        self.assertEqual(data_dict['step_100000_test_rmse'], 0.46532637)
        self.assertEqual(data_dict['step_100000_batching_time_mean'], 0.0005266756558418273)
        self.assertEqual(data_dict['step_100000_update_time_mean'], 0.0031542533135414125)



if __name__ == '__main__':
    unittest.main()
