#!/mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2020.02/bin/python3.7

import jax

devices = jax.local_devices()

print(devices)