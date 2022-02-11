import ml_collections
import numpy as np
import unittest

import train
from configs import default_test as cfg

class TestHelperFunctions(unittest.TestCase):
    def test_config_import(self):
        config = cfg.get_config()
        self.assertEqual(config.batch_size, 32)
        
    
if __name__ == '__main__':
    unittest.main()
