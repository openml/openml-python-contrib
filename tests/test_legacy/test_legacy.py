import openmlcontrib
import os
import pickle
import unittest

from openmlcontrib.testing import TestBase


class TestLegacyFunctions(TestBase):
    
    def test_get_active_hyperparameters(self):
        cs = TestBase._get_libsvm_svc_config_space()
        
        expected_active_parameters = {
            'poly': {'C', 'kernel', 'degree', 'gamma', 'coef0'}, 
            'rbf': {'C', 'kernel', 'gamma'}, 
            'sigmoid': {'C', 'kernel', 'gamma', 'coef0'}, 
        }

        for setup_file in os.listdir('../data/setups'):
            with open(os.path.join('../data/setups', setup_file), 'rb') as fp:
                setup = pickle.load(fp)

            # this function calls to openmlcontrib.legacy.get_active_hyperparameters()
            setup_dict = openmlcontrib.setups.setup_to_parameter_dict(setup, 'parameter_name', cs)

            self.assertSetEqual(expected_active_parameters[setup_dict['kernel']], set(setup_dict.keys()))
