import openml
import openmlcontrib
import os
import pickle

from openmlcontrib.testing import TestBase


class TestLegacyFunctions(TestBase):
    
    def test_get_active_hyperparameters(self):
        # This function used to be part of the legacy library, but since
        # automl/ConfigSpace:#85 integrated in ConfigSpace
        expected_active_parameters = TestBase._libsvm_expected_active_hyperparameters()
        cs = TestBase._get_libsvm_svc_config_space()

        for setup_file in os.listdir('../data/setups'):
            with open(os.path.join('../data/setups', setup_file), 'rb') as fp:
                setup = pickle.load(fp)

            with open('../data/flows/%d.pkl' % setup.flow_id, 'rb') as fp:
                flow = pickle.load(fp)

            # this function calls to Configuration.get_active_hyperparameters()
            setup_dict = openmlcontrib.setups.setup_to_parameter_dict(setup, flow, True, cs)

            self.assertSetEqual(expected_active_parameters[setup_dict['classifier__kernel']],
                                set(setup_dict.keys()))
