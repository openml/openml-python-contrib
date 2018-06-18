import ConfigSpace
import numpy as np
import openml
import openmlcontrib
import pandas as pd
import unittest


class TestMetaFunctions(unittest.TestCase):

    @staticmethod
    def _get_valid_config_space():
        C = ConfigSpace.UniformFloatHyperparameter("C", 0.03125, 32768, log=True, default_value=1.0)
        kernel = ConfigSpace.CategoricalHyperparameter(name="kernel",
                                                       choices=["rbf", "poly", "sigmoid"], default_value="rbf")
        degree = ConfigSpace.UniformIntegerHyperparameter("degree", 1, 5, default_value=3)
        gamma = ConfigSpace.UniformFloatHyperparameter("gamma", 3.0517578125e-05, 8, log=True,
                                                       default_value=0.1)
        coef0 = ConfigSpace.UniformFloatHyperparameter("coef0", -1, 1, default_value=0)

        cs = ConfigSpace.ConfigurationSpace()
        cs.add_hyperparameters([C, kernel, degree, gamma, coef0])
        return cs

    def test_get_task_flow_results_as_dataframe(self):
        openml.config.server = 'https://www.openml.org/api/v1/'

        num_configs = 50

        cs = TestMetaFunctions._get_valid_config_space()
        df = openmlcontrib.meta.get_task_flow_results_as_dataframe(59, 7707, num_configs, configuration_space=cs,
                                                                   parameter_field='parameter_name',
                                                                   evaluation_measure='predictive_accuracy',
                                                                   cache_directory=None)

        self.assertEqual(type(df), pd.DataFrame)
        self.assertEqual(df.shape, (num_configs, len(cs.get_hyperparameter_names()) + 1))

        for param in cs.get_hyperparameters():
            if isinstance(param, ConfigSpace.UniformFloatHyperparameter):
                self.assertEquals(df[param.name].dtype, np.float64)
            elif isinstance(param, ConfigSpace.UniformIntegerHyperparameter):
                self.assertEquals(df[param.name].dtype, np.int64)
            elif isinstance(param, ConfigSpace.CategoricalHyperparameter):
                self.assertIn(df[param.name].dtype, [object])
            else:
                raise ValueError()
        self.assertIn(df['y'].dtype, [np.float64, np.int64])
