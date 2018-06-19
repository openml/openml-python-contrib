import ConfigSpace
import numpy as np
import openml
import openmlcontrib
import pandas as pd

from openmlcontrib.testing import TestBase


class TestMetaFunctions(TestBase):

    def test_get_task_flow_results_as_dataframe(self):
        openml.config.server = 'https://www.openml.org/api/v1/'

        num_configs = 50

        cs = TestBase._get_libsvm_svc_config_space()
        df = openmlcontrib.meta.get_task_flow_results_as_dataframe(59, 7707, num_configs, configuration_space=cs,
                                                                   parameter_field='parameter_name',
                                                                   evaluation_measure='predictive_accuracy',
                                                                   cache_directory=None)

        self.assertEqual(type(df), pd.DataFrame)
        self.assertEqual(df.shape, (num_configs, len(cs.get_hyperparameter_names()) + 1))

        for param in cs.get_hyperparameters():
            if isinstance(param, ConfigSpace.hyperparameters.NumericalHyperparameter):
                self.assertEqual(df[param.name].dtype, np.float64)
                self.assertGreater(df[param.name].min(), -1000000)
                self.assertLessEqual(df[param.name].max(), 1000000)
            elif isinstance(param, ConfigSpace.CategoricalHyperparameter):
                self.assertIn(df[param.name].dtype, [object])
            elif isinstance(param, ConfigSpace.UnParametrizedHyperparameter):
                self.assertIn(df[param.name].dtype, [object])
            else:
                raise ValueError()
        self.assertGreater(df['y'].min(), -1000000)
        self.assertLessEqual(df['y'].max(), 1000000)
        self.assertIn(df['y'].dtype, [np.float64, np.int64])
