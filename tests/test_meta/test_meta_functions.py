import ConfigSpace
import numpy as np
import openml
import openmlcontrib
import pandas as pd

from openmlcontrib.testing import TestBase


class TestMetaFunctions(TestBase):

    def get_task_flow_results_as_dataframe(self, task_id, flow_id, num_configs, config_space):
        df = openmlcontrib.meta.get_task_flow_results_as_dataframe(task_id, flow_id, num_configs,
                                                                   raise_few_runs=True,
                                                                   configuration_space=config_space,
                                                                   parameter_field='parameter_name',
                                                                   evaluation_measure='predictive_accuracy',
                                                                   cache_directory=None)

        self.assertEqual(type(df), pd.DataFrame)
        self.assertEqual(df.shape, (num_configs, len(config_space.get_hyperparameter_names()) + 1))

        for param in config_space.get_hyperparameters():
            if isinstance(param, ConfigSpace.hyperparameters.NumericalHyperparameter):
                self.assertIn(df[param.name].dtype, [np.float64, np.int64])
                self.assertGreater(df[param.name].min(), -1000000)
                self.assertLessEqual(df[param.name].max(), 1000000)
            elif isinstance(param, ConfigSpace.CategoricalHyperparameter):
                self.assertIn(df[param.name].dtype, [object])
            elif isinstance(param, ConfigSpace.Constant):
                self.assertIn(df[param.name].dtype, [np.float64, np.int64, object])
            elif isinstance(param, ConfigSpace.UnParametrizedHyperparameter):
                self.assertIn(df[param.name].dtype, [np.float64, np.int64, object])
            else:
                raise ValueError()
        self.assertGreater(df['y'].min(), -1000000)
        self.assertLessEqual(df['y'].max(), 1000000)
        self.assertIn(df['y'].dtype, [np.float64, np.int64])

    def test_get_task_flow_results_as_dataframe_svm(self):
        openml.config.server = 'https://www.openml.org/api/v1/'
        openml.config.apikey = None

        task_id = 59
        flow_id = 7707
        num_configs = 50
        cs = TestBase._get_libsvm_svc_config_space()

        self.get_task_flow_results_as_dataframe(task_id, flow_id, num_configs, cs)

    def test_get_task_flow_results_as_dataframe_rf(self):
        openml.config.server = 'https://www.openml.org/api/v1/'
        openml.config.apikey = None

        task_id = 3
        flow_id = 6969
        num_configs = 50
        cs = TestBase._get_random_forest_default_search_space()

        self.get_task_flow_results_as_dataframe(task_id, flow_id, num_configs, cs)

    def test_dataframe_to_arff(self):
        raise NotImplementedError()

    def test_arff_to_dataframe(self):
        raise NotImplementedError()
