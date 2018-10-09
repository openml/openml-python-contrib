import ConfigSpace
import numpy as np
import openml
import openmlcontrib
import pandas as pd

from openmlcontrib.testing import TestBase


class TestMetaFunctions(TestBase):

    def get_task_flow_results_as_dataframe(self, task_id, flow_id, num_configs, measures, config_space):
        df = openmlcontrib.meta.get_task_flow_results_as_dataframe(task_id, flow_id, num_configs,
                                                                   raise_few_runs=True,
                                                                   configuration_space=config_space,
                                                                   evaluation_measures=measures,
                                                                   cache_directory=None)
        n_columns = len(config_space.get_hyperparameter_names()) + len(measures)
        self.assertEqual(type(df), pd.DataFrame)
        self.assertEqual(df.shape, (num_configs, n_columns))

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

        for measure in measures:
            # please only test on measures that comply to these numbers
            self.assertGreaterEqual(df[measure].min(), 0)
            self.assertLessEqual(df[measure].max(), 1)
            self.assertIn(df[measure].dtype, [np.float64, np.int64])

        # note that we test the to and from dataframe features in this function
        arff = openmlcontrib.meta.dataframe_to_arff(df, 'name', None)
        for param in config_space.get_hyperparameters():
            # note that arff does not maintain knowledge about integer vs float.
            # therefore, we transform integer columns to floats
            if isinstance(param, ConfigSpace.hyperparameters.UniformIntegerHyperparameter):
                df[param.name] = df[param.name].astype(np.float64)

        df_reproduced = openmlcontrib.meta.arff_to_dataframe(arff)
        pd.testing.assert_frame_equal(df, df_reproduced)

    def test_get_task_flow_results_as_dataframe_svm(self):
        openml.config.server = 'https://www.openml.org/api/v1/'
        openml.config.apikey = None

        task_id = 59
        flow_id = 7707
        num_configs = 50
        cs = TestBase._get_libsvm_svc_config_space()
        measures = ['predictive_accuracy']

        self.get_task_flow_results_as_dataframe(task_id, flow_id, num_configs, measures, cs)

    def test_get_task_flow_results_as_dataframe_rf(self):
        openml.config.server = 'https://www.openml.org/api/v1/'
        openml.config.apikey = None

        task_id = 3
        flow_id = 6969
        num_configs = 50
        cs = TestBase._get_random_forest_default_search_space()
        measures = ['predictive_accuracy', 'f_measure', 'area_under_roc_curve']

        self.get_task_flow_results_as_dataframe(task_id, flow_id, num_configs, measures, cs)
