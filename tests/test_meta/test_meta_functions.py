import ConfigSpace
import numpy as np
import openml
import openmlcontrib
import pandas as pd

from openmlcontrib.testing import TestBase


class TestMetaFunctions(TestBase):

    @staticmethod
    def is_integer_hyperparameter(hyperparameter):
        if isinstance(hyperparameter, ConfigSpace.hyperparameters.UniformIntegerHyperparameter):
            return True
        if isinstance(hyperparameter, ConfigSpace.hyperparameters.UnParametrizedHyperparameter) \
                and isinstance(hyperparameter.value, int):
            return True
        if isinstance(hyperparameter, ConfigSpace.hyperparameters.Constant) \
                and isinstance(hyperparameter.value, int):
            return True
        return False

    def get_task_flow_results_as_dataframe(self, task_id, flow_id, num_configs, num_folds, measures, config_space):
        if num_folds is None:
            df = openmlcontrib.meta.get_task_flow_results_as_dataframe(task_id, flow_id, num_configs,
                                                                       raise_few_runs=True,
                                                                       configuration_space=config_space,
                                                                       evaluation_measures=measures,
                                                                       cache_directory=None)

            n_columns = len(config_space.get_hyperparameter_names()) + len(measures)
            n_observations = num_configs
        else:
            df = openmlcontrib.meta.get_task_flow_results_per_fold_as_dataframe(
                task_id, flow_id, num_configs, raise_few_runs=True, configuration_space=config_space,
                evaluation_measures=measures)
            n_columns = len(config_space.get_hyperparameter_names()) + len(measures) + 2
            n_observations = num_configs * num_folds

        self.assertEqual(type(df), pd.DataFrame)
        self.assertEqual(df.shape, (n_observations, n_columns))

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
            if 'usercpu_time' not in measure:
                self.assertLessEqual(df[measure].max(), 1)
            self.assertIn(df[measure].dtype, [np.float64, np.int64])

        # note that we test the to and from dataframe features in this function
        for param in config_space.get_hyperparameters():
            # note that arff does not maintain knowledge about integer vs float.
            # therefore, we transform integer columns to floats
            if TestMetaFunctions.is_integer_hyperparameter(param):
                df[param.name] = df[param.name].astype(np.float64)
        # also fix repeat nr and fold nr, in case of the per fold result
        for rf in ['repeat_nr', 'fold_nr']:
            if rf in df.columns.values:
                df[rf] = df[rf].astype(np.float64)

        arff = openmlcontrib.meta.dataframe_to_arff(df, 'name', None)
        df_reproduced = openmlcontrib.meta.arff_to_dataframe(arff)
        pd.testing.assert_frame_equal(df, df_reproduced)

    def test_get_task_flow_results_as_dataframe_svm(self):
        openml.config.server = 'https://www.openml.org/api/v1/'
        openml.config.apikey = None

        task_id = 59
        num_folds = 10
        flow_id = 7707
        num_configs_global = 50
        num_configs_perfold = 10
        cs = TestBase._get_libsvm_svc_config_space()
        measures = ['predictive_accuracy']

        self.get_task_flow_results_as_dataframe(task_id=task_id,
                                                flow_id=flow_id,
                                                num_configs=num_configs_perfold,
                                                num_folds=num_folds,
                                                measures=measures,
                                                config_space=cs)
        self.get_task_flow_results_as_dataframe(task_id=task_id,
                                                flow_id=flow_id,
                                                num_configs=num_configs_global,
                                                num_folds=None,  # invoke test on global frame
                                                measures=measures,
                                                config_space=cs)

    def test_get_task_flow_results_as_dataframe_rf(self):
        openml.config.server = 'https://www.openml.org/api/v1/'
        openml.config.apikey = None

        task_id = 3
        num_folds = 10
        flow_id = 6969
        num_configs_global = 50
        num_configs_perfold = 10
        cs = TestBase._get_random_forest_default_search_space()
        measures_global = [
            'predictive_accuracy',
            'f_measure',
            'area_under_roc_curve'
        ]
        measures_perfold = [
            'predictive_accuracy',
            'usercpu_time_millis',
            'usercpu_time_millis_training',
            'usercpu_time_millis_testing'
        ]

        self.get_task_flow_results_as_dataframe(task_id=task_id,
                                                flow_id=flow_id,
                                                num_configs=num_configs_global,
                                                num_folds=None,  # invoke test on global frame
                                                measures=measures_global,
                                                config_space=cs)
        self.get_task_flow_results_as_dataframe(task_id=task_id,
                                                flow_id=flow_id,
                                                num_configs=num_configs_perfold,
                                                num_folds=num_folds,
                                                measures=measures_perfold,
                                                config_space=cs)
