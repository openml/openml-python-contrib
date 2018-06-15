import numpy as np
import openml
import openmlcontrib
import pandas as pd
import unittest


class TestMetaFunctions(unittest.TestCase):
    
    def test_get_task_flow_results_as_dataframe(self):
        openml.config.server = 'https://www.openml.org/api/v1/'

        relevant_parameters = {'C': 'numeric', 'gamma': 'numeric', 'degree': 'numeric', 'strategy': 'nominal'}
        num_configs = 50

        df = openmlcontrib.meta.get_task_flow_results_as_dataframe(59, 7707, num_configs,
                                                                   relevant_parameters=relevant_parameters,
                                                                   evaluation_measure='predictive_accuracy',
                                                                   cache_directory=None)

        self.assertEquals(type(df), pd.DataFrame)
        self.assertEquals(df.shape, (num_configs, len(relevant_parameters) + 1))

        for param, data_type in relevant_parameters.items():
            if data_type == 'numeric':
                self.assertIn(df[param].dtype, [np.float64, np.int64])
            else:
                self.assertIn(df[param].dtype, [object])
        self.assertIn(df['y'].dtype, [np.float64, np.int64])
