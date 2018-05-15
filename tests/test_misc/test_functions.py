import openml
import openmlcontrib
import unittest


class TestMiscFunctions(unittest.TestCase):

    def setUp(self):
        openml.config.server = "https://test.openml.org/api/v1/xml/"

    def test_filter_data_listing(self):
        prop = 'name'
        value = 'kr-vs-kp'
        data_list = openml.datasets.list_datasets(tag='study_14')

        data_list_prime = openmlcontrib.misc.filter_listing(data_list, prop, [value])

        self.assertEqual(len(data_list_prime), 1)
        data_id = next(iter(data_list_prime))
        self.assertEqual(data_list_prime[data_id][prop], value)

    def test_filter_task_listing(self):
        prop = 'did'
        value = 1
        data_list = openml.datasets.list_datasets(tag='study_14')

        task_list_prime = openmlcontrib.misc.filter_listing(data_list, prop, [value])

        self.assertEqual(len(task_list_prime), 1)
        task_id = next(iter(task_list_prime))
        self.assertEqual(task_list_prime[task_id][prop], value)

    def test_filter_task_listing_illeg(self):
        prop = 'doesntexist'
        value = 1
        data_list = openml.datasets.list_datasets(tag='study_14')

        with self.assertRaises(ValueError):
            openmlcontrib.misc.filter_listing(data_list, prop, value)