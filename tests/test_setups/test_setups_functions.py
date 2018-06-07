import ConfigSpace
import openml
import openmlcontrib
import os
import pickle
import unittest


class TestSetupFunctions(unittest.TestCase):

    def setUp(self):
        self.live_server = "https://www.openml.org/api/v1/xml/"
        self.test_server = "https://test.openml.org/api/v1/xml/"
        openml.config.server = self.test_server

    def test_obtain_setups_by_ids(self):
        setup_ids = [i for i in range(1, 30)]
        setups = openmlcontrib.setups.obtain_setups_by_ids(setup_ids, limit=7)
        self.assertEquals(set(setups.keys()), set(setup_ids))

    def test_obtain_setups_by_ids_incomplete_raise(self):
        with self.assertRaises(ValueError):
            setup_ids = [i for i in range(30)]
            openmlcontrib.setups.obtain_setups_by_ids(setup_ids, limit=7)

    def test_obtain_setups_by_ids_incomplete(self):
        setup_ids = [i for i in range(30)]
        openmlcontrib.setups.obtain_setups_by_ids(setup_ids, require_all=False, limit=7)

    def test_filter_setup_list_nominal(self):
        openml.config.server = self.live_server
        setupid_setup = openml.setups.list_setups(flow=7707, size=100)  # pipeline with libsvm svc

        poly_setups = openmlcontrib.setups.filter_setup_list(setupid_setup, 'kernel', allowed_values=['poly'])
        sigm_setups = openmlcontrib.setups.filter_setup_list(setupid_setup, 'kernel', allowed_values=['sigmoid'])
        poly_ids = set(poly_setups.keys())
        sigm_ids = set(sigm_setups.keys())
        inters = poly_ids.intersection(sigm_ids)

        self.assertEquals(len(inters), 0)
        self.assertGreater(len(poly_ids) + len(sigm_ids), 20)
        self.assertGreater(len(poly_ids), 10)
        self.assertGreater(len(sigm_ids), 10)

        poly_setups_prime = openmlcontrib.setups.filter_setup_list(poly_setups, 'kernel', allowed_values=['poly'])
        self.assertEquals(poly_ids, set(poly_setups_prime.keys()))

    def test_filter_setup_list_nominal_numeric(self):
        openml.config.server = self.live_server
        setupid_setup = openml.setups.list_setups(flow=7707, size=100)  # pipeline with libsvm svc
        threshold = 3
        poly_setups = openmlcontrib.setups.filter_setup_list(setupid_setup, 'kernel', allowed_values=['poly'])

        poly_setups_smaller = openmlcontrib.setups.filter_setup_list(setupid_setup, 'degree', max=threshold)
        poly_setups_bigger = openmlcontrib.setups.filter_setup_list(setupid_setup, 'degree', min=threshold+1)

        smaller_ids = set(poly_setups_smaller.keys())
        bigger_ids = set(poly_setups_bigger.keys())
        all_ids = set(poly_setups.keys())
        inters = smaller_ids.intersection(bigger_ids)

        self.assertEquals(len(inters), 0)
        self.assertEquals(len(smaller_ids) + len(bigger_ids), len(all_ids))

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

    def test_setup_to_configuration(self):
        cs = TestSetupFunctions._get_valid_config_space()

        for setup_file in os.listdir('../data/setups'):
            with open(os.path.join('../data/setups', setup_file), 'rb') as fp:
                setup = pickle.load(fp)

            openmlcontrib.setups.setup_to_configuration(setup, cs)
            self.assertTrue(openmlcontrib.setups.setup_in_config_space(setup, cs))


    def filter_setup_list_by_config_space(self):
        cs = TestSetupFunctions._get_valid_config_space()

        setups = {}
        for setup_file in os.listdir('../data/setups'):
            with open(os.path.join('../data/setups', setup_file), 'rb') as fp:
                setup = pickle.load(fp)
            setups[setup.id] = setup
        self.assertEquals(len(setups), 20)
        setups_filtered = openmlcontrib.setups.filter_setup_list_by_config_space(setups, cs)
        self.assertEquals(len(setups), len(setups_filtered))

    def filter_setup_list_by_config_space_fails(self):
        degree = ConfigSpace.UniformIntegerHyperparameter("degree", -5, -1, default_value=-3)

        cs = ConfigSpace.ConfigurationSpace()
        cs.add_hyperparameters([degree])

        setups = {}
        for setup_file in os.listdir('../data/setups'):
            with open(os.path.join('../data/setups', setup_file), 'rb') as fp:
                setup = pickle.load(fp)
            setups[setup.id] = setup
        self.assertEquals(len(setups), 20)
        setups_filtered = openmlcontrib.setups.filter_setup_list_by_config_space(setups, cs)
        self.assertEquals(len(setups_filtered), 0)

    def test_setup_to_configuration_raises_illegal_value(self):
        degree = ConfigSpace.UniformIntegerHyperparameter("degree", -5, -1, default_value=-3)
        cs = ConfigSpace.ConfigurationSpace()
        cs.add_hyperparameters([degree])

        for setup_file in os.listdir('../data/setups'):
            with open(os.path.join('../data/setups', setup_file), 'rb') as fp:
                setup = pickle.load(fp)

            with self.assertRaises(ValueError):
                openmlcontrib.setups.setup_to_configuration(setup, cs)
            self.assertFalse(openmlcontrib.setups.setup_in_config_space(setup, cs))

    def test_setup_to_configuration_raises_param_not_present(self):
        degree = ConfigSpace.UniformIntegerHyperparameter("test123", -20, 20, default_value=-3)
        cs = ConfigSpace.ConfigurationSpace()
        cs.add_hyperparameters([degree])

        for setup_file in os.listdir('../data/setups'):
            with open(os.path.join('../data/setups', setup_file), 'rb') as fp:
                setup = pickle.load(fp)

            with self.assertRaises(ValueError):
                openmlcontrib.setups.setup_to_configuration(setup, cs)
            self.assertFalse(openmlcontrib.setups.setup_in_config_space(setup, cs))
