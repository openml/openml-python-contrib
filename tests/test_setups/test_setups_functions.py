import openml
import openmlcontrib
import unittest


class TestSetupFunctions(unittest.TestCase):

    def setUp(self):
        self.live_server = "https://www.openml.org/api/v1/xml/"
        self.test_server = "https://test.openml.org/api/v1/xml/"
        openml.config.server = self.test_server

    def test_obtain_setups_by_ids(self):
        setup_ids = range(1, 30)
        setups = openmlcontrib.setups.obtain_setups_by_ids(setup_ids, limit=7)
        self.assertEquals(set(setups.keys()), set(setup_ids))

    def test_obtain_setups_by_ids_incomplete_raise(self):
        with self.assertRaises(ValueError):
            setup_ids = range(30)
            openmlcontrib.setups.obtain_setups_by_ids(setup_ids, limit=7)

    def test_obtain_setups_by_ids_incomplete(self):
        setup_ids = range(30)
        openmlcontrib.setups.obtain_setups_by_ids(setup_ids, require_all=False, limit=7)

    def test_filter_setup_list_nominal(self):
        openml.config.server = self.live_server
        setupid_setup = openml.setups.list_setups(flow=7707, size=100)  # pipeline with libsvm svc

        poly_setups = openmlcontrib.setups.filter_setup_list(setupid_setup, 'kernel', allowed_values='poly')
        sigm_setups = openmlcontrib.setups.filter_setup_list(setupid_setup, 'kernel', allowed_values='sigmoid')
        poly_ids = set(poly_setups.keys())
        sigm_ids = set(sigm_setups.keys())
        inters = poly_ids.intersection(sigm_ids)

        self.assertEquals(len(inters), 0)
        self.assertGreater(len(poly_ids) + len(sigm_ids), 20)
        self.assertGreater(len(poly_ids), 10)
        self.assertGreater(len(sigm_ids), 10)

        poly_setups_prime = openmlcontrib.setups.filter_setup_list(poly_setups, 'kernel', allowed_values='poly')
        self.assertEquals(poly_ids, set(poly_setups_prime.keys()))

    def test_filter_setup_list_nominal(self):
        openml.config.server = self.live_server
        setupid_setup = openml.setups.list_setups(flow=7707, size=100)  # pipeline with libsvm svc
        threshold = 3
        poly_setups = openmlcontrib.setups.filter_setup_list(setupid_setup, 'kernel', allowed_values='poly')

        poly_setups_smaller = openmlcontrib.setups.filter_setup_list(setupid_setup, 'degree', max=threshold)
        poly_setups_bigger = openmlcontrib.setups.filter_setup_list(setupid_setup, 'degree', min=threshold+1)

        smaller_ids = set(poly_setups_smaller.keys())
        bigger_ids = set(poly_setups_bigger.keys())
        all_ids = set(poly_setups.keys())
        inters = smaller_ids.intersection(bigger_ids)

        self.assertEquals(len(inters), 0)
        self.assertEquals(len(smaller_ids) + len(bigger_ids), len(all_ids))
