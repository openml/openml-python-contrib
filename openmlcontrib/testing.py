import ConfigSpace
import unittest


class TestBase(unittest.TestCase):

    @staticmethod
    def _get_libsvm_svc_config_space():
        C = ConfigSpace.UniformFloatHyperparameter("C", 0.03125, 32768, log=True, default_value=1.0)
        kernel = ConfigSpace.CategoricalHyperparameter(name="kernel",
                                                       choices=["rbf", "poly", "sigmoid"], default_value="rbf")
        degree = ConfigSpace.UniformIntegerHyperparameter("degree", 1, 5, default_value=3)
        gamma = ConfigSpace.UniformFloatHyperparameter("gamma", 3.0517578125e-05, 8, log=True,
                                                       default_value=0.1)
        coef0 = ConfigSpace.UniformFloatHyperparameter("coef0", -1, 1, default_value=0)

        cs = ConfigSpace.ConfigurationSpace()
        cs.add_hyperparameters([C, kernel, degree, gamma, coef0])

        degree_depends_on_poly = ConfigSpace.EqualsCondition(degree, kernel, "poly")
        coef0_condition = ConfigSpace.InCondition(coef0, kernel, ["poly", "sigmoid"])
        cs.add_condition(degree_depends_on_poly)
        cs.add_condition(coef0_condition)
        return cs


__all__ = ['TestBase']
