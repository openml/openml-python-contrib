import argparse
import ConfigSpace
import matplotlib.pyplot as plt
import numpy as np
import openmlcontrib.testing
import os
import pandas as pd
import sklearn.ensemble



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_directory', type=str, default=os.path.expanduser('~') + '/experiments/openml_cache')
    parser.add_argument('--task_id', type=int, default=3)
    parser.add_argument('--n_estimators', type=int, default=64)
    parser.add_argument('--num_runs', type=int, default=1500)
    parser.add_argument('--evaluation_measure', type=str, default='predictive_accuracy')
    parser.add_argument('--x_param', type=str, default='C')
    parser.add_argument('--y_param', type=str, default='gamma')
    parser.add_argument('--resolution', type=int, default=16)

    return parser.parse_args()


def get_flow_and_config_space():
    flow_id = 7707

    kernel = ConfigSpace.Constant(name="kernel", value="rbf")
    complexity = ConfigSpace.UniformFloatHyperparameter("C", 0.03125, 32768, log=True, default_value=1.0)
    gamma = ConfigSpace.UniformFloatHyperparameter("gamma", 3.0517578125e-05, 8, log=True, default_value=0.1)

    cs = ConfigSpace.ConfigurationSpace()
    cs.add_hyperparameters([complexity, kernel, gamma])

    return flow_id, cs


def get_param_indices(param, resolution):
    if param.log:
        return np.logspace(np.log(param.lower), np.log(param.upper), resolution)
    else:
        return np.linspace(param.lower, param.upper, resolution)


def run():
    args = parse_args()
    flow_id, configuration_space = get_flow_and_config_space()

    df = openmlcontrib.meta.get_task_flow_results_as_dataframe(args.task_id, flow_id, args.num_runs,
                                                               configuration_space, 'parameter_name',
                                                               args.evaluation_measure, args.cache_directory)
    # categoricals = [idx for idx, dtype in enumerate(df.dtypes) if not np.issubdtype(dtype, np.number)]
    # print(categoricals)
    estimator = sklearn.pipeline.Pipeline(steps=[
        ('estimator', sklearn.ensemble.RandomForestRegressor(n_estimators=args.n_estimators))
    ])

    X = np.concatenate((df[args.x_param].values.reshape(-1, 1), df[args.y_param].values.reshape(-1, 1)), axis=1)
    y = df['y'].values
    estimator.fit(X, y)

    x_param = configuration_space.get_hyperparameter(args.x_param)
    y_param = configuration_space.get_hyperparameter(args.y_param)

    XX, YY = np.meshgrid(get_param_indices(x_param, args.resolution), get_param_indices(y_param, args.resolution))
    ZZ = np.zeros((len(YY), len(XX)), dtype=np.float64)
    for i in range(args.resolution):
        for j in range(args.resolution):
            data_point = [[YY[i, j], XX[i, j]]]
            # TODO check indices (x / y mixup)
            ZZ[i, j] = estimator.predict(data_point)[0]

    fig, axes = plt.subplots(1, 1)
    axes.contour(XX, YY, ZZ)
    axes.set_xlim(x_param.lower, x_param.upper)
    axes.set_ylim(y_param.lower, y_param.upper)
    if x_param.log:
        axes.set_xscale("log")
    if y_param.log:
        axes.set_yscale("log")
    plt.show()


if __name__ == '__main__':
    run()
