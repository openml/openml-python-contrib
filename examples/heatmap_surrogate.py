import argparse
import ConfigSpace
import logging
import matplotlib.pyplot as plt
import numpy as np
import openml
import openmlcontrib.testing
import os
import sklearn.ensemble


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_directory', type=str, default=os.path.expanduser('~') + '/experiments/openml_cache')
    parser.add_argument('--task_id', type=int, default=None)
    parser.add_argument('--study_id', type=str, default='OpenML100')
    parser.add_argument('--n_estimators', type=int, default=64)
    parser.add_argument('--num_runs', type=int, default=500)
    parser.add_argument('--evaluation_measure', type=str, default='predictive_accuracy')
    parser.add_argument('--x_param', type=str, default='classifier__C')
    parser.add_argument('--y_param', type=str, default='classifier__gamma')
    parser.add_argument('--resolution', type=int, default=32)
    parser.add_argument('--n_contour_lines', type=int, default=16)
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~') + '/experiments/openml')

    args_ = parser.parse_args()
    if (args_.task_id is None) == (args_.study_id is None):
        raise ValueError('Please set task_id XOR study_id')

    return args_


def get_flow_and_config_space():
    flow_id = 7707

    kernel = ConfigSpace.Constant(name="classifier__kernel", value="rbf")
    complexity = ConfigSpace.UniformFloatHyperparameter("classifier__C", 0.03125, 32768, log=True, default_value=1.0)
    gamma = ConfigSpace.UniformFloatHyperparameter("classifier__gamma", 3.0517578125e-05, 8, log=True, default_value=0.1)

    cs = ConfigSpace.ConfigurationSpace()
    cs.add_hyperparameters([complexity, kernel, gamma])

    return flow_id, cs


def get_param_indices(param: ConfigSpace.hyperparameters.Hyperparameter,
                      resolution: int,
                      tol: float=0.00001):
    if param.log:
        space = np.logspace(np.log10(param.lower), np.log10(param.upper), resolution, base=10)
    else:
        space = np.linspace(param.lower, param.upper, resolution)
    if np.abs(space[0] - param.lower) > tol:
        raise ValueError('Space does not start with param lower bound %f: %f' % (param.lower, space[0]))
    if np.abs(space[-1] - param.upper) > tol:
        raise ValueError('Space does not end with param upper bound %f: %f' % (param.upper, space[-1]))
    return space


def run(args):
    flow_id, configuration_space = get_flow_and_config_space()
    dataset_name = openml.tasks.get_task(args.task_id).get_dataset().name
    df = openmlcontrib.meta.get_task_flow_results_as_dataframe(args.task_id, flow_id, args.num_runs,
                                                               True, configuration_space,
                                                               [args.evaluation_measure], args.cache_directory)
    estimator = sklearn.pipeline.Pipeline(steps=[
        ('estimator', sklearn.ensemble.RandomForestRegressor(n_estimators=args.n_estimators))
    ])
    X = np.concatenate((df[args.x_param].values.reshape(-1, 1), df[args.y_param].values.reshape(-1, 1)), axis=1)
    y = df[args.evaluation_measure].values
    estimator.fit(X, y)

    x_param = configuration_space.get_hyperparameter(args.x_param)
    y_param = configuration_space.get_hyperparameter(args.y_param)
    x_resolution = args.resolution
    y_resolution = args.resolution
    XX, YY = np.meshgrid(get_param_indices(x_param, x_resolution), get_param_indices(y_param, y_resolution))
    ZZ = np.zeros((y_resolution, x_resolution), dtype=np.float64)

    for i in range(y_resolution):
        for j in range(x_resolution):
            data_point = [[XX[i, j], YY[i, j]]]
            ZZ[i, j] = estimator.predict(data_point)[0]

    fig, axes = plt.subplots(1, 1)
    contours = axes.contourf(XX, YY, ZZ, args.n_contour_lines, cmap=plt.cm.viridis)
    fig.colorbar(contours)

    axes.set_xlim(x_param.lower, x_param.upper)
    axes.set_ylim(y_param.lower, y_param.upper)
    if x_param.log:
        axes.set_xscale("log")
    if y_param.log:
        axes.set_yscale("log")

    axes.set_title('Contour plot on Task %d: %s' % (args.task_id, dataset_name))
    axes.set_xlabel(args.x_param)
    axes.set_ylabel(args.y_param)
    if args.output_directory:
        os.makedirs(args.output_directory, exist_ok=True)
        output_file = os.path.join(args.output_directory, 'contourplot-task-%d.png' % args.task_id)
        plt.savefig(output_file)
        logging.info('Saved plot to %s' % output_file)
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    
    args_ = parse_args()

    if args_.task_id:
        run(args_)
    else:
        tasks = openml.study.get_study(args_.study_id, 'tasks').tasks
        for i in tasks:
            args_.task_id = i
            run(args_)
