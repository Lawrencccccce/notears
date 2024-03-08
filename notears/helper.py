
from datetime import datetime
from pytz import timezone
import io
import numpy as np
import os
import matplotlib.pyplot as plt
import pathlib
import sys
import time


def create_dir(dir_path):
    """
    dir_path - A path of directory to create if it is not found
    :param dir:
    :return exit_code: 0:success -1:failed
    """
    try:
        if not os.path.exists(dir_path):
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

        return 0
    except Exception as err:
        sys.exit(-1)



def generate_data_path(dataset):
    # Load the data
    current_dir = os.getcwd()
    datapath = os.path.join('datasets', dataset + '.npy')
    datapath = os.path.join(current_dir, datapath)

    sol_path = os.path.join('datasets', dataset + '_sol.npy')
    sol_path = os.path.join(current_dir, sol_path)

    plot_dir = os.path.join(current_dir, 'plots')
    create_dir(plot_dir)

    return datapath, sol_path, plot_dir


def plot_result(estimate_g, ground_truth, plot_dir, dataset, title, accuracy = None):

    fig = plt.figure(3)
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title(title + f" SHD: {accuracy['shd']}")
    ax.imshow(np.around(estimate_g).astype(int),cmap=plt.cm.binary)
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('true_graph')
    ax.imshow(ground_truth, cmap=plt.cm.binary)
    plt.savefig('{}/{}_estimated_graph_{}.png'.format(plot_dir, dataset, datetime.now(timezone('Australia/Sydney')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]))
    plt.close()
    print(f"{title} \n {accuracy}")