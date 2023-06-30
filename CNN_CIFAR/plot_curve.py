import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import seaborn as sns

names = ['AdaM3_seed0',
          'AdaM3_seed1',
          'AdaM3_seed2',
          'AdaM3_seed3',
          'AdaM3_seed4',
          'SGDM_seed0',
          'SGDM_seed1',
          'SGDM_seed2',
          'SGDM_seed3',
          'SGDM_seed4',
          'Adam_seed0',
          'Adam_seed1',
          'Adam_seed2',
          'Adam_seed3',
          'Adam_seed4',
          'AdamW_seed0',
          'AdamW_seed1',
          'AdamW_seed2',
          'AdamW_seed3',
          'AdamW_seed4',
          'Yogi_seed0',
          'Yogi_seed1',
          'Yogi_seed2',
          'Yogi_seed3',
          'Yogi_seed4',
          'AdaBound_seed0',
          'AdaBound_seed1',
          'AdaBound_seed2',
          'AdaBound_seed3',
          'AdaBound_seed4',
          'RAdam_seed0',
          'RAdam_seed1',
          'RAdam_seed2',
          'RAdam_seed3',
          'RAdam_seed4',
          'AdaBelief_seed0',
          'AdaBelief_seed1',
          'AdaBelief_seed2',
          'AdaBelief_seed3',
          'AdaBelief_seed4']

params = {'axes.labelsize': 17,
          'axes.titlesize': 17,
         }

labels = ['AdaM3',
          'SGDM',
          'Adam',
          'AdamW',
          'Yogi',
          'AdaBound',
          'RAdam',
          'AdaBelief'
          ]
labels = np.repeat(labels,5)

plt.rcParams.update(params)


def get_curve_data():
    folder_path = 'curve_cifar10/dense-121'
    filenames = [name for name in os.listdir(folder_path)]
    paths = [os.path.join(folder_path, name) for name in filenames]
    keys = [name for name in filenames]
    return {key: torch.load(fp) for key, fp in zip(keys, paths)}


def plot(optimizers=None, curve_type='train'):
    assert curve_type in ['train', 'test'], 'Invalid curve type: {}'.format(curve_type)
    assert all(_ in names for _ in optimizers), 'Invalid optimizer'

    curve_data = get_curve_data()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('{} Accuracy (%)'.format(curve_type.capitalize()))
    plt.ylim(87,100 if curve_type == 'train' else 97)

    lenth = len(optimizers)

    index = np.arange(200)
    index = np.tile(index,lenth)

    accuracies=[]
    alg_name = []

    for optim in optimizers:
        accuracy = np.array(curve_data[optim[:-6].lower()+optim[-6:]]['{}_acc'.format(curve_type)])
        accuracies.append(accuracy)
        # print(optim[:-6])
        alg_name.append(np.tile(np.array([optim[:-6]]), 200))
        # plt.plot(accuracies, label=optim, ls=linestyle)
    
    accuracies = np.concatenate(accuracies, axis=0)
    alg_name = np.concatenate(alg_name, axis=0)
    acc_pd = pd.DataFrame({'time':index, 'acc': accuracies, 'alg':alg_name })
    sns.lineplot(x='time', y='acc', data=acc_pd, hue='alg')

    plt.grid(ls='--')
    plt.legend()
    # plt.show()
    plt.savefig('{}_curve_dense_cifar10_multi.pdf'.format(curve_type), bbox_inches='tight')


# print value
curve_data = get_curve_data()

# for optim in names:
#     accuracies = np.array(curve_data[optim.lower()]['test_acc'])
#     accuracy = np.max(accuracies)
#     print(accuracy)
plot(optimizers=names, curve_type='train')
plot(optimizers=names, curve_type='test')
