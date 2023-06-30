import os
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd

# sns.set_theme(style="darkgrid")


params = {'axes.labelsize': 17,
          'axes.titlesize': 17,
         }

plt.rcParams.update(params)


def get_data(names):
    folder_path = './curve'
    paths = [os.path.join(folder_path, name) for name in names]
    return {name: torch.load(fp) for name, fp in zip(names, paths)}


def plot(names, curve_type='train', labels=None, ylim=(30, 120)):
    plt.figure()
    plt.ylim(ylim)  # if curve_type == 'train' else 96)
    curve_data = get_data(names)
    count=0

    # lenth = len(names)

    # index = np.arange(200)
    # index = np.tile(index,lenth)
    acc_multi = []
    # alg_name = []
    for i, label in zip(curve_data.keys(), labels):
        acc = np.array(curve_data[i]['{}_loss'.format(curve_type.lower())])
        acc_min = np.min(acc)
        # print(acc_min)
        acc_multi.append(acc_min)
        if count%5==4:
            print(label+': {:.2f}+-{:.2f}'.format(np.mean(acc_multi), np.std(acc_multi)))
            acc_multi=[]

        # print(acc)
        
        # print(label)
        # alg_name.append(np.tile(np.array([label]), 200))
        count = count + 1
        

    
    # alg_name = np.concatenate(alg_name, axis=0)
    # acc_pd = pd.DataFrame({'time':index, 'acc': acc_multi, 'alg':alg_name })
    # sns.lineplot(x='time', y='acc', data=acc_pd, hue='alg')

    


    # plt.grid(ls='--')
    # plt.legend(fontsize=12)
    # # plt.legend(fontsize=14, loc='lower left')
    # # plt.title('{} set perplexity ~ training epoch'.format(curve_type))
    # plt.xlabel('Epoch')
    # plt.ylabel('{} Perplexity'.format(curve_type.capitalize()))

# 1 layer
names = ['PTB.pt-niter-200-optimizer-adam3-nlayers1-lr0.001-clip-0.25-eps1e-16-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-adam3-nlayers1-lr0.001-clip-0.25-eps1e-16-betas-0.9-0.999-run1-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-adam3-nlayers1-lr0.001-clip-0.25-eps1e-16-betas-0.9-0.999-run2-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-adam3-nlayers1-lr0.001-clip-0.25-eps1e-16-betas-0.9-0.999-run3-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-adam3-nlayers1-lr0.001-clip-0.25-eps1e-16-betas-0.9-0.999-run4-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-sgd-nlayers1-lr30.0-clip-0.25-eps1e-08-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-sgd-nlayers1-lr30.0-clip-0.25-eps1e-08-betas-0.9-0.999-run1-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-sgd-nlayers1-lr30.0-clip-0.25-eps1e-08-betas-0.9-0.999-run2-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-sgd-nlayers1-lr30.0-clip-0.25-eps1e-08-betas-0.9-0.999-run3-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-sgd-nlayers1-lr30.0-clip-0.25-eps1e-08-betas-0.9-0.999-run4-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-adam-nlayers1-lr0.001-clip-0.25-eps1e-12-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-adam-nlayers1-lr0.001-clip-0.25-eps1e-12-betas-0.9-0.999-run1-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-adam-nlayers1-lr0.001-clip-0.25-eps1e-12-betas-0.9-0.999-run2-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-adam-nlayers1-lr0.001-clip-0.25-eps1e-12-betas-0.9-0.999-run3-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-adam-nlayers1-lr0.001-clip-0.25-eps1e-12-betas-0.9-0.999-run4-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-adamw-nlayers1-lr0.001-clip-0.25-eps1e-12-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-adamw-nlayers1-lr0.001-clip-0.25-eps1e-12-betas-0.9-0.999-run1-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-adamw-nlayers1-lr0.001-clip-0.25-eps1e-12-betas-0.9-0.999-run2-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-adamw-nlayers1-lr0.001-clip-0.25-eps1e-12-betas-0.9-0.999-run3-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-adamw-nlayers1-lr0.001-clip-0.25-eps1e-12-betas-0.9-0.999-run4-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-yogi-nlayers1-lr0.01-clip-0.25-eps0.001-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-yogi-nlayers1-lr0.01-clip-0.25-eps0.001-betas-0.9-0.999-run1-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-yogi-nlayers1-lr0.01-clip-0.25-eps0.001-betas-0.9-0.999-run2-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-yogi-nlayers1-lr0.01-clip-0.25-eps0.001-betas-0.9-0.999-run3-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-yogi-nlayers1-lr0.01-clip-0.25-eps0.001-betas-0.9-0.999-run4-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-adabound-nlayers1-lr0.01-clip-0.25-eps1e-08-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-adabound-nlayers1-lr0.01-clip-0.25-eps1e-08-betas-0.9-0.999-run1-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-adabound-nlayers1-lr0.01-clip-0.25-eps1e-08-betas-0.9-0.999-run2-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-adabound-nlayers1-lr0.01-clip-0.25-eps1e-08-betas-0.9-0.999-run3-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-adabound-nlayers1-lr0.01-clip-0.25-eps1e-08-betas-0.9-0.999-run4-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-radam-nlayers1-lr0.001-clip-0.25-eps1e-12-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-radam-nlayers1-lr0.001-clip-0.25-eps1e-12-betas-0.9-0.999-run1-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-radam-nlayers1-lr0.001-clip-0.25-eps1e-12-betas-0.9-0.999-run2-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-radam-nlayers1-lr0.001-clip-0.25-eps1e-12-betas-0.9-0.999-run3-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-radam-nlayers1-lr0.001-clip-0.25-eps1e-12-betas-0.9-0.999-run4-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-adabelief-nlayers1-lr0.001-clip-0.25-eps1e-16-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-adabelief-nlayers1-lr0.001-clip-0.25-eps1e-16-betas-0.9-0.999-run1-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-adabelief-nlayers1-lr0.001-clip-0.25-eps1e-16-betas-0.9-0.999-run2-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-adabelief-nlayers1-lr0.001-clip-0.25-eps1e-16-betas-0.9-0.999-run3-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-adabelief-nlayers1-lr0.001-clip-0.25-eps1e-16-betas-0.9-0.999-run4-wdecay1.2e-06-when-[100, 145]'
         ]
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

print("1 layer")
plot(names, 'Test', ylim=(80,130), labels = labels)


# 2 layer
names = ['PTB.pt-niter-200-optimizer-adam3-nlayers2-lr0.001-clip-0.25-eps1e-16-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adam3-nlayers2-lr0.001-clip-0.25-eps1e-16-betas-0.9-0.999-run1-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adam3-nlayers2-lr0.001-clip-0.25-eps1e-16-betas-0.9-0.999-run2-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adam3-nlayers2-lr0.001-clip-0.25-eps1e-16-betas-0.9-0.999-run3-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adam3-nlayers2-lr0.001-clip-0.25-eps1e-16-betas-0.9-0.999-run4-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-sgd-nlayers2-lr30.0-clip-0.25-eps1e-08-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-sgd-nlayers2-lr30.0-clip-0.25-eps1e-08-betas-0.9-0.999-run1-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-sgd-nlayers2-lr30.0-clip-0.25-eps1e-08-betas-0.9-0.999-run2-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-sgd-nlayers2-lr30.0-clip-0.25-eps1e-08-betas-0.9-0.999-run3-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-sgd-nlayers2-lr30.0-clip-0.25-eps1e-08-betas-0.9-0.999-run4-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adam-nlayers2-lr0.01-clip-0.25-eps1e-08-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adam-nlayers2-lr0.01-clip-0.25-eps1e-08-betas-0.9-0.999-run1-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adam-nlayers2-lr0.01-clip-0.25-eps1e-08-betas-0.9-0.999-run2-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adam-nlayers2-lr0.01-clip-0.25-eps1e-08-betas-0.9-0.999-run3-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adam-nlayers2-lr0.01-clip-0.25-eps1e-08-betas-0.9-0.999-run4-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adamw-nlayers2-lr0.001-clip-0.25-eps1e-08-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adamw-nlayers2-lr0.001-clip-0.25-eps1e-08-betas-0.9-0.999-run1-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adamw-nlayers2-lr0.001-clip-0.25-eps1e-08-betas-0.9-0.999-run2-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adamw-nlayers2-lr0.001-clip-0.25-eps1e-08-betas-0.9-0.999-run3-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adamw-nlayers2-lr0.001-clip-0.25-eps1e-08-betas-0.9-0.999-run4-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-yogi-nlayers2-lr0.01-clip-0.25-eps0.001-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-yogi-nlayers2-lr0.01-clip-0.25-eps0.001-betas-0.9-0.999-run1-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-yogi-nlayers2-lr0.01-clip-0.25-eps0.001-betas-0.9-0.999-run2-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-yogi-nlayers2-lr0.01-clip-0.25-eps0.001-betas-0.9-0.999-run3-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-yogi-nlayers2-lr0.01-clip-0.25-eps0.001-betas-0.9-0.999-run4-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adabound-nlayers2-lr0.01-clip-0.25-eps1e-08-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adabound-nlayers2-lr0.01-clip-0.25-eps1e-08-betas-0.9-0.999-run1-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adabound-nlayers2-lr0.01-clip-0.25-eps1e-08-betas-0.9-0.999-run2-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adabound-nlayers2-lr0.01-clip-0.25-eps1e-08-betas-0.9-0.999-run3-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adabound-nlayers2-lr0.01-clip-0.25-eps1e-08-betas-0.9-0.999-run4-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-radam-nlayers2-lr0.001-clip-0.25-eps1e-08-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-radam-nlayers2-lr0.001-clip-0.25-eps1e-08-betas-0.9-0.999-run1-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-radam-nlayers2-lr0.001-clip-0.25-eps1e-08-betas-0.9-0.999-run2-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-radam-nlayers2-lr0.001-clip-0.25-eps1e-08-betas-0.9-0.999-run3-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-radam-nlayers2-lr0.001-clip-0.25-eps1e-08-betas-0.9-0.999-run4-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adabelief-nlayers2-lr0.01-clip-0.25-eps1e-12-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adabelief-nlayers2-lr0.01-clip-0.25-eps1e-12-betas-0.9-0.999-run1-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adabelief-nlayers2-lr0.01-clip-0.25-eps1e-12-betas-0.9-0.999-run2-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adabelief-nlayers2-lr0.01-clip-0.25-eps1e-12-betas-0.9-0.999-run3-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adabelief-nlayers2-lr0.01-clip-0.25-eps1e-12-betas-0.9-0.999-run4-wdecay1.2e-06-when-[100, 145]'
         ]
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

print("2 layer")


plot(names, 'Test', ylim=(60,110), labels = labels)


# # 3 layer
names = ['PTB.pt-niter-200-optimizer-adam3-nlayers3-lr0.001-clip-0.25-eps1e-16-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adam3-nlayers3-lr0.001-clip-0.25-eps1e-16-betas-0.9-0.999-run1-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adam3-nlayers3-lr0.001-clip-0.25-eps1e-16-betas-0.9-0.999-run2-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adam3-nlayers3-lr0.001-clip-0.25-eps1e-16-betas-0.9-0.999-run3-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adam3-nlayers3-lr0.001-clip-0.25-eps1e-16-betas-0.9-0.999-run4-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-sgd-nlayers3-lr30.0-clip-0.25-eps1e-08-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-sgd-nlayers3-lr30.0-clip-0.25-eps1e-08-betas-0.9-0.999-run1-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-sgd-nlayers3-lr30.0-clip-0.25-eps1e-08-betas-0.9-0.999-run2-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-sgd-nlayers3-lr30.0-clip-0.25-eps1e-08-betas-0.9-0.999-run3-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-sgd-nlayers3-lr30.0-clip-0.25-eps1e-08-betas-0.9-0.999-run4-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adam-nlayers3-lr0.01-clip-0.25-eps1e-08-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adam-nlayers3-lr0.01-clip-0.25-eps1e-08-betas-0.9-0.999-run1-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adam-nlayers3-lr0.01-clip-0.25-eps1e-08-betas-0.9-0.999-run2-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adam-nlayers3-lr0.01-clip-0.25-eps1e-08-betas-0.9-0.999-run3-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adam-nlayers3-lr0.01-clip-0.25-eps1e-08-betas-0.9-0.999-run4-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adamw-nlayers3-lr0.001-clip-0.25-eps1e-08-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adamw-nlayers3-lr0.001-clip-0.25-eps1e-08-betas-0.9-0.999-run1-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adamw-nlayers3-lr0.001-clip-0.25-eps1e-08-betas-0.9-0.999-run2-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adamw-nlayers3-lr0.001-clip-0.25-eps1e-08-betas-0.9-0.999-run3-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adamw-nlayers3-lr0.001-clip-0.25-eps1e-08-betas-0.9-0.999-run4-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-yogi-nlayers3-lr0.01-clip-0.25-eps0.001-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-yogi-nlayers3-lr0.01-clip-0.25-eps0.001-betas-0.9-0.999-run1-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-yogi-nlayers3-lr0.01-clip-0.25-eps0.001-betas-0.9-0.999-run2-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-yogi-nlayers3-lr0.01-clip-0.25-eps0.001-betas-0.9-0.999-run3-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-yogi-nlayers3-lr0.01-clip-0.25-eps0.001-betas-0.9-0.999-run4-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adabound-nlayers3-lr0.01-clip-0.25-eps1e-08-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adabound-nlayers3-lr0.01-clip-0.25-eps1e-08-betas-0.9-0.999-run1-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adabound-nlayers3-lr0.01-clip-0.25-eps1e-08-betas-0.9-0.999-run2-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adabound-nlayers3-lr0.01-clip-0.25-eps1e-08-betas-0.9-0.999-run3-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-adabound-nlayers3-lr0.01-clip-0.25-eps1e-08-betas-0.9-0.999-run4-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-radam-nlayers3-lr0.001-clip-0.25-eps1e-08-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-radam-nlayers3-lr0.001-clip-0.25-eps1e-08-betas-0.9-0.999-run1-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-radam-nlayers3-lr0.001-clip-0.25-eps1e-08-betas-0.9-0.999-run2-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-radam-nlayers3-lr0.001-clip-0.25-eps1e-08-betas-0.9-0.999-run3-wdecay1.2e-06-when-[100, 145]',
        'PTB.pt-niter-200-optimizer-radam-nlayers3-lr0.001-clip-0.25-eps1e-08-betas-0.9-0.999-run4-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-adabelief-nlayers3-lr0.01-clip-0.25-eps1e-12-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-adabelief-nlayers3-lr0.01-clip-0.25-eps1e-12-betas-0.9-0.999-run1-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-adabelief-nlayers3-lr0.01-clip-0.25-eps1e-12-betas-0.9-0.999-run2-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-adabelief-nlayers3-lr0.01-clip-0.25-eps1e-12-betas-0.9-0.999-run3-wdecay1.2e-06-when-[100, 145]',
         'PTB.pt-niter-200-optimizer-adabelief-nlayers3-lr0.01-clip-0.25-eps1e-12-betas-0.9-0.999-run4-wdecay1.2e-06-when-[100, 145]'
         ]
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

print("3 layer")

plot(names, 'Test', ylim=(55, 110), labels=labels)

