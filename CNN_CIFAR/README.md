# CNN for Image Classification

This folder contains the implementation of training CNNs (ResNet, VGGNet and DenseNet) on 
CIFAR-10. Each experiment is run $5$ times independently using random seeds 0 1 2 3 4.

## Environment setup

```
conda env create -f environment.yml
```

## Reproducing the results

### VGGNet-16 on CIFAR-10 
```
sh main_all_vgg_cifar10.sh
```

### ResNet-34 on CIFAR-10 
```
sh main_all_res_cifar10.sh
```

### DenseNet-121 on CIFAR-10
```
sh main_all_res_cifar10.sh
```

The log data in dictionary format are in folder `./curve_cifar10`.

##### To plot the curves
```
python plot_curve.py
```

##### To print the numerical results
```
python print_value.py
```


