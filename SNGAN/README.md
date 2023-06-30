# Generatvie Adversarial Network

This folder contains the implementation of training SNGAN on CIFAR-10. Each experiment is run $5$ times independently.

## Environment setup

```
conda env create -f environment.yml
```


## Reproducing the results
```
python main.py -t -e -c configs/CIFAR10/SNGAN-adam3.json
```

