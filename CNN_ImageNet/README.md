# CNN for ImageNet

This folder contains the implementation of training ResNet-18 on ImageNet using different optimizers. Download ImageNet data into folder `.\ImageNet`.

## Environment setup

```
conda env create -f environment.yml
```

## Reproducing the results



### Run SGDM optimizer
```
python main.py --optimizer sgd --lr 0.1 --wd 1e-4 --beta1 0.9 --beta2 0.999 --eps 1e-8  --batch-size 256 --arch resnet18  --epochs 100 --lr_decay cosine --last_lr 1e-4
```
### Run Adam optimizer
```
python main.py --optimizer adam --lr 1e-3 --wd 1e-4 --beta1 0.9 --beta2 0.999 --eps 1e-8  --batch-size 256 --arch resnet18 --epochs 100 --lr_decay cosine --last_lr 5e-6
```
### Run AdaM3 optimzier
```
python main.py --optimizer adam3 --lr 1e-3 --wd 5e-2 --beta1 0.9 --beta2 0.999 --eps 1e-16 --batch-size 256 --arch resnet18 --epochs 100 --lr_decay cosine --last_lr 5e-6
```