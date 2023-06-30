# Generatvie Adversarial Network

This folder contains the implementation of training DCGAN and BigGAN on CIFAR-10. Each experiment is run $5$ times independently.

## Environment setup

```
conda env create -f environment.yml
```

## Reproducing the results of DCGAN
```
python src/main.py -t -hdf5 -l -batch_stat -metrics fid  -ref "test" -cfg src/configs/CIFAR10/DCGAN_adam3.yaml -save checkpoints/ -data data/ --seed 0 

python src/main.py -t -hdf5 -l -batch_stat -metrics fid  -ref "test" -cfg src/configs/CIFAR10/DCGAN_adam3.yaml -save checkpoints/ -data data/ --seed 1

python src/main.py -t -hdf5 -l -batch_stat -metrics fid  -ref "test" -cfg src/configs/CIFAR10/DCGAN_adam3.yaml -save checkpoints/ -data data/ --seed 2

python src/main.py -t -hdf5 -l -batch_stat -metrics fid  -ref "test" -cfg src/configs/CIFAR10/DCGAN_adam3.yaml -save checkpoints/ -data data/ --seed 3

python src/main.py -t -hdf5 -l -batch_stat -metrics fid  -ref "test" -cfg src/configs/CIFAR10/DCGAN_adam3.yaml -save checkpoints/ -data data/ --seed 4
```

## Reproducing the results of BigGAN
```
python src/main.py -t -hdf5 -l -batch_stat -metrics fid  -ref "test" -cfg src/configs/CIFAR10/BigGAN-Mod-CR_adam3.yaml -save checkpoints/ -data data/ --seed 0 

python src/main.py -t -hdf5 -l -batch_stat -metrics fid  -ref "test" -cfg src/configs/CIFAR10/BigGAN-Mod-CR_adam3.yaml -save checkpoints/ -data data/ --seed 1

python src/main.py -t -hdf5 -l -batch_stat -metrics fid  -ref "test" -cfg src/configs/CIFAR10/BigGAN-Mod-CR_adam3.yaml -save checkpoints/ -data data/ --seed 2

python src/main.py -t -hdf5 -l -batch_stat -metrics fid  -ref "test" -cfg src/configs/CIFAR10/BigGAN-Mod-CR_adam3.yaml -save checkpoints/ -data data/ --seed 3

python src/main.py -t -hdf5 -l -batch_stat -metrics fid  -ref "test" -cfg src/configs/CIFAR10/BigGAN-Mod-CR_adam3.yaml -save checkpoints/ -data data/ --seed 4
```

