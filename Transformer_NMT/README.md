# Transformer for Neural Machine Translation

This folder contains the implementation of training transformers on IWSTL'14 DE-EN dataset. Each experiment is run $5$ times independently using random seeds 0 1 2 3 4. The code is partially adapted from https://github.com/pytorch/fairseq.git.

## Environment setup
### create conda environment
```
conda env create -f environment.yml
```
### set up fairseq
```
pip install --editable .
```

## Prepare data
```
sh prepare-iwslt14.sh
```

## Reproducing the results

### SGDM

```
sh sh_files/sgdm.sh
```
### Adam

```
sh sh_files/adam.sh
```
### AdamW

```
sh sh_files/adamw.sh
```
### AdaBelief

```
sh sh_files/adabelief.sh
```

### AdaM3

```
sh sh_files/adam3.sh
```



 