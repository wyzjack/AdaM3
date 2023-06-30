# LSTM for Language Modeling

This folder contains the implementation of training LSTMs on Penn Treebank dataset. Each experiment is run $5$ times independently using random seeds 0 1 2 3 4.

## Environment setup

```
conda env create -f environment.yml
```

## Prepare data
```
sh getdata.sh
```

## Reproducing the results

### 1-layer LSTM
```
sh main_all_1layer.sh
```

### 2-layer LSTM
```
sh main_all_2layer.sh
```

### 3-layer LSTM
```
sh main_all_3layer.sh
```

### To plot the curves
```
python plot_curve.py
```

### To print the numerical results
```
python print_value.py
```

