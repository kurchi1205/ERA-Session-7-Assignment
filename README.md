# ERA-Session-7-Assignment

## Problem Statement 
To write a model of less than 8000 parameters that achieves 99.4% validation accuracy consistently for few epochs on MNIST data.

## Approach
I have written 3 models to reach the objective. Model training is present in each of the colab files, based on the model you are trying to train.
Code is as modular as possible. 
- All data transforms are in data.py
- train and test function are in utils.py
- models are in model.py with their receptive field calculation.


## Result
- train Accuracy: 99.27
- test Accuracy: 99.32
- model parameters: 7870
- receptive field: 28
