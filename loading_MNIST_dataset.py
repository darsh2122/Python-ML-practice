#!/usr/bin/env python
# coding: utf-8
'''
darsh M.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#change location as required
df = pd.read_csv('../mnist_train.csv') 
print(df.head())
print(df.shape)

#storing actual digit
l = df['label']
d = df.drop('label',axis=1)
plt.figure(figsize=(7,7))
grid = d.iloc[150].as_matrix().reshape(28,28)
plt.imshow(grid,interpolation = "none",cmap="gray")
print(l[150])





