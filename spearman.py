# -*- coding: utf-8 -*-
"""
Created on Thu May  7 04:22:26 2020

@author: karim
"""

import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
x = [-5, -4, -3, -2, -2 , -2 ,-2  -1, 0, 1, 2, 3, 4, 5]
y = [-3.2, -2.5, -2.1, -0.8, -1, 1.1, 2.1, 3.8, 6.5, 9.1, 13.8]
#plt.figure(figsize=(8, 6))
#plt.scatter(x, y, color='red')
#plt.xlabel("x-values", labelpad=13)
#plt.ylabel("y-values", labelpad=13)
#plt.title("Monotonic Relationship Between Two Variables", y=1.015)
#pear =  stats.pearsonr(x, y)
#spear = stats.spearmanr(x,y)

n, bins, patches = plt.hist(x=x, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('My Very Own Histogram')
plt.text(23, 45, r'$\mu=15, b=3$')

maxfreq = n.max()
# Set a clean upper y-axis limit.
#plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

print (pear)
print (spear)