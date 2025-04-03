import matplotlib.pyplot as plt
import numpy as np
import statistics as stats
from math import sqrt
from math import cos

beta0 = np.random.uniform(low = -10, high = 10)
beta1 = np.random.uniform(low = -1, high = 1)
err = np.random.choice([1, 2, 3])
plt.style.use('fast')
plt.rcParams["font.family"] = 'Open Sans'
x = np.random.normal(loc = np.random.uniform(low = -20, high = 20), scale = 10, size = 300)
y = np.random.normal(loc = np.random.uniform(low = -20, high = 20), scale = 20, size = 300)
plt.scatter(x, y, c = 'black', s=14)

plt.show()