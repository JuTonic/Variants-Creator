import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np

plt.style.use('fast')
plt.rcParams["font.family"] = 'Open Sans'

dist1_ = np.random.normal(loc = 0, scale = 0.8, size = 400)
dist1__ = np.random.normal(loc = 0, scale = 2, size = 100)
dist2_ = np.random.normal(loc = 0, scale = 0.8, size = 100)
dist2__ = np.random.normal(loc = 0, scale = 2, size = 400)
bins = [-5.5 + i for i in range(12)]

dist1 = np.concatenate((dist1_, dist1__))
dist2 = np.concatenate((dist2_, dist2__))

fig, axs = plt.subplots(1, 2)
fig.set_size_inches(12, 3.2)
fig.subplots_adjust(bottom=0.2)

counts, edges, bars = axs[0].hist(dist1, color = "grey", ec='black', bins = bins)
counts1, edges1, bars1 = axs[1].hist(dist2, color = "grey", ec='black', bins = bins)
xlim_min = min(edges1)
xlim_max = max(edges1)
axs[0].set_title('Распределение отклонений Ивана', size=13, pad = 10)
axs[0].set_xlabel('на сколько метров стрела отклонилась влево или вправо')
axs[0].set_ylabel('частота')
axs[0].set_xlim(xlim_min, xlim_max)
axs[0].set_ylim(0, max(counts) + 20)
axs[0].bar_label(bars)
axs[1].set_title('Распределение отклонений Арсения', size=13, pad = 10)
axs[1].set_xlabel('на сколько метров стрела отклонилась влево или вправо')
axs[1].set_xlim(xlim_min, xlim_max)
axs[1].set_ylim(0, max(counts) + 20)
axs[1].bar_label(bars1)


plt.savefig('plots/plot.svg')