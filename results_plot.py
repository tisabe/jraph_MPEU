import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import gaussian_kde

df_test_pre = pandas.read_csv('results_test/test_pre.csv')
df_train_pre = pandas.read_csv('results_test/train_pre.csv')

df_test_post = pandas.read_csv('results_test/test_post.csv')
df_train_post = pandas.read_csv('results_test/train_post.csv')

#print(df_test_post['x'].to_numpy())
#print(df_test_post['y'])
fig, ax = plt.subplots(2, 2)

marker_size = 0.3
ax[0,0].scatter(df_train_pre['x'].to_numpy(), df_train_pre['y'].to_numpy(), s=marker_size)
ax[0,0].axline((0,0), slope=1, color='red')
ax[0,0].set_xlim([-40,10])
ax[0,0].set_ylim([-40,10])
ax[0,0].set_title('train pre')

ax[0,1].scatter(df_train_post['x'].to_numpy(), df_train_post['y'].to_numpy(), s=marker_size)
ax[0,1].axline((0,0), slope=1, color='red')
ax[0,1].set_xlim([-40,10])
ax[0,1].set_ylim([-40,10])
ax[0,1].set_title('train post')

ax[1,0].scatter(df_test_pre['x'].to_numpy(), df_test_pre['y'].to_numpy(), s=marker_size)
ax[1,0].axline((0,0), slope=1, color='red')
ax[1,0].set_xlim([-40,10])
ax[1,0].set_ylim([-40,10])
ax[1,0].set_title('test pre')

ax[1,1].scatter(df_test_post['x'].to_numpy(), df_test_post['y'].to_numpy(), s=marker_size)
ax[1,1].axline((0,0), slope=1, color='red')
ax[1,1].set_xlim([-40,10])
ax[1,1].set_ylim([-40,10])
ax[1,1].set_title('test post')

plt.show()
# other plot styles
'''
x = df_train_post['x'].to_numpy()
y = df_train_post['y'].to_numpy()
# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=10)
ax.set_xlim([-40,10])
ax.set_ylim([-40,10])
plt.show()

fig, ax = plt.subplots()
ax.hist2d(x, y, bins=100, range=((-40,10),(-40,10)))
plt.show()
'''


