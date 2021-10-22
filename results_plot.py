import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib

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
'''
t = np.arange(-1, 2, .01)
s = np.sin(2 * np.pi * t)

fig, ax = plt.subplots()

ax.plot(t, s)
# Thick red horizontal line at y=0 that spans the xrange.
ax.axhline(linewidth=8, color='#d62728')
# Horizontal line at y=1 that spans the xrange.
ax.axhline(y=1)
# Vertical line at x=1 that spans the yrange.
ax.axvline(x=1)
# Thick blue vertical line at x=0 that spans the upper quadrant of the yrange.
ax.axvline(x=0, ymin=0.75, linewidth=8, color='#1f77b4')
# Default hline at y=.5 that spans the middle half of the axes.
ax.axhline(y=.5, xmin=0.25, xmax=0.75)
# Infinite black line going through (0, 0) to (1, 1).
ax.axline((0, 0), (1, 1), color='k')
# 50%-gray rectangle spanning the axes' width from y=0.25 to y=0.75.
ax.axhspan(0.25, 0.75, facecolor='0.5')
# Green rectangle spanning the axes' height from x=1.25 to x=1.55.
ax.axvspan(1.25, 1.55, facecolor='#2ca02c')

plt.show()'''