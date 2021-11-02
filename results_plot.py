import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score

folder = 'results_test/res_50ep_avg_5lr_binaries_enthalpy/'
#folder = 'results_norm/'
#folder = 'results_test/'
df_test_pre = pandas.read_csv(folder+'test_pre.csv')
df_train_pre = pandas.read_csv(folder+'train_pre.csv')

df_test_post = pandas.read_csv(folder+'test_post.csv')
df_train_post = pandas.read_csv(folder+'train_post.csv')

# calculate fit metrics
x_train = df_train_post['x'].to_numpy()
y_train = df_train_post['y'].to_numpy()
r2_train = r2_score(x_train, y_train)
print('R2 score on training data: {}'.format(r2_train))
error = np.abs(x_train - y_train)
mae = np.mean(error)
rmse = np.sqrt(np.mean(np.square(error)))
print('MAE on training data: {}'.format(mae))
print('RMSE score on training data: {}'.format(rmse))

x_test = df_test_post['x'].to_numpy()
y_test = df_test_post['y'].to_numpy()
r2_test = r2_score(x_test, y_test)
print('R2 score on test data: {}'.format(r2_test))
error = np.abs(x_test - y_test)
mae = np.mean(error)
rmse = np.sqrt(np.mean(np.square(error)))
print('MAE on test data: {}'.format(mae))
print('RMSE score on test data: {}'.format(rmse))

#print(df_test_post['x'].to_numpy())
#print(df_test_post['y'])
fig, ax = plt.subplots(2, 2)

marker_size = 0.3
min_x = -20
max_x = 10
min_y = min_x
max_y = max_x
ax[0,0].scatter(df_train_pre['x'].to_numpy(), df_train_pre['y'].to_numpy(), s=marker_size, label='prediction')
ax[0,0].axline((0,0), slope=1, color='red', label='x=y')
#ax[0,0].set_xlim([min_x,max_x])
#ax[0,0].set_ylim([min_y,max_y])
ax[0,0].set_title('train pre')
ax[0,0].set_xlabel('target')
ax[0,0].set_ylabel('prediction')
ax[0,0].legend()

ax[0,1].scatter(df_train_post['x'].to_numpy(), df_train_post['y'].to_numpy(), s=marker_size)
ax[0,1].axline((0,0), slope=1, color='red', label='x=y')
#ax[0,1].set_xlim([min_x,max_x])
#ax[0,1].set_ylim([min_y,max_y])
ax[0,1].set_title('train post')
ax[0,1].set_xlabel('target')
ax[0,1].set_ylabel('prediction')
ax[0,1].legend()

ax[1,0].scatter(df_test_pre['x'].to_numpy(), df_test_pre['y'].to_numpy(), s=marker_size)
ax[1,0].axline((0,0), slope=1, color='red', label='x=y')
#ax[1,0].set_xlim([min_x,max_x])
#ax[1,0].set_ylim([min_y,max_y])
ax[1,0].set_title('test pre')
ax[1,0].set_xlabel('target')
ax[1,0].set_ylabel('prediction')
ax[1,0].legend()

ax[1,1].scatter(df_test_post['x'].to_numpy(), df_test_post['y'].to_numpy(), s=marker_size)
ax[1,1].axline((0,0), slope=1, color='red', label='x=y')
#ax[1,1].set_xlim([min_x,max_x])
#ax[1,1].set_ylim([min_y,max_y])
ax[1,1].set_title('test post')
ax[1,1].set_xlabel('target')
ax[1,1].set_ylabel('prediction')
ax[1,1].legend()

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
fig, ax = plt.subplots()
ax.scatter(df_test_post['x'].to_numpy(), df_test_post['y'].to_numpy(), s=2)
ax.axline((0,0), slope=1, color='red')
#ax[1,1].set_xlim([min_x,max_x])
#ax[1,1].set_ylim([min_y,max_y])
ax.set_title('test post')

plt.show()

### plot learning curves
from numpy import genfromtxt
test_loss = genfromtxt(folder+'test_loss_arr.csv', delimiter=',')
train_loss = genfromtxt(folder+'train_loss_arr.csv', delimiter=',')

fig, ax = plt.subplots()
ax.plot(test_loss[:,0], test_loss[:,1], label='test data')
ax.plot(train_loss[:,0], train_loss[:,1], label='train data')
ax.legend()
ax.set_xlabel('epoch')
ax.set_ylabel('loss (MAE)')
plt.yscale('log')
plt.show()

