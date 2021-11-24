import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score

folder = 'results_test/res_slow_learn/'
#folder = 'results_norm/'
#folder = 'results_test/'

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
fig, ax = plt.subplots()

marker_size = 0.3
min_x = -20
max_x = 10
min_y = min_x
max_y = max_x

ax.scatter(df_train_post['x'].to_numpy(), df_train_post['y'].to_numpy(), s=marker_size, label='training')
ax.scatter(df_test_post['x'].to_numpy(), df_test_post['y'].to_numpy(), s=marker_size, label='testing')

ax.axline((0,0), slope=1, color='red', label='x=y')
#ax[1,1].set_xlim([min_x,max_x])
#ax[1,1].set_ylim([min_y,max_y])
ax.set_title('test post')
ax.set_xlabel('target')
ax.set_ylabel('prediction')
ax.legend()

plt.show()

### plot learning curves
from numpy import genfromtxt
test_loss = genfromtxt(folder+'test_loss_arr.csv', delimiter=',')
train_loss = genfromtxt(folder+'train_loss_arr.csv', delimiter=',')

fig, ax = plt.subplots()
ax.plot(test_loss[:,0], test_loss[:,1], label='test data')
ax.plot(train_loss[:,0], train_loss[:,1], label='train data')
#ax.set_ylim([1e-3,1e2])
ax.legend()
ax.set_xlabel('epoch')
ax.set_ylabel('loss (MAE)')
plt.yscale('log')
plt.show()

