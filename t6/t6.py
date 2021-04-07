import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Divide dataframe into two for training and testing
df = pd.read_csv('trabalho6_dados_01.csv')
train_df, test_df = train_test_split(df, test_size = 0.3, random_state = 0)

def get_best_knn_n_neighbors(low, high):
    """ Return the best number of neighbors in range [low, high] to be used in knn algorithm. """
    n_neighbors = []
    mae_knn = []

    for n in range(low, high + 1):
        knn_regressor = KNeighborsRegressor(n_neighbors = n, weights = 'distance')
        knn_regressor.fit(train_df[['temperatura', 'vacuo']], train_df[['energia']])
        energia_knn = knn_regressor.predict(test_df[['temperatura', 'vacuo']])
        
        n_neighbors.append(n)
        mae_knn.append(metrics.mean_absolute_error(test_df['energia'], energia_knn))

    best_n_neighbors = n_neighbors[np.argmin(mae_knn)]

    fig, ax = plt.subplots()
    ax.set_title('Parameter evaluation for KNN')
    ax.set_xlabel('Number of neighbors')
    ax.set_ylabel('Mean absolute error')
    ax.set_xlim(low - 1, high)
    ax.set_xticks(list(ax.get_xticks()) + [best_n_neighbors])
    ax.plot(n_neighbors, mae_knn, linewidth = 2)
    fig.savefig('knn_param.png')

    return best_n_neighbors

def get_best_rnn_radius(low, high, step):
    """ Return the best radius value in step range [low, high] to be used in rnn algorithm. """
    radii = []
    mae_rnn = []

    for r in np.arange(low, high + step, step):
        rnn_regressor = RadiusNeighborsRegressor(radius = r, weights = 'distance')
        rnn_regressor.fit(train_df[['temperatura', 'vacuo']], train_df[['energia']])
        energia_rnn = rnn_regressor.predict(test_df[['temperatura', 'vacuo']])

        radii.append(r)
        mae_rnn.append(metrics.mean_absolute_error(test_df['energia'], energia_rnn))

    best_radius = radii[np.argmin(mae_rnn)]

    fig, ax = plt.subplots()
    ax.set_title('Parameter evaluation for RNN')
    ax.set_xlabel('Radius')
    ax.set_ylabel('Mean absolute error')
    ax.set_xlim(low, high)
    ax.set_xticks(list(ax.get_xticks()) + [best_radius])
    ax.plot(radii, mae_rnn, c = 'orange', linewidth = 2)
    fig.savefig('rnn_param.png')

    return best_radius

knn_regressor = KNeighborsRegressor(n_neighbors = get_best_knn_n_neighbors(1, 100), weights = 'distance')
knn_regressor.fit(train_df[['temperatura', 'vacuo']], train_df[['energia']])

rnn_regressor = RadiusNeighborsRegressor(radius = get_best_rnn_radius(1.7, 3.0, 0.05), weights = 'distance')
rnn_regressor.fit(train_df[['temperatura', 'vacuo']], train_df[['energia']])

lr_regressor = LinearRegression()
lr_regressor.fit(train_df[['temperatura', 'vacuo']], train_df[['energia']])

energia_knn = knn_regressor.predict(test_df[['temperatura', 'vacuo']])
energia_rnn = rnn_regressor.predict(test_df[['temperatura', 'vacuo']])
energia_lr = lr_regressor.predict(test_df[['temperatura', 'vacuo']])

fig, ax = plt.subplots()
ax.set_title('Evaluation of regression algorithms')
ax.set_ylabel('Mean absolute error')
ax.set_ylim(0, 5)
ax.set_yticks(np.arange(0, 5, 1.5))

rects = ax.bar(x = ['kNN', 'rNN', 'LR'], height = [
    metrics.mean_absolute_error(test_df['energia'], energia_knn),
    metrics.mean_absolute_error(test_df['energia'], energia_rnn),
    metrics.mean_absolute_error(test_df['energia'], energia_lr),
], color = ['darkred', 'darkgreen', 'darkblue'])

for rect in rects:
    height = rect.get_height()
    ax.annotate(
        '{:.2f}'.format(height),
        xy = (rect.get_x() + rect.get_width()/2, height),
        xytext = (0, 3),
        textcoords = 'offset points',
        ha = 'center', 
        va = 'bottom'
    )

fig.savefig('regressor_histogram.png')

predict_df = pd.read_csv('trabalho6_teste.csv')
rows = []

for i, row in predict_df.iterrows():
    rows.append([
        row['temperatura'], 
        row['vacuo'],
        knn_regressor.predict(np.array(row).reshape(1, -1))[0][0],
        rnn_regressor.predict(np.array(row).reshape(1, -1))[0][0],
        lr_regressor.predict(np.array(row).reshape(1, -1))[0][0]
    ])

np.savetxt('predict_results.csv', rows, delimiter = ',', fmt = '%.2f', header = 'temperatura,vacuo,energia_knn,energia_rnn,energia_lr')