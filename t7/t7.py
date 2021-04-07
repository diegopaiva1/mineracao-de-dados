import pandas as pd
import numpy as np
import pydotplus
import matplotlib.pyplot as plt 
from random import randint
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import GridSearchCV

X = pd.read_csv('trabalho7_dados_01.csv')
y = X.pop('classe')
X = pd.DataFrame(MinMaxScaler().fit_transform(X), columns = X.columns)

def grid_search(estimator, params = dict()):
    """ Return tuple (best estimator, best score) by performing grid search. """
    clf = GridSearchCV(estimator, params, n_jobs = -1, cv = 10, scoring = 'accuracy')
    clf.fit(X, y)
    return clf.best_estimator_, clf.best_score_

def compare_classifiers(scores):
    """ Produce png output comparing all classifiers based on their scores. """
    fig, ax = plt.subplots()
    ax.set_title('Evaluation of classifiers using stratified 10-fold')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 105)

    color_list = []

    for _ in range(len(scores)):
        rgb = (randint(0, 255)/255.0, randint(0, 255)/255.0, randint(0, 255)/255.0)
        color_list.append(rgb)

    rects = ax.bar(
        x = ['kNN', 'Logistic', 'Naïve Bayes', 'Decision tree'], 
        height = [x * 100.0 for x in scores], 
        color = color_list
    )

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

    fig.savefig('classifiers_histogram.png')

knn_model, knn_score = grid_search(KNeighborsClassifier(), params = {
    'n_neighbors': range(1, 151),
    'weights': ['uniform', 'distance']
})

print('k = {}, weights = {}'.format(knn_model.n_neighbors, knn_model.weights))

log_model, log_score = grid_search(LogisticRegression(), params = {'max_iter': [1000]})

nb_model, nb_score = grid_search(GaussianNB())

dt_model, dt_score = grid_search(DecisionTreeClassifier(), params = {
    'max_depth': range(1, 14),
    'criterion': ['gini', 'entropy']
})

print('max_depth = {}, criterion = {}'.format(dt_model.max_depth, dt_model.criterion))

compare_classifiers([knn_score, log_score, nb_score, dt_score])

FILE_NAME = 'tree'

export_graphviz(
    dt_model, 
    out_file = FILE_NAME + '.dot', 
    filled = True, 
    rounded = True, 
    special_characters = True,
    feature_names = X.columns,
    class_names = ['0', '1']
)

graph = pydotplus.graph_from_dot_file(FILE_NAME + '.dot')
graph.write_png(FILE_NAME + '.png')

X_pred = pd.read_csv('trabalho7_dados_teste.csv')
X_pred = pd.DataFrame(MinMaxScaler().fit_transform(X_pred), columns = X_pred.columns)
rows = []

for i, row in X_pred.iterrows():
    sample = np.array(X_pred.loc[i]).reshape(1, -1)

    rows.append([
        knn_model.predict(sample)[0],
        log_model.predict(sample)[0],
        nb_model.predict(sample)[0],
        dt_model.predict(sample)[0],
    ])

np.savetxt('results.csv', rows, delimiter = ',', fmt = '%d', header = 'knn,logistic,nb,dt')