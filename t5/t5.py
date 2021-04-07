import pandas as pd
import numpy as np
from sklearn import preprocessing, cluster, decomposition, neighbors
from matplotlib import pyplot as plt

def plot_clusters(df, label, figname, algorithm_name):
    df['label'] = label

    fig, ax = plt.subplots()
    ax.set_title('{} clustering - {} samples'.format(algorithm_name, len(df)))
    ax.set_xlabel('1st principal component')
    ax.set_ylabel('2nd principal component')
    colors = ['r', 'g', 'c', 'm', 'y', 'b'] 

    for i, cluster in df.groupby('label'):
        ax.scatter(
            cluster['pc1'], 
            cluster['pc2'], 
            c = colors[i],
            alpha = 0.5,
            label = str(len(cluster)) + ' elements',
        )

    ax.legend(bbox_to_anchor = (1.33, 1.02), shadow = True)
    fig.savefig(figname, bbox_inches = 'tight')
    return

scaler = preprocessing.StandardScaler()
df = pd.read_csv('./trabalho5_dados_01.csv')
print(df)
df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)

# Apply k-means algorithm
label_kmeans = cluster.MiniBatchKMeans(n_clusters = 2, random_state = 0).fit_predict(df)

# Apply AGNES algorithm
label_agnes = cluster.AgglomerativeClustering(
    n_clusters = 2, affinity = 'euclidean', linkage = 'ward'
).fit_predict(df)

# Apply DBSCAN algorithm
label_dbscan = cluster.DBSCAN(eps = 0.6, min_samples = 8, metric = 'euclidean').fit_predict(df)

# Dimensionality reduction
pca = decomposition.PCA(n_components = 2)
pca_df = pd.DataFrame(pca.fit_transform(df), columns = ['pc1', 'pc2'])

# Plotar os dados originais
plt.scatter(pca_df['pc1'], pca_df['pc2'], c = 'black', alpha = 0.5)
plt.xlabel('1st principal component')
plt.ylabel('2nd principal component')
plt.savefig('./scatter_data.png')

plot_clusters(pca_df, label_kmeans, './kmeans_clusters.png', 'k-means')
plot_clusters(pca_df, label_agnes, './agnes_clusters.png', 'AGNES')
plot_clusters(pca_df, label_dbscan, './dbscan_clusters.png', 'DBSCAN')

# Join labels
df = pd.DataFrame(scaler.inverse_transform(df), columns = df.columns)
df['label_kmeans'] = label_kmeans
df['label_agnes'] = label_agnes
df['label_dbscan'] = label_dbscan

# Show mean values by cluster
print(np.around(df.groupby('label_kmeans').agg(['mean', 'count']), decimals = 2), end = '\n\n')
print(np.around(df.groupby('label_agnes').agg(['mean', 'count']), decimals = 2), end = '\n\n')
print(np.around(df.groupby('label_dbscan').agg(['mean', 'count']), decimals = 2))