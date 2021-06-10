import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


# Reference: https://github.com/wikibook/ml-definitive-guide/blob/master/7%EC%9E%A5/7-2_Cluster%20evaluation.ipynb

def visualize_silhouette(cluster_lists, X_features, args): 
  n_cols = len(cluster_lists)
  
  fig, axs = plt.subplots(figsize=(4*n_cols, 4), nrows=1, ncols=n_cols)
  
  for ind, n_cluster in enumerate(cluster_lists):
    clusterer = KMeans(n_clusters=n_cluster, max_iter=args.kmeans_max_iter, random_state=args.seed)
    cluster_labels = clusterer.fit_predict(X_features)
    
    sil_avg = silhouette_score(X_features, cluster_labels)
    sil_values = silhouette_samples(X_features, cluster_labels)
    
    y_lower = 10
    axs[ind].set_title('Number of Cluster : '+ str(n_cluster)+'\n' \
                      'Silhouette Score :' + str(round(sil_avg,3)) )
    axs[ind].set_xlabel("The silhouette coefficient values")
    axs[ind].set_ylabel("Cluster label")
    axs[ind].set_xlim([-0.1, 1])
    axs[ind].set_ylim([0, len(X_features) + (n_cluster + 1) * 10])
    axs[ind].set_yticks([])  # Clear the yaxis labels / ticks
    axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    for i in range(n_cluster):
      ith_cluster_sil_values = sil_values[cluster_labels==i]
      ith_cluster_sil_values.sort()
      
      size_cluster_i = ith_cluster_sil_values.shape[0]
      y_upper = y_lower + size_cluster_i
      
      color = cm.nipy_spectral(float(i) / n_cluster)
      axs[ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, \
                          facecolor=color, edgecolor=color, alpha=0.7)
      axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
      y_lower = y_upper + 10
        
    axs[ind].axvline(x=sil_avg, color="red", linestyle="--")

def visualize_history(history):
    y_loss = history[0]
    y_vloss = history[1]
    x_len = np.arange(len(y_loss))
    plt.plot(x_len, y_vloss, marker='.', c='red', label='valid_loss')
    plt.plot(x_len, y_loss, marker='.', c='blue', label='train_loss')
    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
