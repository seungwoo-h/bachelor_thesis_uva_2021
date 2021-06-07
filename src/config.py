import torch

# Configurations

class CFG:
  # General options
  train_data_path = '/content/drive/MyDrive/thesis/code/data/sberbank-russian-housing-market/train.csv'
  test_data_path = '/content/drive/MyDrive/thesis/code/data/sberbank-russian-housing-market/test.csv'
  sample_submission_path = '/content/drive/MyDrive/thesis/code/data/sberbank-russian-housing-market/sample_submission.csv'
  seed = 42

  # Training options
  num_cv_split = 5

  # Feature selection options
  use_top_n_features = 20

  # K-means options
  kmeans_max_iter = 300
  kmeans_n_clusters = 8

  # Autoencoder options
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  ae_val_size = 0.1
  ae_learning_rate = 3e-4
  ae_batch_size = 32
  ae_num_workers = 4
  ae_hidden_dim = 8
  ae_num_epochs = 500
  ae_checkpoint_dir = '/content/drive/MyDrive/thesis/code/checkpoints'
  ae_load_model_dir = '/content/drive/MyDrive/thesis/code/checkpoints/model-457.bin' # None

  # PCA options
  pca_n_components = 8