import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA

from .metric import rmsle
from .autoencoder import AutoencoderDataset

# K-Fold training

def train_cv(X_data, y_data, pipeline, args):
  X, y = X_data.to_numpy(), y_data.to_numpy()
  trained_models, scores = list(), list()
  # K-fold
  kf = KFold(n_splits=args.num_cv_split, random_state=args.seed)
  for iter, (train_index, valid_index) in enumerate(kf.split(X)):
    X_train, X_valid, y_train, y_valid = X[train_index], X[valid_index], y[train_index], y[valid_index]
    model = copy.deepcopy(pipeline)
    # Train
    model.fit(X_train, y_train)
    # Validate
    y_pred = model.predict(X_valid)
    y_pred = np.abs(y_pred)
    score = rmsle(y_pred, y_valid)
    # Save
    trained_models.append(model)
    scores.append(score)
    # Log
    print(f"[{iter+1}/{args.num_cv_split}] Train Size: {len(train_index)} | Val Size: {len(valid_index)} | RMSLE: {round(score, 3)}")
  print(f"Average RMSLE: {round(np.mean(scores), 2)} | CV RMSLE Std. Dev: {round(np.std(scores), 2)} \n")
  return trained_models, scores

# K-Fold training with Kmeans

def train_kmeans_cv(X_data, y_data, pipeline, selected_features, excluded_features, args):
  X, y = X_data.to_numpy(), y_data.to_numpy()
  trained_models, clusterers = list(), list()
  scores = list()
  # K-fold
  kf = KFold(n_splits=args.num_cv_split, random_state=args.seed)
  for iter, (train_index, valid_index) in enumerate(kf.split(X)):
    X_train, X_valid, y_train, y_valid = X[train_index], X[valid_index], y[train_index], y[valid_index]
    X_train, X_valid = pd.DataFrame(X_train, columns=X_data.columns), pd.DataFrame(X_valid, columns=X_data.columns)
    X_train_selected, X_train_unselected = X_train[selected_features], X_train[excluded_features]
    X_valid_selected, X_valid_unselected = X_valid[selected_features], X_valid[excluded_features]
    # Kmeans
    kmeans = KMeans(n_clusters=args.kmeans_n_clusters, init='k-means++', max_iter=args.kmeans_max_iter, random_state=args.seed)
    kmeans.fit(X_train_unselected)
    X_train_selected['cluster'] = kmeans.labels_
    X_valid_selected['cluster'] = kmeans.predict(X_valid_unselected)
    X_train_selected = pd.get_dummies(X_train_selected, columns=['cluster'])
    X_valid_selected = pd.get_dummies(X_valid_selected, columns=['cluster'])
    X_train_selected, X_valid_selected = X_train_selected.to_numpy(), X_valid_selected.to_numpy()  
    # Train
    model = copy.deepcopy(pipeline)
    model.fit(X_train_selected, y_train)
    # Validate
    y_pred = model.predict(X_valid_selected)
    y_pred = np.abs(y_pred) # for linear reg
    score = rmsle(y_pred, y_valid)
    # Save
    trained_models.append(model)
    clusterers.append(kmeans)
    scores.append(score)
    # Log
    print(f"[{iter+1}/{args.num_cv_split}] Train Size: {len(train_index)} | Val Size: {len(valid_index)} | RMSLE: {round(score, 3)}")
  print(f"Average RMSLE: {round(np.mean(scores), 2)} | CV RMSLE Std. Dev: {round(np.std(scores), 2)} \n")
  return trained_models, clusterers, scores

# K-Fold training with Autoencoder

def _encode(loader, encoder_model):
  outputs = []
  encoder_model.eval()
  with torch.no_grad():
    for batch in loader:
      x = batch.float().to(device)
      output = encoder_model.encode(x)
      outputs.append(output)
    outputs = torch.cat(outputs).cpu().numpy()
  return outputs

def train_ae_cv(X_data, y_data, pipeline, selected_features, excluded_features, encoder_model, args):
  X, y = X_data.to_numpy(), y_data.to_numpy()
  trained_models, scalers = list(), list()
  scores = list()
  # K-fold
  kf = KFold(n_splits=args.num_cv_split, random_state=args.seed)
  for iter, (train_index, valid_index) in enumerate(kf.split(X)):
    X_train, X_valid, y_train, y_valid = X[train_index], X[valid_index], y[train_index], y[valid_index]
    X_train, X_valid = pd.DataFrame(X_train, columns=X_data.columns), pd.DataFrame(X_valid, columns=X_data.columns)
    X_train_selected, X_train_unselected = X_train[selected_features], X_train[excluded_features]
    X_valid_selected, X_valid_unselected = X_valid[selected_features], X_valid[excluded_features]
    # Autoencoder
    autoencoder_scaler = MinMaxScaler()
    X_train_unselected = autoencoder_scaler.fit_transform(X_train_unselected)
    X_valid_unselected = autoencoder_scaler.transform(X_valid_unselected)
    train_enc_dataset = AutoencoderDataset(X_train_unselected)
    val_enc_dataset = AutoencoderDataset(X_valid_unselected)
    train_enc_loader = DataLoader(train_enc_dataset, batch_size=args.ae_batch_size, shuffle=False, num_workers=args.ae_num_workers)
    val_enc_loader = DataLoader(val_enc_dataset, batch_size=args.ae_batch_size, shuffle=False, num_workers=args.ae_num_workers)
    out_train_enc = _encode(train_enc_loader, encoder_model)
    out_val_enc = _encode(val_enc_loader, encoder_model)

    X_train_selected = np.concatenate([X_train_selected, out_train_enc], axis=1)
    X_valid_selected = np.concatenate([X_valid_selected, out_val_enc], axis=1)

    # Train
    model = copy.deepcopy(pipeline)
    model.fit(X_train_selected, y_train)
    # Validate
    y_pred = model.predict(X_valid_selected)
    y_pred = np.abs(y_pred) # for linear reg
    score = rmsle(y_pred, y_valid)
    # Save
    trained_models.append(model)
    scalers.append(autoencoder_scaler)
    scores.append(score)
    # Log
    print(f"[{iter+1}/{args.num_cv_split}] Train Size: {len(train_index)} | Val Size: {len(valid_index)} | RMSLE: {round(score, 3)}")
  print(f"Average RMSLE: {round(np.mean(scores), 2)} | CV RMSLE Std. Dev: {round(np.std(scores), 2)} \n")
  return trained_models, scalers, scores

# K-Fold training with PCA

def train_pca_cv(X_data, y_data, pipeline, selected_features, excluded_features, args):
  X, y = X_data.to_numpy(), y_data.to_numpy()
  trained_models, scalers, pca_models = list(), list(), list()
  scores = list()
  # K-fold
  kf = KFold(n_splits=args.num_cv_split, random_state=args.seed)
  for iter, (train_index, valid_index) in enumerate(kf.split(X)):
    X_train, X_valid, y_train, y_valid = X[train_index], X[valid_index], y[train_index], y[valid_index]
    X_train, X_valid = pd.DataFrame(X_train, columns=X_data.columns), pd.DataFrame(X_valid, columns=X_data.columns)
    X_train_selected, X_train_unselected = X_train[selected_features], X_train[excluded_features]
    X_valid_selected, X_valid_unselected = X_valid[selected_features], X_valid[excluded_features]
    # PCA
    pca_scaler = MinMaxScaler()
    X_train_unselected = pca_scaler.fit_transform(X_train_unselected)
    X_valid_unselected = pca_scaler.transform(X_valid_unselected)
    pca = PCA(n_components=args.pca_n_components, random_state=args.seed)
    pca_features_train = pca.fit_transform(X_train_unselected)
    pca_features_valid = pca.transform(X_valid_unselected)

    X_train_selected = np.concatenate([X_train_selected, pca_features_train], axis=1)
    X_valid_selected = np.concatenate([X_valid_selected, pca_features_valid], axis=1)

    # Train
    model = copy.deepcopy(pipeline)
    model.fit(X_train_selected, y_train)
    # Validate
    y_pred = model.predict(X_valid_selected)
    y_pred = np.abs(y_pred) # for linear reg
    score = rmsle(y_pred, y_valid)
    # Save
    trained_models.append(model)
    scalers.append(pca_scaler)
    pca_models.append(pca)
    scores.append(score)
    # Log
    print(f"[{iter+1}/{args.num_cv_split}] Train Size: {len(train_index)} | Val Size: {len(valid_index)} | RMSLE: {round(score, 3)}")
  print(f"Average RMSLE: {round(np.mean(scores), 2)} | CV RMSLE Std. Dev: {round(np.std(scores), 2)} \n")
  return trained_models, scalers, pca_models, scores