import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .autoencoder import AutoencoderDataset, autoencode

# Inference

def cv_ensemble(models, test_data, selected_features=None):
    if selected_features is not None:
      test_data = test_data[selected_features]
    results = []
    for model in models:
        pred = model.predict(test_data)
        pred = np.abs(pred)
        results.append(pred.tolist())
    results = np.array(results).mean(axis=0)
    return results

def cv_ensemble_kmeans(models, clusterers, test_data, selected_features, excluded_features):
    results = []
    for model, clusterer in zip(models, clusterers):
      test_data_selected = test_data[selected_features].copy()
      test_data_selected['cluster'] = clusterer.predict(test_data[excluded_features])
      test_data_selected = pd.get_dummies(test_data_selected, columns=['cluster'])
      pred = model.predict(test_data_selected)
      pred = np.abs(pred)
      results.append(pred.tolist())
    results = np.array(results).mean(axis=0)
    return results

def cv_ensemble_ae(models, scalers, test_data, selected_features, excluded_features, encoder_model):
    results = []
    for model, scaler in zip(models, scalers):
      test_data_selected, test_data_unselected = test_data[selected_features].copy(), test_data[excluded_features].copy()
      test_data_unselected = scaler.transform(test_data_unselected)
      test_enc_dataset = AutoencoderDataset(test_data_unselected)
      test_enc_loader = DataLoader(test_enc_dataset, batch_size=16, shuffle=False, num_workers=4)
      out_test_enc = autoencode(test_enc_loader, encoder_model)
      test_data_selected = np.concatenate([test_data_selected, out_test_enc], axis=1)

      pred = model.predict(test_data_selected)
      pred = np.abs(pred)
      results.append(pred.tolist())
    results = np.array(results).mean(axis=0)
    return results

def cv_ensemble_pca(models, scalers, pca_models, test_data, selected_features, excluded_features):
    results = []
    for model, scaler, pca in zip(models, scalers, pca_models):
      test_data_selected, test_data_unselected = test_data[selected_features].copy(), test_data[excluded_features].copy()
      test_data_unselected = scaler.transform(test_data_unselected)
      test_data_unselected = pca.transform(test_data_unselected)
      test_data_selected = np.concatenate([test_data_selected, test_data_unselected], axis=1)

      pred = model.predict(test_data_selected)
      pred = np.abs(pred)
      results.append(pred.tolist())
    results = np.array(results).mean(axis=0)
    return results
