import warnings
import argparse
import os
import pandas as pd
import numpy as np
import torch

from src.preprocessor import basic_preprocess, test_preprocess, select_features
from src.model import get_model, get_autoencoder
from src.metric import rmsle
from src.train import train_cv, train_pca_cv, train_kmeans_cv, train_ae_cv
from src.inference import cv_ensemble, cv_ensemble_pca, cv_ensemble_kmeans, cv_ensemble_ae

warnings.filterwarnings('ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Set Tasks
    parser.add_argument("--feature_selection_method", type=str, default="none") # Option: none, wrapper, filter, embedded
    parser.add_argument("--feature_extraction_method", type=str, default="none") # Option: none, pca, kmeans, autoencoder
    parser.add_argument("--base_model", type=str, default="lgbm") # Option: linear_reg, random_forest, lgbm

    # General Options
    parser.add_argument("--extract_dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_data_path", type=str, default="./data/sberbank-russian-housing-market/train.csv")
    parser.add_argument("--test_data_path", type=str, default="./data/sberbank-russian-housing-market/test.csv")
    parser.add_argument("--sample_submission_path", type=str, default="./data/sberbank-russian-housing-market/sample_submission.csv")
    parser.add_argument("--submission_path", type=str, default="./output/")

    # Training Options
    parser.add_argument("--num_cv_split", type=int, default=5)
    parser.add_argument("--use_top_n_features", type=int, default=20)

    # K-Means Options
    parser.add_argument("--kmeans_max_iter", type=int, default=300)
    parser.add_argument("--kmeans_n_clusters", type=int, default=8)

    # Autoencoder Options
    parser.add_argument("--ae_train", type=str, default="false")
    parser.add_argument("--ae_val_size", type=float, default=0.1)
    parser.add_argument("--ae_learning_rate", type=float, default=3e-4)
    parser.add_argument("--ae_batch_size", type=int, default=32)
    parser.add_argument("--ae_num_workers", type=int, default=4)
    parser.add_argument("--ae_hidden_dim", type=int, default=8)
    parser.add_argument("--ae_num_epochs", type=int, default=500)
    parser.add_argument("--ae_checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--ae_load_model_dir", type=str, default="./checkpoints/model-457.bin")

    # PCA Options
    parser.add_argument("--pca_n_components", type=int, default=8)

    args = parser.parse_args()
    args.device = device

    if not os.path.exists(args.submission_path):
        os.mkdir(args.submission_path)
        
    # Load and process data
    train_data = pd.read_csv(args.train_data_path)
    train_data = basic_preprocess(train_data, args)
    X = train_data.drop('price_doc', axis=1)
    y = train_data['price_doc']

    args.processed_dim = len(X.columns)

    print(f"Start Experiment. - Base Model: {args.base_model}")
    print(f"Feature Selection Method: {args.feature_selection_method} | Feature Extraction Method: {args.feature_extraction_method}")

    # Get pipeline
    pipeline = get_model(args)
    
    # Feature selection
    if args.feature_selection_method == "none":
        print("No feature selection.")

    elif args.feature_selection_method == "embedded":
        print("Processing embedded feature selection.")
        fs_base_models_, _ = train_cv(X, y, get_model(args, True), args)
        selected_features, excluded_features = select_features(fs_base_models_, train_data, args)
        X_selected = X[selected_features] 
        X_unselected = X[excluded_features]

    elif args.feature_selection_method == "filter":
        print("Processing filter feature selection.")
        selected_features, excluded_features = select_features(None, train_data, args)
        X_selected = X[selected_features] 
        X_unselected = X[excluded_features]

    # Feature extraction & Train base model
    if args.feature_selection_method == "none":
        print("No feature extraction.")
        models_, _ = train_cv(X, y, pipeline, args)

    elif args.feature_extraction_method == "none":
        print("No feature extraction. Using only selected features.")
        models_, _ = train_cv(X_selected, y, pipeline, args)

    if args.feature_extraction_method == "pca":
        print("Feature extraction with PCA, training...")
        models_, scalers_, pcas_, _ = train_pca_cv(X, y, pipeline, selected_features, excluded_features, args)

    elif args.feature_extraction_method == "kmeans":
        print("Feature extraction with K-Means, training...")
        models_, clusterers_, _ = train_kmeans_cv(X, y, pipeline, selected_features, excluded_features, args)

    elif args.feature_extraction_method == "autoencoder":
        print("Feature extraction with Autoencoder, training...")
        ae_model = get_autoencoder(args)
        models_, scalers_, _ = train_ae_cv(X, y, pipeline, selected_features, excluded_features, ae_model, args)

    # Inference

    test_data = pd.read_csv(args.test_data_path)
    test_data = test_preprocess(test_data, args)

    if args.feature_selection_method == "none":
        print("Inferencing without feature selection.")
        test_pred = cv_ensemble(models_, test_data)

    elif args.feature_extraction_method == "none":
        print("Inferencing with selected features only. No feature extraction.")
        test_pred = cv_ensemble(models_, test_data, selected_features)
        
    if args.feature_extraction_method == "pca":
        print("Inferencing with selected features only. PCA feature extraction.")
        test_pred = cv_ensemble_pca(models_, scalers_, pcas_, test_data, selected_features, excluded_features)

    elif args.feature_extraction_method == "kmeans":
        print("Inferencing with selected features only. K-Means feature extraction.")
        test_pred = cv_ensemble_kmeans(models_, clusterers_, test_data, selected_features, excluded_features)

    elif args.feature_extraction_method == "autoencoder":
        print("Inferencing with selected features only. Autoencoder feature extraction.")
        test_pred = cv_ensemble_ae(models_, scalers_, test_data, selected_features, excluded_features, ae_model)

    # Save

    submission = pd.read_csv(args.sample_submission_path)
    submission['price_doc'] = test_pred
    sub_path = f"{args.submission_path}result_{args.feature_selection_method}_{args.feature_extraction_method}_{args.base_model}_seed{args.seed}.csv"
    submission.to_csv(sub_path, index=False)
    print("Saved. Done. \n")