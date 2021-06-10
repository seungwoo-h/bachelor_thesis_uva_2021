#!/bin/bash

python ./run.py --feature_selection_method none --feature_extraction_method none --base_model linear_reg
# python ./run.py --feature_selection_method none --feature_extraction_method none --base_model random_forest
python ./run.py --feature_selection_method none --feature_extraction_method none --base_model lgbm


python ./run.py --feature_selection_method filter --feature_extraction_method none --base_model linear_reg
python ./run.py --feature_selection_method filter --feature_extraction_method pca --base_model linear_reg
python ./run.py --feature_selection_method filter --feature_extraction_method kmeans --base_model linear_reg
# python ./run.py --feature_selection_method filter --feature_extraction_method autoencoder --base_model linear_reg

# python ./run.py --feature_selection_method filter --feature_extraction_method none --base_model random_forest
# python ./run.py --feature_selection_method filter --feature_extraction_method pca --base_model random_forest
# python ./run.py --feature_selection_method filter --feature_extraction_method kmeans --base_model random_forest
# python ./run.py --feature_selection_method filter --feature_extraction_method autoencoder --base_model random_forest

python ./run.py --feature_selection_method filter --feature_extraction_method none --base_model lgbm
python ./run.py --feature_selection_method filter --feature_extraction_method pca --base_model lgbm
python ./run.py --feature_selection_method filter --feature_extraction_method kmeans --base_model lgbm
# python ./run.py --feature_selection_method filter --feature_extraction_method autoencoder --base_model lgbm


python ./run.py --feature_selection_method embedded --feature_extraction_method none --base_model linear_reg
python ./run.py --feature_selection_method embedded --feature_extraction_method pca --base_model linear_reg
python ./run.py --feature_selection_method embedded --feature_extraction_method kmeans --base_model linear_reg
# python ./run.py --feature_selection_method embedded --feature_extraction_method autoencoder --base_model linear_reg

# python ./run.py --feature_selection_method embedded --feature_extraction_method none --base_model random_forest
# python ./run.py --feature_selection_method embedded --feature_extraction_method pca --base_model random_forest
# python ./run.py --feature_selection_method embedded --feature_extraction_method kmeans --base_model random_forest
# python ./run.py --feature_selection_method embedded --feature_extraction_method autoencoder --base_model random_forest

python ./run.py --feature_selection_method embedded --feature_extraction_method none --base_model lgbm
python ./run.py --feature_selection_method embedded --feature_extraction_method pca --base_model lgbm
python ./run.py --feature_selection_method embedded --feature_extraction_method kmeans --base_model lgbm
python ./run.py --feature_selection_method embedded --feature_extraction_method autoencoder --base_model lgbm
