# Bachelor's Thesis 2021
---
Source code for experiments.

### Dataset
https://www.kaggle.com/c/sberbank-russian-housing-market/overview

---

## Usage

```bash
$ git clone https://github.com/seungwoo-h/bachelor_thesis_uva_2021.git
```
run.py will train a new model from a given training dataset and generate kaggle submission file.

### To run all experiments
- 13 Dataset
```bash
$ sh run_experiments.sh
```

### To run a single experiment
```bash
$ python run.py --feature_selection_method [METHOD1] --feature_extraction_method [METHOD2] --base_model [MODEL]
```
---

## Experimental Results
*From Kaggle Private Leaderboard*
| Dataset           |                       | Linear Regression | Random Forest | LightGBM |
|-------------------|-----------------------|-------------------|---------------|----------|
| Original Dataset  |                       | 0.81583           | 0.33617       | 0.33011  |
| Feature Selection | Feature Extraction    |                   |               |          |
| Wrapper Methods   | No Feature Extraction | 0.4158            | 0.33605       | 0.32494  |
|                   | PCA                   | 0.41444           | 0.32817       | 0.32608  |
|                   | K-Means               | 0.42679           | 0.33613       | 0.32515  |
|                   | Autoencoder           | 0.41362           | 0.33017       | 0.32522  |
| Filter Method     | No Feature Extraction | 0.40262           | 0.34416       | 0.32711  |
|                   | PCA                   | 0.42417           | 0.3311        | 0.32843  |
|                   | K-Means               | 0.40315           | 0.34451       | 0.32861  |
|                   | Autoencoder           | 0.40931           | 0.33175       | 0.33089  |
| Embedded Method   | No Feature Extraction | 0.4133            | 0.33513       | 0.32532  |
|                   | PCA                   | 0.41622           | 0.32839       | 0.32506  |
|                   | K-Means               | 0.40897           | 0.33503       | 0.32545  |
|                   | Autoencoder           | 0.4151            | 0.33026       | 0.32838  |
