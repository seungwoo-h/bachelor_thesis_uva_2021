# Bachelor's Thesis 2021
---
Source code for experiments.

https://www.kaggle.com/c/sberbank-russian-housing-market/overview

```bash
$ git clone https://github.com/seungwoo-h/bachelor_thesis_uva_2021.git
```
Usage: run.py will train a new model from a given training dataset and generate kaggle submission file.

### To run all experiments
```bash
$ sh run_experiments.sh
```

### To run a single experiment
```bash
$ python run.py --feature_selection_method [METHOD1] --feature_extraction_method [METHOD2] --base_model [MODEL]
```
