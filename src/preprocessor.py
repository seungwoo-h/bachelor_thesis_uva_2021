import pandas as pd

def basic_preprocess(data, args):
  # Handle nans
  _dropped_cols = []
  for col in data.columns:
      # Drop every columns which are 'cafe' related
      if col.split('_')[0] == 'cafe':
          data.drop(col, axis=1, inplace=True)
          _dropped_cols.append(col)
      # Drop column if > 1000
      elif data[col].isna().sum() > 1000:
          data.drop(col, axis=1, inplace=True)
          _dropped_cols.append(col)
  # Drop rest axis=0        
  data.dropna(axis=0, inplace=True)
  # Drop id
  data = data.drop(['id'], axis=1)

  # Handle timestamp
  data['timestamp'] = pd.to_datetime(data['timestamp'], format="%Y-%m-%d")
  data['timestamp_year'] = data['timestamp'].dt.year
  data['timestamp_month'] = data['timestamp'].dt.month
  data = data.drop(['timestamp'], axis=1)

  # One hot encode
  label_cols = [col for col in data.columns if data[col].dtype == object]
  label_cols.extend(['timestamp_year', 'timestamp_month'])
  data = pd.get_dummies(data, columns=label_cols)

  # Order
  train_columns = sorted(data.columns)
  data = data[train_columns]
  train_columns.remove('price_doc')
   
  # Update args
  args.deleted_columns = _dropped_cols
  args.train_columns = train_columns
  return data

def test_preprocess(data, args):
  # Handle nans
  data = data.drop(args.deleted_columns, axis=1)
  for col in data.columns:
    if data[col].isna().sum() > 0:
      if data[col].dtype == object:
        data[col] = data[col].fillna(data[col].mode()[0])
      else:
        data[col] = data[col].fillna(data[col].mean())
  # Handle timestamp
  data['timestamp'] = pd.to_datetime(data['timestamp'], format="%Y-%m-%d")
  data['timestamp_year'] = data['timestamp'].dt.year
  data['timestamp_month'] = data['timestamp'].dt.month
  data = data.drop(['id', 'timestamp'], axis=1)

  # One hot encode
  label_cols = [col for col in data.columns if data[col].dtype == object]
  label_cols.extend(['timestamp_year', 'timestamp_month'])
  data = pd.get_dummies(data, columns=label_cols)
  _missing_cols = list(set(args.train_columns) - set(data.columns))
  # _missing_cols.remove('price_doc')
  for col in _missing_cols:
    data[col] = 0

  # Order
  data = data[args.train_columns]
  return data

def select_features(base_model_pipelines, args):
    TOP_N = args.use_top_n_features
    # Get average feature importance from lgbm
    for i, pipeline in enumerate(base_model_pipelines):
        model = pipeline['regressor']
        if i == 0:
            fi = model.feature_importances_.copy()
            continue
        fi += model.feature_importances_.copy()
    fi = fi / args.num_cv_split
    feature_imp = pd.DataFrame(sorted(zip(fi, X.columns)), columns=['Value','Feature'])
    selected_features_w_imp = feature_imp.sort_values(by="Value", ascending=False)[:TOP_N] # TOP N
    selected_features = selected_features_w_imp['Feature'].to_list()
    excluded_features = list(set(X) - set(selected_features))
    return selected_features, excluded_features