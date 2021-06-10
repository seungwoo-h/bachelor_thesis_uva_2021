from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from lightgbm import LGBMRegressor
from .autoencoder import Autoencoder

def get_model(args, feature_selection=False):
    if feature_selection:
        if args.feature_selection_method == 'embedded':
            pipeline_lgbm = Pipeline([
                ('scaler', RobustScaler()),
                ('regressor', LGBMRegressor(random_state=args.seed)),
                ])
            return pipeline_lgbm
        elif args.feature_selection_method == 'wrapper':
            pipeline_rfe = Pipeline([
                ('scaler', RobustScaler()),
                ('rfe', RFE(
                    estimator=LGBMRegressor(random_state=args.seed), 
                    n_features_to_select=args.use_top_n_features, 
                    step=args.rfe_step_size, 
                    verbose=2)),
                ('regressor', LGBMRegressor(random_state=args.seed)),
                ])
            return pipeline_rfe
        
    if args.base_model == "lgbm":
        pipeline_lgbm = Pipeline([
            ('scaler', RobustScaler()),
            ('regressor', LGBMRegressor(random_state=args.seed)),
            ])
        return pipeline_lgbm

    elif args.base_model == "random_forest":
        pipeline_rf = Pipeline([
            ('scaler', RobustScaler()),
            ('regressor', RandomForestRegressor(random_state=args.seed)),
            ])
        return pipeline_rf
    
    elif args.base_model == "linear_reg":
        pipeline_lr = Pipeline([
            ('scaler', RobustScaler()),
            ('regressor', LinearRegression()),
            ])
        return pipeline_lr

def get_autoencoder(args):
    device = args.device
    input_dim = args.processed_dim - args.use_top_n_features
    ae_model = Autoencoder(input_dim=input_dim, z_dim=args.ae_hidden_dim)
    ae_model.to(device)
    return ae_model
