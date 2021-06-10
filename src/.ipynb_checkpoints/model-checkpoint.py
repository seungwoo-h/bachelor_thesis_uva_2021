from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

def get_model(args):
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