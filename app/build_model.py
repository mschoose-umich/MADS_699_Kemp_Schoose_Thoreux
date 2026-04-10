import pickle
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Split the data into features and target
Xy_train = pd.read_csv('data/Xy_train.csv')

features = ['QB_diff',
            'OL_adv_1',
            'OL_adv_2',
            'WR_adv_1',
            'WR_adv_2',
            'RB_adv_1',
            'RB_adv_2',
            'TE_diff',
            'PK_diff',
            'P_diff',
            'QB1_vs_DB_2',
            'QB2_vs_DB_1',
            'PASS_SYNERGY_diff',
            'team_1_is_home']

X_train = Xy_train[features]
y_train = Xy_train['target']

# Build the model pipeline
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", GradientBoostingClassifier(
        subsample= 0.8,
        n_estimators= 100,
        min_samples_split= 5,
        min_samples_leaf= 4,
        max_depth= 3,
        learning_rate= 0.1,
        random_state=42)
        )
])

# Fit the model
pipe.fit(X_train, y_train)

# Save the model
with open("data/production_model.pkl", "wb") as f:
    pickle.dump(pipe, f)
