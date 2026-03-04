import os
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

from sklearn.ensemble import HistGradientBoostingClassifier

from .features import add_features
from .eda import run_eda


RANDOM_STATE = 42


def load_data():
    train_path = os.path.join("data", "train.csv")
    test_path = os.path.join("data", "test.csv")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def build_pipeline():
    # After feature engineering these columns will exist
    features = [
        "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
        "FamilySize", "IsAlone", "Title"
    ]
    target = "Survived"

    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
    categorical_features = ["Sex", "Embarked", "Title"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = HistGradientBoostingClassifier(random_state=RANDOM_STATE)

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return clf, features, target


def main():
    train_df, test_df = load_data()

    # Feature engineering
    train_df = add_features(train_df)
    test_df = add_features(test_df)

    # Quick EDA outputs (optional but nice)
    run_eda(train_df)

    clf, features, target = build_pipeline()
    X = train_df[features]
    y = train_df[target]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    base_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    print(f"Baseline CV accuracy: mean={base_scores.mean():.4f} std={base_scores.std():.4f}")

    # Quick randomized hyperparameter search
    param_dist = {
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [2, 3, 4, None],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [10, 20, 30],
        "model__l2_regularization": [0.0, 0.1, 1.0],
    }

    search = RandomizedSearchCV(
        clf,
        param_distributions=param_dist,
        n_iter=20,
        scoring="accuracy",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X, y)
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    best_model = search.best_estimator_

    # Train on full data
    best_model.fit(X, y)

    test_X = test_df[features]
    test_pred = best_model.predict(test_X)

    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": test_pred.astype(int)
    })

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "submission.csv")
    submission.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()