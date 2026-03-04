# Titanic Competition-Ready ML Pipeline (scikit-learn)

A reproducible Kaggle-style ML pipeline with:
- Feature engineering (FamilySize, IsAlone, Title extraction)
- Preprocessing with sklearn `ColumnTransformer` + `Pipeline`
- 5-fold Stratified CV
- Quick hyperparameter tuning with `RandomizedSearchCV`
- Submission file generation

## Setup
```bash
pip install -r requirements.txt
```

## Execute
```bash
python -m src.train
```