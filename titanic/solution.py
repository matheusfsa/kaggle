import pandas as pd

from sklearn.linear_model import Lasso


def predict_numerical_nan_values(data, feature, id_col):
    train = data[data[feature].notna()]
    test = data[data[feature].isna()].drop(columns=[feature, id_col])
    y = train[feature]
    notna_cols = (data.drop(columns=[feature, id_col]).isna().sum() == 0).values
    num_data = pd.get_dummies(data.iloc[:, notna_cols])

    X = num_data[num_data[feature].notna()].drop(columns=[feature, id_col])

    test = num_data[num_data[feature].isna()].drop(columns=[feature, id_col])
    print((test.isna().sum() == 0))
    model = Lasso().fit(X, y)
    preds = model.predict(test)
    data.loc[data[feature].isna(), feature] = preds
    return data