import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler


def outliers(X, treshold):
    z = (X - X.mean())/X.std()
    return z > treshold


def preprocessing():
    train = pd.read_csv("/home/matheus/PycharmProjects/HousePrice/data/train.csv")
    train = train[outliers(train['SalePrice'], 3) == False]
    test = pd.read_csv("/home/matheus/PycharmProjects/HousePrice/data/test.csv")
    all_data = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                          test.loc[:, 'MSSubClass':'SaleCondition']))
    all_data['PoolQC'] = all_data['PoolQC'].fillna('No')
    all_data['MiscFeature'] = all_data['MiscFeature'].fillna('No')
    all_data['Alley'] = all_data['Alley'].fillna('No')
    all_data['Fence'] = all_data['Fence'].fillna('No')
    all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('No')
    all_data['GarageCond'] = all_data['GarageCond'].fillna('No')
    all_data['GarageQual'] = all_data['GarageQual'].fillna('No')
    all_data['GarageFinish'] = all_data['GarageFinish'].fillna('No')
    all_data['GarageType'] = all_data['GarageType'].fillna('No')
    all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna(0)
    all_data['BsmtCond'] = all_data['BsmtCond'].fillna('No')
    all_data['BsmtQual'] = all_data['BsmtQual'].fillna('No')
    all_data['BsmtExposure'] = all_data['BsmtExposure'].fillna('NoB')
    all_data['BsmtFinType1'] = all_data['BsmtFinType1'].fillna('No')
    all_data['BsmtFinType2'] = all_data['BsmtFinType2'].fillna('No')
    all_data['MasVnrType'] = all_data['MasVnrType'].fillna('No')
    all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)
    all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].value_counts().idxmax())
    all_data['Functional'] = all_data['Functional'].fillna(all_data['Functional'].value_counts().idxmax())
    all_data['Utilities'] = all_data['Utilities'].fillna(all_data['Utilities'].value_counts().idxmax())
    all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].value_counts().idxmax())
    all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].value_counts().idxmax())
    all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].value_counts().idxmax())
    all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].value_counts().idxmax())
    all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].value_counts().idxmax())
    predict_numerical_nan_values(all_data, 'LotFrontage')
    all_data = all_data.fillna(all_data.mean())
    '''
    all_data["LotShape"] = all_data["LotShape"].replace({'IR3': 1, 'IR2': 2,
                                                         'IR1': 3, 'Reg': 4})
    all_data["LandContour"] = all_data["LandContour"].replace({'Low': 1, 'HLS': 2,
                                                               'Bnk': 3, 'Lvl': 4})
    all_data["Utilities"] = all_data["Utilities"].replace({'ELO': 1, 'NoSeWa': 2,
                                                           'NoSewr': 3, 'AllPub': 4})
    all_data["LandSlope"] = all_data["LandSlope"].replace({'Sev': 1, 'Mod': 2,
                                                           'Gtl': 3})
    all_data["ExterQual"] = all_data["ExterQual"].replace({'Po': 0.0, 'Fa': 0.25,
                                                           'TA': 0.5, 'Gd': 0.75, 'Ex': 1.0}) * 5
    all_data["ExterCond"] = all_data["ExterCond"].replace({'Po': 0.0, 'Fa': 0.25,
                                                           'TA': 0.5, 'Gd': 0.75, 'Ex': 1.0}) * 5
    all_data["BsmtQual"] = all_data["BsmtQual"].replace({'No': 0.0, 'Po': 0.2, 'Fa': 0.4,
                                                         'TA': 0.6, 'Gd': 0.8, 'Ex': 1.0}) * 6
    all_data["BsmtCond"] = all_data["BsmtCond"].replace({'No': 0.0, 'Po': 0.2, 'Fa': 0.4,
                                                         'TA': 0.6, 'Gd': 0.8, 'Ex': 1.0}) * 6
    all_data["BsmtExposure"] = all_data["BsmtExposure"].replace({'NoB': 0.0, 'No': 0.25,
                                                                 'Mn': 0.5, 'Av': 0.75, 'Gd': 1.0}) * 5
    all_data["BsmtFinType1"] = all_data["BsmtFinType1"].replace({'No': 0.0, 'Unf': 1 / 6,
                                                                 'LwQ': 2 / 6, 'Rec': 3 / 6, 'BLQ': 4 / 6,
                                                                 'ALQ': 5 / 6, 'GLQ': 1.0}) * 6
    all_data["BsmtFinType2"] = all_data["BsmtFinType2"].replace({'No': 0.0, 'Unf': 1 / 6,
                                                                 'LwQ': 2 / 6, 'Rec': 3 / 6, 'BLQ': 4 / 6,
                                                                 'ALQ': 5 / 6, 'GLQ': 1.0}) * 6
    all_data["HeatingQC"] = all_data["HeatingQC"].replace({'Po': 0.0, 'Fa': 0.25,
                                                           'TA': 0.5, 'Gd': 0.75, 'Ex': 1.0}) * 4
    all_data["KitchenQual"] = all_data["KitchenQual"].replace({'Po': 0.0, 'Fa': 0.25,
                                                               'TA': 0.5, 'Gd': 0.75, 'Ex': 1.0}) * 4
    all_data["Functional"] = all_data["Functional"].replace({'Sal': 1, 'Sev': 2,
                                                             'Maj2': 3, 'Maj1': 4, 'Mod': 5,
                                                             'Min2': 6, 'Min1': 7, 'Typ': 8})
    all_data["GarageFinish"] = all_data["GarageFinish"].replace({'No': 0, 'Unf': 1,
                                                                 'RFn': 2, 'Fin': 3})
    all_data["GarageQual"] = all_data["GarageQual"].replace({'No': 0, 'Po': 1, 'Fa': 2,
                                                             'TA': 3, 'Gd': 4, 'Ex': 5})
    all_data["GarageCond"] = all_data["GarageCond"].replace({'No': 0, 'Po': 1, 'Fa': 2,
                                                             'TA': 3, 'Gd': 4, 'Ex': 5})
    all_data["PoolQC"] = all_data["PoolQC"].replace({'No': 0, 'Fa': 1,
                                                     'TA': 2, 'Gd': 3, 'Ex': 4})
    all_data["PavedDrive"] = all_data["PavedDrive"].replace({'N': 0, 'P': 1, 'Y': 2})
    '''
    return all_data, train, test

def predict_numerical_nan_values(data, feature):
    all_data = data.drop(columns=[feature])
    notna_cols = (all_data.isna().any() == False).values
    all_data =  all_data.iloc[:, notna_cols]
    all_data = pd.get_dummies(all_data)
    X = all_data[data[feature].notna()]
    y = data[data[feature].notna()][feature]
    test = all_data[data[feature].isna()]
    model = Lasso().fit(X, y)
    preds = model.predict(test)
    data.loc[data[feature].isna(),feature] = preds