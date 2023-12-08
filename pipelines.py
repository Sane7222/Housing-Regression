import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.compose import TransformedTargetRegressor

# SelectColumns was provided by Professor William
class SelectColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns): # pass the function we want to apply to the column 'SalePrice'
        self.columns = columns
    def fit(self, xs, ys, **params): # don't need to do anything
        return self
    def transform(self, xs): # actually perform the selection
        return xs[self.columns]

def remove_NaNs(data, column): # Function to remove Nulls and NaNs
    data[column] = data[column].fillna(0)

# Get and segregate data into Xs and Ys
data = pd.read_csv('AmesHousing.csv')
xs = data.drop(columns= ['SalePrice'])
ys = data['SalePrice']

# Get some dummies for Neighborhood
df_encoded = pd.get_dummies(xs, columns= ['Neighborhood'])
xs['Neighborhood_1'] = df_encoded['Neighborhood_NridgHt']
xs['Neighborhood_2'] = df_encoded['Neighborhood_NoRidge']

# Map categories to numbers, instead of get_dummies()
ms_zone_map = {'A (agr)': 0, 'I (all)': 1, 'C (all)': 2, 'RH': 3, 'RM': 4, 'FV': 5, 'RL': 6}
xs['MS Zoning'] = xs['MS Zoning'].map(ms_zone_map)

bsmt_qual_map = {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
xs['Bsmt Qual'] = xs['Bsmt Qual'].map(bsmt_qual_map)

kitchen_qual_map = {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
xs['Kitchen Qual'] = xs['Kitchen Qual'].map(kitchen_qual_map)

exter_qual_map = {'Fa': 0, 'TA': 1, 'Gd': 2, 'Ex': 3}
xs['Exter Qual'] = xs['Exter Qual'].map(exter_qual_map)

heat_qual_map = {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
xs['Heating QC'] = xs['Heating QC'].map(heat_qual_map)

garage_fin_map = {'Unf': 0, 'RFn': 1, 'Fin': 2}
xs['Garage Finish'] = xs['Garage Finish'].map(garage_fin_map)

fire_qual_map = {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
xs['Fireplace Qu'] = xs['Fireplace Qu'].map(fire_qual_map)

# Clean up any NaNs or Nulls
for column in xs.columns:
    remove_NaNs(xs, column)

steps = [
    ('column_select', SelectColumns(['Overall Qual'])),
    ('linear_regression', LinearRegression(n_jobs= -1))
]

grid = {
    'column_select__columns':
        [['Gr Liv Area', 'Overall Qual',
        'Bsmt Qual', 'MS Zoning',
        'Wood Deck SF', 'Kitchen Qual',
        'Exter Qual', 'Mas Vnr Area',
        'Fireplace Qu', 'BsmtFin SF 1',
        '1st Flr SF', 'Garage Area',
        'Garage Finish', 'Heating QC',
        'Neighborhood_1', 'Neighborhood_2']]
    ,
    'linear_regression':[
        LinearRegression(n_jobs= -1), # no transformation
        TransformedTargetRegressor(
            LinearRegression(n_jobs= -1),
            func= np.sqrt,
            inverse_func= np.square
        ),
        TransformedTargetRegressor(
            LinearRegression(n_jobs= -1),
            func= np.cbrt,
            inverse_func= lambda y: np.power(y, 3)
        ),
        TransformedTargetRegressor(
            LinearRegression(n_jobs= -1),
            func= np.log,
            inverse_func= np.exp
        )
    ]
}

pipe = Pipeline(steps)
search = GridSearchCV(pipe, grid, scoring= 'r2', n_jobs= -1) # Specifying r^2
search.fit(xs, ys)

print(search.best_score_) # Best r^2
print(search.best_params_) # Best Combination