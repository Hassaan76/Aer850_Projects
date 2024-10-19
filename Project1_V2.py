import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV



df = pd.read_csv("C:/Users/hassa/Documents/GitHub/Projects/Project_1_Data.csv")

print(df.info())


#the use of stratified sampling is strongly recommended
df["income_categories"] = pd.cut(df["median_income"],
                          bins=[0, 2, 4, 6, np.inf],
                          labels=[1, 2, 3, 4])
my_splitter = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 420)
for train_index, test_index in my_splitter.split(df, df["income_categories"]):
    strat_df_train = df.loc[train_index].reset_index(drop=True)
    strat_df_test = df.loc[test_index].reset_index(drop=True)
strat_df_train = strat_df_train.drop(columns=["income_categories"], axis = 1)
strat_df_test = strat_df_test.drop(columns=["income_categories"], axis = 1)



#in this method of variable selection, we just drop our outcome measure column from training data
X_train = strat_df_train.drop("Step", axis = 1)
y_train = strat_df_train["Step"]
X_test = strat_df_test.drop("Step", axis = 1)
y_test = strat_df_test["Step"]


"""Correlation Matrix"""
corr_matrix = (X_train.iloc[:,0:-5]).corr()
sns.heatmap(np.abs(corr_matrix))


plt.figure()
corr_matrix_2 = (X_train.iloc[:,0:-5]).corr()
sns.heatmap(np.abs(corr_matrix_2))



# """Training First Model"""
# my_model1 = LinearRegression()
# my_model1.fit(X_train, y_train)
# y_pred_train1 = my_model1.predict(X_train)

# for i in range(5):
#     print("Predictions:", y_pred_train1[i], "Actual values:", y_train[i])

# mae_train1 = mean_absolute_error(y_pred_train1, y_train)
# print("Model 1 training MAE is: ", round(mae_train1,2))



# """Cross Validation Model 1"""
# my_model1 = LinearRegression()
# cv_scores_model1 = cross_val_score(my_model1, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
# cv_mae1 = -cv_scores_model1.mean()
# print("Model 1 Mean Absolute Error (CV):", round(cv_mae1, 2))
# """NOTE THAT THERE IS DATA LEAK IN THIS WAY OF IMPLEMENTING CV. WHY?"""


# """Training Second Model"""
# my_model2 = RandomForestRegressor(n_estimators=30, random_state=42)
# my_model2.fit(X_train, y_train)
# y_pred_train2 = my_model2.predict(X_train)
# mae_train2 = mean_absolute_error(y_pred_train2, y_train)
# print("Model 2 training MAE is: ", round(mae_train2,2))





# for i in range(5):
#     print("Mode 1 Predictions:",
#           round(y_pred_train1[i],2),
#           "Mode 2 Predictions:",
#           round(y_pred_train2[i],2),
#           "Actual values:",
#           round(y_train[i],2))


# """GridSearchCV"""
# param_grid = {
#     'n_estimators': [10, 30, 50],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['sqrt', 'log2']
# }
# my_model3 = RandomForestRegressor(random_state=42)
# grid_search = GridSearchCV(my_model3, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=1)
# grid_search.fit(X_train, y_train)
# best_params = grid_search.best_params_
# print("Best Hyperparameters:", best_params)
# best_model3 = grid_search.best_estimator_