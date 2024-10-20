import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib import colormaps
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import joblib
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset and remove any missing values
df = pd.read_csv("C:/Users/hassa/Documents/GitHub/Projects/Project_1_Data.csv")
df = df.dropna()
df = df.reset_index(drop=True)

X = df[['X', 'Y', 'Z']]  
y = df['Step']  

# Split the dataset into stratified train and test sets
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42069)

for train_idx, test_idx in splitter.split(df, df['Step']):
    strat_train = df.loc[train_idx].reset_index(drop=True)
    strat_test = df.loc[test_idx].reset_index(drop=True)

# Define training and test sets
X_train = strat_train[['X', 'Y', 'Z']]
y_train = strat_train['Step']
X_test = strat_test[['X', 'Y', 'Z']]
y_test = strat_test['Step']

# Combine the training and test sets for visualization
X_all = pd.concat([X_train, X_test], ignore_index=True)
y_all = pd.concat([y_train, y_test], ignore_index=True)

# Plotting the correlation heatmap
plt.figure(figsize=(8, 6))
corr_matrix = df[['X', 'Y', 'Z', 'Step']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='flag')
plt.title('Correlation Matrix for X, Y, Z, and Step')
plt.show()

# 3D Scatter Plot with all data (both train and test)
fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot all the data (both train and test)
sc = ax.scatter(X_all['X'], X_all['Y'], X_all['Z'], c=y_all, cmap='viridis', marker='o', s=5)
cbar = plt.colorbar(sc)
cbar.set_label('Step')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot of X, Y, Z with Step')

plt.show()


# PART 4 | CLASSIFICATION MODEL DEVELOPMENT/ENGINEERING

# Logistic Regression using GridSearchCV
logistic_reg = LogisticRegression(max_iter=42069) 
param_grid_lr = {
    'penalty': ['l1', 'l2', 'none'], 
    'C': [0.01, 0.1, 1, 10, 100],  
    'solver': ['liblinear', 'lbfgs', 'saga']  
}
grid_search_lr = GridSearchCV(logistic_reg, param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_lr.fit(X_train, y_train)
best_model_lr = grid_search_lr.best_estimator_
print("Best Logistic Regression Model:", best_model_lr)

# Support Vector Machine Classifier (SVC) using GridSearchCV
svc = SVC()  
param_grid_svc = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10, 100],  
    'gamma': ['scale', 'auto']  
}
grid_search_svc = GridSearchCV(svc, param_grid_svc, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_svc.fit(X_train, y_train)
best_model_svc = grid_search_svc.best_estimator_
print("Best SVC Model:", best_model_svc)

# Decision Tree using GridSearchCV
decision_tree = DecisionTreeClassifier(random_state=42069)
param_grid_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_dt = GridSearchCV(decision_tree, param_grid_dt, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_dt.fit(X_train, y_train)
best_model_dt = grid_search_dt.best_estimator_
print("Best Decision Tree Model:", best_model_dt)

# Random Forest using RandomizedSearchCV
random_forest = RandomForestClassifier(random_state=42069)
param_dist_rf = {
    'n_estimators': [10, 30, 50, 100],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
random_search_rf = RandomizedSearchCV(random_forest, param_distributions=param_dist_rf, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=420)
random_search_rf.fit(X_train, y_train)
best_model_rf = random_search_rf.best_estimator_
print("Best Random Forest Model:", best_model_rf)


# PART 5 | MODEL PERFORMANCE ANALYSIS FOR EACH MODEL
# Logistic Regression Performance
y_test_pred_lr = best_model_lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_test_pred_lr)
precision_lr = precision_score(y_test, y_test_pred_lr, average='weighted')
f1_lr = f1_score(y_test, y_test_pred_lr, average='weighted')

# SVC Performance
y_test_pred_svc = best_model_svc.predict(X_test)
accuracy_svc = accuracy_score(y_test, y_test_pred_svc)
precision_svc = precision_score(y_test, y_test_pred_svc, average='weighted')
f1_svc = f1_score(y_test, y_test_pred_svc, average='weighted')

# Decision Tree Performance
y_test_pred_dt = best_model_dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_test_pred_dt)
precision_dt = precision_score(y_test, y_test_pred_dt, average='weighted')
f1_dt = f1_score(y_test, y_test_pred_dt, average='weighted')

# Random Forest Performance
y_test_pred_rf = best_model_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_test_pred_rf)
precision_rf = precision_score(y_test, y_test_pred_rf, average='weighted')
f1_rf = f1_score(y_test, y_test_pred_rf, average='weighted')

# Print performance metrics for each model
print(f"Logistic Regression - Accuracy: {accuracy_lr}, Precision: {precision_lr}, F1-Score: {f1_lr}")
print(f"Decision Tree - Accuracy: {accuracy_dt}, Precision: {precision_dt}, F1-Score: {f1_dt}")
print(f"Random Forest - Accuracy: {accuracy_rf}, Precision: {precision_rf}, F1-Score: {f1_rf}")
print(f"SVC - Accuracy: {accuracy_svc}, Precision: {precision_svc}, F1-Score: {f1_svc}")

# Comparing F1 Scores and determining the best model
best_f1_score = max(f1_lr, f1_dt, f1_rf, f1_svc)
if best_f1_score == f1_rf:
    best_model_name = "Random Forest"
    best_model = best_model_rf
    y_test_pred_best = y_test_pred_rf
elif best_f1_score == f1_dt:
    best_model_name = "Decision Tree"
    best_model = best_model_dt
    y_test_pred_best = y_test_pred_dt
elif best_f1_score == f1_svc:
    best_model_name = "SVC"
    best_model = best_model_svc
    y_test_pred_best = y_test_pred_svc
else:
    best_model_name = "Logistic Regression"
    best_model = best_model_lr
    y_test_pred_best = y_test_pred_lr

print(f"\nThe best model is: {best_model_name} with an F1-Score of: {best_f1_score}")

# Confusion Matrix for the Best Model
cm_best = confusion_matrix(y_test, y_test_pred_best)

# Display confusion matrix
disp_best = ConfusionMatrixDisplay(confusion_matrix=cm_best, display_labels=np.unique(y_test))
disp_best.plot(cmap='rocket')

plt.xlabel('Predicted Value')
plt.ylabel('True Value')
plt.title(f'Confusion Matrix for {best_model_name}')
plt.show()



#PART 6 | STACKED MODEL PERFORMANCE ANALYSIS 
# Step 1: Compute F1 scores for all models, including SVC
f1_scores = {
    'Logistic Regression': f1_lr,
    'Decision Tree': f1_dt,
    'Random Forest': f1_rf,
    'SVC': f1_score(y_test, best_model_svc.predict(X_test), average='weighted')
}

# Step 2: Sort models based on F1 score in descending order
sorted_models = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)

# Step 3: Select the top 2 models
best_model_1_name, best_f1_model_1 = sorted_models[0]
best_model_2_name, best_f1_model_2 = sorted_models[1]

# Map model names to actual trained models
model_dict = {
    'Logistic Regression': best_model_lr,
    'Decision Tree': best_model_dt,
    'Random Forest': best_model_rf,
    'SVC': best_model_svc
}

best_model_1 = model_dict[best_model_1_name]
best_model_2 = model_dict[best_model_2_name]

print(f"Selected models for stacking: {best_model_1_name} and {best_model_2_name}")

# Step 4: Create StackingClassifier with the best two models
base_models = [
    (best_model_1_name.lower().replace(' ', '_'), best_model_1),
    (best_model_2_name.lower().replace(' ', '_'), best_model_2)
]

#Train StackingClassifier
stacked_clf = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(), cv=5)

stacked_clf.fit(X_train, y_train)

# Step 6: Evaluate Stacked Model
y_test_pred_stacked = stacked_clf.predict(X_test)

accuracy_stacked = accuracy_score(y_test, y_test_pred_stacked)
precision_stacked = precision_score(y_test, y_test_pred_stacked, average='weighted')
f1_stacked = f1_score(y_test, y_test_pred_stacked, average='weighted')

# Step 7: Print performance metrics
print(f"Stacked Model - Accuracy: {accuracy_stacked}")
print(f"Stacked Model - Precision: {precision_stacked}")
print(f"Stacked Model - F1 Score: {f1_stacked}")

# Step 8: Confusion Matrix for the Stacked Model
cm_stacked = confusion_matrix(y_test, y_test_pred_stacked)

# Display confusion matrix
disp_stacked = ConfusionMatrixDisplay(confusion_matrix=cm_stacked, display_labels=stacked_clf.classes_)
disp_stacked.plot(cmap='rocket')
plt.xlabel('Predicted Value')
plt.ylabel('True Value')
plt.title(f'Confusion Matrix for Stacked Model: {best_model_1_name} and {best_model_2_name}')
plt.show()


#PART 7 | MODEL EVALUATION
# Save the best SVC model
import joblib
import numpy as np

# Function to save the model
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved as '{filename}'")

# Function to load the model
def load_model(filename):
    return joblib.load(filename)

def predict_maintenance_steps(model, coordinates):
    return model.predict(coordinates)


SVC_Project_1 = "best_svc_model.joblib"  
coordinates = np.array([
    [9.375, 3.0625, 1.51], 
    [6.995, 5.125, 0.3875], 
    [0, 3.0625, 1.93], 
    [9.4, 3, 1.8], 
    [9.4, 3, 1.3]
])

# Save, load, and predict using functions
save_model(best_model_svc, SVC_Project_1)
loaded_svc_model = load_model(SVC_Project_1)
svc_predictions = predict_maintenance_steps(loaded_svc_model, coordinates)

print("Predicted steps using SVC:", svc_predictions)



