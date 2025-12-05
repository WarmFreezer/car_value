# Thomas Eubank
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
from scipy.stats import uniform
import pandas as pd 
import numpy as np


# fetch dataset 
car_evaluation = fetch_ucirepo(id=19) 
  
# data (as pandas dataframes) 
X = car_evaluation.data.features.copy() 
y = car_evaluation.data.targets.copy() 

# Prevent annoying future warning
pd.set_option('future.no_silent_downcasting', True)

# Replace the string values with numerical values
X = X.replace(['vhigh', 'high', 'med', 'low'], [4, 3, 2, 1])
X = X.replace(['5more', 'more'], [7, 5])
X = X.replace(['big', 'med', 'small'], [3, 2, 1])
y = y.replace(['unacc', 'acc', 'good', 'vgood'], [0, 1, 2, 3])

y = y.values.ravel()

# Split and train model
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure labels are integer 1D arrays (sklearn requires discrete class labels)
train_y = np.asarray(train_y, dtype=int)
test_y = np.asarray(test_y, dtype=int)

# Scale the features
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# Train Logistic Regression model
log_reg = LogisticRegression(random_state=42, solver='lbfgs')

param_distributions = {
	'C': uniform(loc=0.01, scale=100),  # A continuous distribution for C
	'penalty': ['l2'],
	'class_weight': [None, 'balanced'],
	'max_iter': [100, 200, 300, 400, 500]
}

carVal = RandomizedSearchCV(
	estimator=log_reg,
	param_distributions=param_distributions,
	n_iter=50,  # Number of parameter settings that are sampled
	cv=5,       # Number of cross-validation folds
	scoring='accuracy', # Or other suitable metrics like 'f1', 'roc_auc'
	random_state=42,
	n_jobs=-1   # Use all available CPU cores
)

carVal.fit(train_x, train_y)

# predict
predictions = carVal.predict(test_x)
print(predictions - test_y)

#####
# feature names (after your replacing step)
feature_names = list(X.columns)  # or original_X.columns

# carVal is RandomizedSearchCV; access the best trained model via best_estimator_
# best_estimator_.coef_ shape: (n_classes, n_features) for multinomial
coefs = carVal.best_estimator_.coef_  # shape (n_classes, n_features)

# aggregate: mean absolute coefficient across classes
mean_abs_coef = np.mean(np.abs(coefs), axis=0)
max_abs_coef = np.max(np.abs(coefs), axis=0)

coef_df = pd.DataFrame({
    'feature': feature_names,
    'mean_abs_coef': mean_abs_coef,
    'max_abs_coef': max_abs_coef,
    # optional: signed coef from one class (e.g., class 0)
    'coef_class0': coefs[0]
})
coef_df = coef_df.sort_values('mean_abs_coef', ascending=False).reset_index(drop=True)
print("Top features by mean |coef|:") 
print(coef_df.head(15))
# quick bar plot
coef_df.set_index('feature')['mean_abs_coef'].head(20).plot(kind='barh', figsize=(8,6), title='Feature importance (mean |coef|)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
#####

# accuracy
accuracy = accuracy_score(test_y, predictions)
print(f"Accuracy: {accuracy}")

# precision
precision = precision_score(test_y, predictions, average='weighted', zero_division=0)
print(f"Precision: {precision}")

# confusion matrix
conf_matrix = confusion_matrix(test_y, predictions, labels=carVal.classes_)
print("Confusion Matrix:")
print(conf_matrix)

# hyperparameters
print("Best Hyperparameters:")
print(carVal.best_params_)

# Plot confusion matrix
# Normalize confusion matrix to get proportions within each true class
row_sums = conf_matrix.sum(axis=1, keepdims=True)
with np.errstate(divide='ignore', invalid='ignore'):
	conf_norm = conf_matrix / row_sums
conf_norm = np.nan_to_num(conf_norm)

# Create heatmap
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(conf_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
ax.set_title('Confusion Matrix (proportion within true class)')
fig.colorbar(cax, ax=ax, label='Proportion within true class')
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
ax.set_xticks(np.arange(len(carVal.classes_)))
ax.set_yticks(np.arange(len(carVal.classes_)))
ax.set_xticklabels(carVal.classes_)
ax.set_yticklabels(carVal.classes_)

# Annotate each cell with raw count and proportion (count\nprop)
for i in range(conf_matrix.shape[0]):
	for j in range(conf_matrix.shape[1]):
		count = int(conf_matrix[i, j])
		prop = conf_norm[i, j]
		text = f"{count}\n{prop:.2f}"
		# Choose text color for readability
		color = 'white' if prop > 0.5 else 'black'
		ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9)

plt.tight_layout()
plt.show()