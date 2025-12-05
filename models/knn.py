import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

#Feature Names
column_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]

#Read the data and assign data and target to X and y
df = pd.read_csv("car.data", names=column_names)
X = df.drop("class", axis=1)
y = df["class"]

#Switch string data to numeric values
X = X.replace(['vhigh', 'high', 'med', 'low'], [4, 3, 2, 1])
X = X.replace(['5more', 'more'], [7, 5])
X = X.replace(['big', 'med', 'small'], [3, 2, 1])
y = y.replace(['unacc', 'acc', 'good', 'vgood'], [0, 1, 2, 3])

#Training and Testing split of the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

#Scaling the data for better performance of the model
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
print("Scaled training features")
display(pd.DataFrame(X_train_scaled, columns=X_train.columns).head())

#Finding the optimal k cluster value for the model
kf = KFold(n_splits=10, shuffle=True, random_state=42)

k_values = range(1,31)
mean_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=kf, scoring='accuracy')
    mean_accuracies.append(scores.mean())

# Plot the results to find the optimal k
plt.figure(figsize=(12, 6))
plt.plot(k_values, mean_accuracies, marker='o', linestyle='--')
plt.title('K-NN Accuracy vs. K Value on Training Data')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Average Cross-Validation Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# Find the k value with the highest accuracy
optimal_k = k_values[np.argmax(mean_accuracies)]
print(f"The optimal k value is: {optimal_k}")

# Define a list of distance metrics to compare
distance_metrics = [
    'Euclidean',
    'Manhattan'
]

# A dictionary to store the results
results = {}

# Evaluate each distance metric using the optimal k and cross-validation on the training data
for name in distance_metrics:
    knn = KNeighborsClassifier(n_neighbors=optimal_k, metric=name.lower())
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=kf, scoring='accuracy')
    results[name] = scores.mean()

# Print the results
print("Performance with different distance metrics (using optimal k):")
for name, accuracy in results.items():
    print(f"- {name} Distance: {accuracy:.4f} Accuracy")

# Plot the comparison
plt.figure(figsize=(8, 5))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title('Comparison of K-NN Distance Metrics')
plt.ylabel('Average Cross-Validation Accuracy')
plt.ylim(0.9, 1.0)
plt.show()

# Identify the best metric from the cross-validation results
best_metric = max(results, key=results.get).lower()
print(f"The best performing distance metric is: {best_metric}")

#Training the model on the best k value and best distance metric and calculating respective values
final_knn = KNeighborsClassifier(n_neighbors=optimal_k, metric=best_metric)
final_knn.fit(X_train_scaled, y_train)

y_pred = final_knn.predict(X_test_scaled)

from sklearn.metrics import accuracy_score
test_accuracy = accuracy_score(y_test, y_pred)

print("\nFinal Model Evaluation:")
print(f"Optimal k: {optimal_k}")
print(f"Optimal Distance Metric: {best_metric}")
print(f"Accuracy on unseen Test Set: {test_accuracy:.4f}")

cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['unacc', 'acc', 'good', 'vgood'])
disp.plot()
plt.show()
