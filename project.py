#1
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = fetch_openml('mnist_784')

# Get the data and labels
X = mnist.data
y = mnist.target.astype(int)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print dataset dimensions
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")


#2
# Visualize a few sample images and their corresponding labels
fig, axes = plt.subplots(1, 5, figsize=(12, 6))
for i, ax in enumerate(axes):
    ax.imshow(X_train.iloc[i].values.reshape(28, 28), cmap='gray')
    ax.set_title(f"Label: {y_train.iloc[i]}")
    ax.axis('off')
plt.show()


#3
from sklearn.svm import SVC

# Initialize the SVM classifier
svm = SVC()

# Fit the model to the training data
svm.fit(X_train, y_train)

# Print the accuracy score on the test set
print(f"Test set accuracy: {svm.score(X_test, y_test):.4f}")


#4
from sklearn.model_selection import cross_val_score

# Perform 5-fold Cross-Validation
cv_scores = cross_val_score(svm, X_train, y_train, cv=5)

# Report the min and max accuracy scores
print(f"Minimum Cross-Validation Accuracy: {cv_scores.min():.4f}")
print(f"Maximum Cross-Validation Accuracy: {cv_scores.max():.4f}")


#5
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Define the hyperparameter grid
param_dist = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf', 'poly']
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(SVC(), param_distributions=param_dist, n_iter=10, cv=5, random_state=42)

# Fit the RandomizedSearchCV
random_search.fit(X_train, y_train)

# Report the best hyperparameters and score
print(f"Best hyperparameters from RandomizedSearchCV: {random_search.best_params_}")
print(f"Best score from RandomizedSearchCV: {random_search.best_score_:.4f}")


#6
from sklearn.model_selection import GridSearchCV

# Narrow down the hyperparameter grid
param_grid = {
    'C': [1, 10],
    'gamma': ['scale'],
    'kernel': ['rbf']
}

# Initialize GridSearchCV
grid_search = GridSearchCV(SVC(), param_grid, cv=5)

# Fit the GridSearchCV
grid_search.fit(X_train, y_train)

# Report the best hyperparameters and score
print(f"Best hyperparameters from GridSearchCV: {grid_search.best_params_}")
print(f"Best score from GridSearchCV: {grid_search.best_score_:.4f}")


#7
import pandas as pd

# Create a DataFrame to store the results
results = pd.DataFrame({
    'Model': ['Baseline', 'RandomizedSearchCV', 'GridSearchCV'],
    'Best Score': [svm.score(X_test, y_test), random_search.best_score_, grid_search.best_score_],
    'Best Hyperparameters': [None, random_search.best_params_, grid_search.best_params_]
})

print(results)


#8
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print(f"Test set accuracy of the best model: {test_accuracy:.4f}")

# Compute the confusion matrix
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
cm_display.plot(cmap='Blues')
plt.show()


