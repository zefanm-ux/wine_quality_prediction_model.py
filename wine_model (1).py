import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import urllib.request

# Download the wine quality dataset (red wine)
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
filename = 'winequality-red.csv'
urllib.request.urlretrieve(url, filename)

# Load the dataset
data = pd.read_csv(filename, sep=';')

# Process the data: categorize quality into 'good' (>=6) and 'bad' (<6)
data['quality'] = data['quality'].apply(lambda x: 'good' if x >= 6 else 'bad')

# Features and target
X = data.drop('quality', axis=1)
y = data['quality']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model on original features
model_original = RandomForestClassifier(random_state=42)
model_original.fit(X_train, y_train)
y_pred_original = model_original.predict(X_test)
accuracy_original = accuracy_score(y_test, y_pred_original)

# PCA for feature reduction
pca = PCA(n_components=5)  # Reduce to 5 components
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Model on PCA features
model_pca = RandomForestClassifier(random_state=42)
model_pca.fit(X_train_pca, y_train)
y_pred_pca = model_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)

# LDA for feature reduction
lda = LDA(n_components=1)  # For binary classification, 1 component
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Model on LDA features
model_lda = RandomForestClassifier(random_state=42)
model_lda.fit(X_train_lda, y_train)
y_pred_lda = model_lda.predict(X_test_lda)
accuracy_lda = accuracy_score(y_test, y_pred_lda)

# Comparison
print(f'Accuracy with original features: {accuracy_original}')
print(f'Accuracy with PCA features: {accuracy_pca}')
print(f'Accuracy with LDA features: {accuracy_lda}')
print('Classification Report for original features:')
print(classification_report(y_test, y_pred_original))
print('Classification Report for PCA features:')
print(classification_report(y_test, y_pred_pca))
print('Classification Report for LDA features:')
print(classification_report(y_test, y_pred_lda))
