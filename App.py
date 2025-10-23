import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the dataset (replace 'your_data.csv' with your actual file)
try:
    data = pd.read_csv('student-mat.csv', sep=';') # Assuming semicolon separated values
except FileNotFoundError:
    print("Please ensure your dataset file is named 'student-mat.csv' and is in the correct directory.")
    # For a placeholder example, you'd typically load or create a synthetic dataset here.
    # We will proceed with the assumption the file is loaded.
    pass

# The final grade (G3) is the column we'll use to create our target variable.
# Let's define a passing grade as G3 >= 10 (out of 20).
data['Pass'] = data['G3'].apply(lambda x: 1 if x >= 10 else 0)

# Drop the original grade columns (G1, G2, G3) and unrelated IDs if they exist
data = data.drop(['G1', 'G2', 'G3'], axis=1)

# Separate features (X) and target (y)
X = data.drop('Pass', axis=1)
y = data['Pass']

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Apply Label Encoding for binary columns (e.g., 'sex', 'schoolsup')
# and One-Hot Encoding for multi-category columns (e.g., 'Mjob', 'reason')

# Label Encoding for binary features (example: 'sex', 'address', 'famsize')
le = LabelEncoder()
for col in ['sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 
            'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']:
    if col in X.columns:
        X[col] = le.fit_transform(X[col])

# One-Hot Encoding for the remaining non-binary categorical features
# This creates new columns for each category (e.g., 'Mjob_at_home', 'Mjob_health', etc.)
X = pd.get_dummies(X, columns=[col for col in categorical_cols if col not in ['sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 
                                                                               'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']], drop_first=True)

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

print("Model training complete.)


# Get feature importances from the Random Forest model
feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)

# Select the top 10 most important features
top_10_features = feature_importances.nlargest(10)

print("\nTop 10 Feature Importances:")
print(top_10_features)

# 
