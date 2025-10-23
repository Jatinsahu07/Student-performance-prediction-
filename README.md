# Student-performance-prediction-

This project uses classification to predict a student's final outcome (Pass/Fail) based on various personal and academic factors.
1. Libraries and Data
Libraries: Import pandas for data handling and scikit-learn (sklearn) for modeling.
Data: Load student data and create a binary target variable (Pass = 1, Fail = 0) from the final grade (e.g., G3 \geq 10 is a Pass).
2. Preparation
Encoding: Convert non-numeric, categorical features (like 'sex', 'schoolsup') into numerical values using Label Encoding and One-Hot Encoding.
Splitting: Separate the processed data into Training (for the model to learn) and Testing (for evaluation) sets.
3. Training and Evaluation
Model: Use a Random Forest Classifier (RandomForestClassifier) from sklearn.
Training: Fit the model to the training data (\text{X\_train}, \text{y\_train}).
Prediction: Use the trained model to predict outcomes on the unseen test data (\text{X\_test}).
Evaluation: Measure performance using Accuracy and a Classification Report (which details Precision and Recall) to assess how well the model predicts Passes and Fails.
