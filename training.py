import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

# dataset
DATASET_NAME = 'mbti_1.csv' 

try:
    print(f"Loading {DATASET_NAME}...")
    df = pd.read_csv(DATASET_NAME) 

    # dataset column names
    X = df['posts'] 
    y = df['type']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model pipeline...")
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('clf', LinearSVC())
    ])

    model_pipeline.fit(X_train, y_train)

    # checks the model accuracy
    score = model_pipeline.score(X_test, y_test)
    print(f"Training Done! Model Accuracy: {score * 100:.2f}%")

    joblib.dump(model_pipeline, 'modelmbti.pkl')
    print("File saved as: modelmbti.pkl")

except FileNotFoundError:
    print(f"Error: Could not find {DATASET_NAME}. Please check the filename.")
