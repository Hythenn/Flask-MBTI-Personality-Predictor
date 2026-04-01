import pandas as pd
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline

def clean_text(text):
    # remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    # remove specific formatting marks like |||
    text = text.replace('|||', ' ')
    # remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# dataset
DATASET_NAME = 'mbti_1.csv' 

try:
    print(f"Loading {DATASET_NAME}...")
    df = pd.read_csv(DATASET_NAME) 

    print("Cleaning text data...")
    df['posts'] = df['posts'].apply(clean_text)

    print("Balancing dataset (oversampling Extroverts to match Introverts)...")
    introverts = df[df['type'].str.startswith('I')]
    extroverts = df[df['type'].str.startswith('E')]
    
    # Check counts
    print(f"Original - Introverts: {len(introverts)}, Extroverts: {len(extroverts)}")
    
    # Oversample extroverts to match introverts
    extroverts_oversampled = extroverts.sample(len(introverts), replace=True, random_state=42)
    
    df_balanced = pd.concat([introverts, extroverts_oversampled])
    # Shuffle the dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Balanced Dataset Total Size: {len(df_balanced)}")

    # dataset column names
    X = df_balanced['posts'] 
    y = df_balanced['type']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training ensemble model pipeline...")
    # Using multiple models to create a Voting Classifier
    estimators = [
        ('svc', LinearSVC(class_weight='balanced')),
        ('lr', LogisticRegression(max_iter=1000, class_weight='balanced')),
        ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1))
    ]
    
    ensemble_clf = VotingClassifier(estimators=estimators, voting='hard')
    
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('clf', ensemble_clf)
    ])

    model_pipeline.fit(X_train, y_train)

    # checks the model accuracy
    score = model_pipeline.score(X_test, y_test)
    print(f"Training Done! Model Accuracy: {score * 100:.2f}%")

    joblib.dump(model_pipeline, 'modelmbti.pkl')
    print("File saved as: modelmbti.pkl")

except FileNotFoundError:
    print(f"Error: Could not find {DATASET_NAME}. Please check the filename.")
