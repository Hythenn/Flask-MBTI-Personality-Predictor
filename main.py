from flask import Flask, render_template, request
import joblib
import re
import math

def clean_text(text):
    # remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    # remove specific formatting marks like |||
    text = text.replace('|||', ' ')
    # remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

app = Flask(__name__)

try:
    model = joblib.load('modelmbti.pkl')
except FileNotFoundError:
    print("Error: modelmbti.pkl not found.")

@app.route('/')
def index(): #test
    return render_template('frontend.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_responses = []
    #30 questions
    for i in range(1, 31):
        response = request.form.get(f'q{i}')
        if response:
            user_responses.append(response)
    
    unique_traits = list(set(user_responses))
    display_text = " ".join(unique_traits)
    combined_text = " ".join(user_responses)
    
    # Clean the input text identically to how the model was trained
    cleaned_input = clean_text(combined_text)
    
    try:
        try:
            # Instead of just .predict(), use .decision_function()
            scores = model.decision_function([cleaned_input])[0]
        except AttributeError:
            # Fallback: Since the model is a VotingClassifier Ensemble, we extract the decision function from its internal LinearSVC
            tfidf_features = model.named_steps['tfidf'].transform([cleaned_input])
            scores = model.named_steps['clf'].named_estimators_['svc'].decision_function(tfidf_features)[0]
            
        prediction = model.predict([cleaned_input])[0]
        # Convert the max raw score to a 70-99% confidence range for better UX
        raw_score = max(scores)
        
        # Standard sigmoid gives 0 to 1. We scale this to give 70 to 99.
        sigmoid_val = 1 / (1 + math.exp(-raw_score))
        confidence_pct = round(70.0 + (29.0 * sigmoid_val), 1)
    except Exception as e:
        prediction = "Error"
        confidence_pct = 0.0
    
    return render_template('frontend.html', prediction=prediction, analyzed_text=display_text, confidence_pct=confidence_pct)

@app.route('/mbti-types')
def mbti_types(): #types of mbti
    return render_template('types.html')

if __name__ == '__main__':
    app.run(debug=True)