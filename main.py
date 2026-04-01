from flask import Flask, render_template, request
import joblib
import re

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
        prediction = model.predict([cleaned_input])[0]
    except Exception as e:
        prediction = "Error"
    
    return render_template('frontend.html', prediction=prediction, analyzed_text=display_text)

@app.route('/mbti-types')
def mbti_types(): #types of mbti
    return render_template('types.html')

if __name__ == '__main__':
    app.run(debug=True)