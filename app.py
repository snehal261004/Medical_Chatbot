import csv
import re
from flask import Flask, request, jsonify, render_template
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def prepare_data_for_naive_bayes(data, symptom_columns):
    X, y = [], []
    for row in data:
        symptoms = []
        for column in symptom_columns:
            text = row[column]
            if isinstance(text, str):
                symptoms.extend(re.findall(r'\b\w+\b', text.lower()))
        X.append(" ".join(symptoms))
        y.append(row['Disease'])  
    return X, y

def map_disease_info(data):
    disease_info = {}
    for row in data:
        disease_name = row['Disease']
        if disease_name not in disease_info:
            disease_info[disease_name] = {
                "description": row['description'],
                "workout": row['workout'],
                "diets": [row[f'Diet_{i}'] for i in range(1, 5) if row[f'Diet_{i}']],
                "medications": [row[f'Medication_{i}'] for i in range(1, 5) if row[f'Medication_{i}']],
                "precautions": [row[f'Precaution_{i}'] for i in range(1, 5) if row[f'Precaution_{i}']]
            }
    return disease_info


def predict_disease_info(user_input, model, vectorizer, disease_info):
    input_vec = vectorizer.transform([user_input])
    predicted_disease = model.predict(input_vec)[0]
    return {
        "disease": predicted_disease,
        "details": disease_info.get(predicted_disease, {})
    }

data_file = 'data/combined_dataset.csv' 
symptom_columns = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']


data = load_data(data_file)


X, y = prepare_data_for_naive_bayes(data, symptom_columns)


vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)


model = MultinomialNB()
model.fit(X_vec, y)


disease_info = map_disease_info(data)


current_info = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    global current_info

    user_message = request.json.get('message').lower()

    if user_message in ["hi", "hello","hey"]:
        return jsonify({"response": "Hello! Please tell me your symptoms."})

    elif user_message == "symptoms":
        return jsonify({"response": "Please enter your symptoms (e.g., 'fever, cough')."})
    
    elif user_message == "thank you" or user_message in ["no thank you", "no, thank you","ok thank you","no thanks"]:
        return jsonify({"response": "You're welcome! If you have any more questions or need assistance, feel free to ask."})

    elif any(x in user_message for x in ["description", "diet", "diets", "medication", "medications", "precaution", "precautions", "workouts"]):
        if "diet" in user_message or "diets" in user_message:
            option = "diet"
        elif "medication" in user_message or "medications" in user_message:
            option = "medications"
        elif "precaution" in user_message or "precautions" in user_message:
            option = "precautions"
        elif "workouts" in user_message:
            option = "workouts"
        elif "description" in user_message:
            option = "description"
        else:
            return jsonify({"response": "I'm sorry, I didn't understand your request. Could you clarify?"})


        response = format_response(option, current_info) + "\n\nWould you like to know more about description, diet, medications, or precautions?"
        return jsonify({"response": response})

    else:
        vectorized_symptoms = user_message 
        result = predict_disease_info(vectorized_symptoms, model, vectorizer, disease_info)
        disease = result['disease']
        print(disease)
        current_info = result['details']

        if disease:
            response = (
                f"The predicted disease is {disease}. Would you like to know more about description, "
                f"diet, medications or precautions"
            )

            print(response)

            if response == "The predicted disease is (vertigo) Paroymsal  Positional Vertigo. Would you like to know more about description, diet, medications or precautions":
                return jsonify({"response": "I'm sorry, I couldn't find a match. Could you provide more details?"})

            return jsonify({"response": response})
        else:
            return jsonify({"response": "I'm sorry, I couldn't find a match. Could you provide more details?"})


def format_response(option, info):
    response_mapping = {
        "description": f"Description: {info['description']}",
        "diet": f"Diet: {', '.join(info['diets'])}",
        "medications": f"Medications: {', '.join(info['medications'])}",
        "precautions": f"Precautions: {', '.join(info['precautions'])}",
        "workouts": f"Workouts: {', '.join(info['workout'])}"
    }
    return response_mapping.get(option.lower(), "I'm here to assist with more details about the disease.")

if __name__ == '__main__':
    app.run(debug=True)
