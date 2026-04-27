from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email_text = request.form["email"]

    transformed_text = vectorizer.transform([email_text])

    prediction = model.predict(transformed_text)[0]

    if prediction == 1:
        result = "Spam Email"
    else:
        result = "Not Spam Email"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)