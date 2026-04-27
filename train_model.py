import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("emails.csv")

# Remove null values
data = data.dropna()

print(data.head())
print(data["spam"].value_counts())

X = data["text"]
y = data["spam"]

# Convert text into vectors
vectorizer = TfidfVectorizer(
    stop_words='english',
    lowercase=True
)

X_vectorized = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Better model
model = LogisticRegression(class_weight='balanced', max_iter=1000)

model.fit(X_train, y_train)

# Test
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("New model saved successfully")