# *coding: utf-8*
# Cisco open ended project.ipynb
!pip install scikit-learn pandas -q

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = {
    "text": [
        "Artificial intelligence is transforming industries worldwide.",
        "The weather is nice today and I went for a walk.",
        "Machine learning models can analyze huge datasets quickly.",
        "I had dinner with my friends yesterday evening.",
        "Deep learning improves computer vision applications.",
        "My mother cooked delicious food today.",
        "Neural networks are inspired by the human brain.",
        "I completed my homework and watched a movie.",
        "AI systems can automate repetitive tasks efficiently.",
        "We played cricket after college classes."
    ],

    # AI = 1, Human = 0
    "label": [1,0,1,0,1,0,1,0,1,0]
}

df = pd.DataFrame(data)

print("Dataset")
print(df)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)

print("Model Training Completed!")

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

print("AI TEXT CHECKER")
user_text = input("Enter text: ")
user_vector = vectorizer.transform([user_text])
result = model.predict(user_vector)

if result[0] == 1:
    print("Prediction: AI Generated Text")
else:
    print("Prediction: Human Written Text")
