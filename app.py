#

import pandas as pd
df = pd.read_csv("News Category (Text).csv")

X=df['headline']
Y=df['category']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_CV = cv.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_CV, Y)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

model = LogisticRegression()
model.fit(X_train, Y_train)
pred = model.predict(X_test)
cm=confusion_matrix(Y_test, pred)
print(cm)
acc=(cm[0,0]+cm[1,1])/sum(sum(cm))
print('Accuracy count vectorizer is ', acc)

#store model
import joblib
joblib.dump(model, 'model.pkl')
joblib.dump(cv, "vectorizer.pkl")

vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("model.pkl")

X_emb = vectorizer.transform(["murder she wrote"])
pred = model.predict(X_emb)
print(pred)

import gradio as gr
import joblib

# Load model + vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_category(text):
    X_emb = vectorizer.transform([text])
    pred = model.predict(X_emb)[0]
    return pred

demo = gr.Interface(
    fn=predict_category,
    inputs=gr.Textbox(lines=2, placeholder="Enter a headline..."),
    outputs="text",
    title="📰 News Headline Classifier",
    description="Type a news headline and get its predicted category."
)

import os

if __name__ == "__main__":
    demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)))
