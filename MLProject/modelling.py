import os 
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("spam")

mlflow.sklearn.autolog()

base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "data_clean.csv")

df = pd.read_csv(file_path)

print(df.head())
print(df.columns)

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("nb", MultinomialNB())
])

with mlflow.start_run():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    
    # ✅ FIX 4: Log metric & model secara eksplisit (backup dari autolog)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "spam-model")

    print("Accuracy:", accuracy)
