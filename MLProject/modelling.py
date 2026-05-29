import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ✅ FIX 1: Tentukan tracking URI yang konsisten
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# ✅ FIX 2: Arahkan ke experiment yang benar
mlflow.set_experiment("spam")

# ✅ FIX 3: Pindahkan autolog ke sini
mlflow.sklearn.autolog()

df = pd.read_csv("data_clean.csv")

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