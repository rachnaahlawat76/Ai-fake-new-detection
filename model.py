
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# File paths
FAKE_FILE = "Fake.csv"
REAL_FILE = "True.csv"
MODEL_DIR = "model"

# Check whether both dataset files exist
if not os.path.exists(FAKE_FILE) or not os.path.exists(REAL_FILE):
    raise FileNotFoundError(
        "Fake.csv and True.csv must be in the project folder."
    )


# Load the datasets
fake_news = pd.read_csv(FAKE_FILE)
real_news = pd.read_csv(REAL_FILE)


# Check that the text column exists
if "text" not in fake_news.columns or "text" not in real_news.columns:
    raise ValueError("Both CSV files must contain a 'text' column.")


# Assign labels: 0 = Fake, 1 = Real
fake_news["label"] = 0
real_news["label"] = 1

# Combine both datasets
data = pd.concat(
    [fake_news, real_news],
    ignore_index=True
)

# Remove missing and empty news text
data = data.dropna(subset=["text"])
data["text"] = data["text"].astype(str).str.strip()
data = data[data["text"] != ""]

# Remove duplicate news articles
data = data.drop_duplicates(subset=["text"])

# Shuffle the dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Use news text as input and labels as output
X = data["text"]
y = data["label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create the text classification pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_df=0.7,
            max_features=50000,
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

# Train the model
print("Training the model...")
model.fit(X_train, y_train)

# Evaluate the model on unseen test data
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel training completed!")
print(f"Test accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Fake", "Real"],
        zero_division=0
    )
)

# Save the trained model
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, os.path.join(MODEL_DIR, "fake_news_model.pkl"))

print("\nModel saved successfully in the model folder.")
