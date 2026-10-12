
import os
import joblib

from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = os.path.join(
    "model",
    "fake_news_model.pkl"
)

# Load the trained model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found. Run 'python train_model.py' first."
    )

model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form.get("news_text", "").strip()

    if not news_text:
        return render_template(
            "index.html",
            error="Please enter some news text."
        )

    if len(news_text) < 30:
        return render_template(
            "index.html",
            error="Please enter a longer news article for analysis."
        )

    try:
        prediction = int(model.predict([news_text])[0])

        if prediction == 0:
            result = "Likely Fake News"
            explanation = (
                "The model classified this text as fake based "
                "on patterns learned from its training data."
            )
            result_class = "fake"
        else:
            result = "Likely Real News"
            explanation = (
                "The model classified this text as real based "
                "on patterns learned from its training data."
            )
            result_class = "real"

        return render_template(
            "index.html",
            news_text=news_text,
            result=result,
            explanation=explanation,
            result_class=result_class
        )

    except Exception:
        return render_template(
            "index.html",
            error="Prediction failed. Please try again."
        )


if __name__ == "__main__":
    app.run(debug=True)
