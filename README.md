
# AI Fake News Detection System

A machine learning web application that classifies news text as Likely Fake or Likely Real using TF-IDF and Logistic Regression.

## Features

- News text input through a web interface
- Machine learning-based text classification
- TF-IDF text feature extraction
- Logistic Regression classifier
- Prediction results displayed on a Flask website
- Responsive user interface

## Technologies Used

- Python
- Flask
- Pandas
- Scikit-learn
- Joblib
- HTML, CSS and JavaScript

## Project Structure

```text
AI-Fake-News-Detection/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── Fake.csv
├── True.csv
├── model/
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── script.js
```

## Setup Instructions

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Add the dataset

Place `Fake.csv` and `True.csv` in the project root folder.

Both files must contain a `text` column. Use a dataset with documented sources and labels.

### 3. Train the model

```bash
python train_model.py
```

This creates `model/fake_news_model.pkl`.

### 4. Start the web application

```bash
python app.py
```

### 5. Open the application

Visit http://127.0.0.1:5000

## How It Works

1. The dataset supplies labeled fake and real news examples.
2. TF-IDF converts text into numerical features.
3. Logistic Regression learns patterns from the training data.
4. The trained pipeline is saved using Joblib.
5. Flask loads the pipeline and predicts the label of submitted news.

## Limitations

- Predictions may be incorrect or biased by the training data.
- Text classification does not independently verify facts.
- Results depend on dataset quality and how well it represents real-world news.
- The application should not be used as the sole source for deciding whether news is true.

## Disclaimer

This is an educational machine learning project. Always verify important claims using reliable independent sources.
