# Sentiment Analysis App

![Python](https://img.shields.io/badge/python-3.x-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)


[Python Version](https://www.python.org/)
[Streamlit](https://streamlit.io/)
[Scikit-Learn](https://scikit-learn.org/)

A simple web application built with Streamlit that predicts the sentiment of a given text using a pre-trained Naive Bayes classifier.

## Performance Metrics

The model was evaluated on the test dataset with the following results:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 0.596 |
| **Recall** | 0.555 |
| **Precision** | 0.737 |
| **F1 Score** | 0.561 |

## Getting Started

### Prerequisites

*   Python 3.x installed.
*   Required libraries (install via requirements.txt).

### Installation

1. Clone the repository:
   git clone <your-repository-url>
   cd sentimentAnalysis

2. Install dependencies:
   pip install -r requirements.txt

### Usage

1. **Train the model (optional):**
   If you wish to retrain the model, run:
   python train.py
   This will generate/overwrite predictor.pkl.

2. **Run the Streamlit App:**
   streamlit run streamlit_app.py

3. Open the provided local URL in your browser to interact with the app.

## Tech Stack

*   **Frontend:** Streamlit
*   **Machine Learning:** Scikit-Learn (Naive Bayes, TF-IDF Vectorizer)
*   **Data Handling:** Pandas, NumPy
*   **Model Persistence:** Joblib
