# Sentiment Analysis for Consumer Behavior Prediction

## 📌 Project Overview  
This project performs **Sentiment Analysis** on customer reviews using **Natural Language Processing (NLP)** techniques. The dataset used is the **IMDB Dataset**, and the goal is to classify reviews as **positive** or **negative**.  

### 🚀 Key Features  
- **Text Preprocessing:** HTML tag removal, case normalization, and stopword removal  
- **Feature Engineering:** TF-IDF Vectorization  
- **Machine Learning Models:**
  - Logistic Regression
  - Passive Aggressive Classifier
  - Naïve Bayes (Baseline)  
- **Performance Evaluation:** Accuracy, Precision, Recall, and F1-score  

### 📊 Model Comparison  
| Model                           | Accuracy |
|--------------------------------|----------|
| Logistic Regression            | 83.3%    |
| Passive Aggressive Classifier  | 82.2%    |
| Naïve Bayes (Baseline)         | 83.2%    |

### 🔍 Error Analysis  
A further error analysis showed that **both Logistic Regression and Naïve Bayes performed almost equally well**, meaning that for lightweight implementations, **Naïve Bayes can be a great choice** due to its speed.  

## 🔧 Tech Stack  
- **Python**  
- **Scikit-learn**  
- **Pandas, NumPy**  
- **Matplotlib for visualizations**  

## 📂 Future Improvements  
✔️ Visualizing sentiment distribution  
✔️ Adding deep learning models (BERT, LSTM) for better accuracy  
✔️ Deploying as an API using Flask or FastAPI  

## 🚀 How to Run  
```bash
pip install -r requirements.txt
python sentiment_analysis.py
