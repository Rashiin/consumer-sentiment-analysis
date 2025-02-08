# Import necessary libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Download stopwords (در صورت اولین اجرای کد)
nltk.download('stopwords')

# 1. Load the dataset
# اطمینان حاصل کنید که فایل 'IMDB_Dataset.csv' در مسیر مناسب قرار دارد.
data = pd.read_csv('IMDB_Dataset.csv')

# برای پروژه‌ای سبک، از یک زیرمجموعه از داده‌ها استفاده می‌کنیم
data = data.sample(5000, random_state=42)  # انتخاب 5000 نمونه تصادفی

# 2. Preprocess the text data
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # تبدیل به حروف کوچک
    text = text.lower()
    # حذف تگ‌های HTML
    text = re.sub(r'<[^>]+>', '', text)
    # حذف کاراکترهای غیرحرفی (فقط حروف a-z باقی می‌مانند)
    text = re.sub(r'[^a-z]', ' ', text)
    # حذف فاصله‌های اضافه
    text = re.sub(r'\s+', ' ', text)
    # حذف کلمات توقف (stopwords)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# اعمال تابع پاکسازی روی ستون 'review'
data['cleaned_review'] = data['review'].apply(clean_text)

# 3. تبدیل برچسب‌های احساس به مقادیر عددی (مثلاً positive=1 و negative=0)
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

# 4. تقسیم داده به مجموعه‌های آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(
    data['cleaned_review'], data['sentiment'], test_size=0.2, random_state=42
)

# 5. تبدیل متن به بردار ویژگی با TF-IDF
# محدود کردن تعداد ویژگی‌ها برای کاهش مصرف منابع
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 6. آموزش مدل سبک با Logistic Regression
clf = LogisticRegression(max_iter=200)
clf.fit(X_train_tfidf, y_train)

# 7. پیش‌بینی و ارزیابی مدل
y_pred = clf.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --------------------------------------------------
# بخش‌های تکمیلی برای نمایش نتایج بیشتر:
# --------------------------------------------------

# 1. نمایش ماتریس سردرگمی (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 2. محاسبه و نمایش ROC AUC Score و منحنی ROC (در صورتی که متد predict_proba وجود داشته باشد)
if hasattr(clf, "predict_proba"):
    y_proba = clf.predict_proba(X_test_tfidf)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print("ROC AUC Score:", auc)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

# 3. استفاده از اعتبارسنجی متقابل (Cross-Validation) برای ارزیابی پایداری مدل
cv_scores = cross_val_score(clf, X_train_tfidf, y_train, cv=5, scoring='accuracy')
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())

# 4. جستجوی شبکه‌ای برای بهینه‌سازی هایپرپارامترها (اختیاری ولی مفید برای بهبود عملکرد)
parameters = {
    'C': [0.1, 1, 10],
    'max_iter': [100, 200]
}
grid_search = GridSearchCV(LogisticRegression(), parameters, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_tfidf, y_train)
print("Best parameters found:", grid_search.best_params_)

# 5. نمونه پیش‌بینی بر روی یک متن دلخواه (مثلاً نظرات مصرف‌کننده)
sample_text = "I absolutely loved this product! It exceeded my expectations."
sample_text_cleaned = clean_text(sample_text)
sample_text_vectorized = vectorizer.transform([sample_text_cleaned])
sample_prediction = clf.predict(sample_text_vectorized)
print("Sample Prediction:", "Positive" if sample_prediction[0] == 1 else "Negative")
