import pandas as pd
import re
import string
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# ====== 1. Load Dataset ======
df = pd.read_excel("data/labeled_dataset_indo_3class.xlsx")
df = df[['full_text', 'sentiment_label']].dropna()

# ====== 2. Preprocessing ======
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

df['clean_text'] = df['full_text'].apply(preprocess)

# ====== 3. TF-IDF ======
X = df['clean_text']
y = df['sentiment_label']
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# ====== 4. Train/Test Split ======
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# ====== 5. Model Definitions ======
models = {
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC(),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# ====== 6. Train & Evaluate ======
os.makedirs("model", exist_ok=True)

for name, model in models.items():
    print(f"\n===== {name} =====")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy      :", f"{accuracy_score(y_test, y_pred):.2f}")
    print("Precision     :", f"{precision_score(y_test, y_pred, average='macro', zero_division=0):.2f}")
    print("Recall        :", f"{recall_score(y_test, y_pred, average='macro', zero_division=0):.2f}")
    print("F1 Score      :", f"{f1_score(y_test, y_pred, average='macro', zero_division=0):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Simpan model
    model_filename = f"model/{name.lower().replace(' ', '_')}_model.pkl"
    joblib.dump(model, model_filename)

# Simpan vectorizer satu kali saja
joblib.dump(vectorizer, "model/vectorizer.pkl")
print("\n✅ Semua model & vectorizer berhasil disimpan ke folder 'model'.")
