import streamlit as st
import pandas as pd
import pickle
import nltk
import re
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import random
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


# Set page config
st.set_page_config(page_title="Dashboard Sentimen", layout="wide")

# Download stopwords if not already
nltk.download('stopwords')

# ==== TEXT PROCESSING FUNCTIONS ====
def casefolding(komen):
    komen = komen.lower()
    komen = komen.strip(" ")
    komen = re.sub(r'(<[A-Za-z0-9]+>)|(#[A-Za-z0-9]+)|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', "", komen)
    return komen

def token(komen):
    return [word for word in komen.split() if word != ""]

def stopword_removal(komen):
    filtering = stopwords.words("indonesian")
    return [word for word in komen if word not in filtering]

def stemming(komen):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemmed = [stemmer.stem(word) for word in komen]
    return " ".join(stemmed)

def preprocessing(text):
    cf = casefolding(text)
    tk = token(cf)
    sw = stopword_removal(tk)
    st = stemming(sw)
    return st

# ==== LOAD SAVED MODEL AND TF-IDF ====
try:
    model = joblib.load("model/naive_bayes_model.pkl")
    tfidf = joblib.load("model/vectorizer.pkl")
except:
    model = None
    tfidf = None

# ==== SIDEBAR MENU ====
with st.sidebar:
    st.markdown(
        """
        <h2 style='text-align: center; color: #3b55f6; font-weight: bold;'>
            ANALISIS SENTIMEN
        </h2>
        <hr style='margin-top: -10px; margin-bottom: 10px;'>
        """,
        unsafe_allow_html=True
    )

    selected = option_menu(
        menu_title=None,
        options=["Data Crawling","Evaluasi Model","Prediksi", "Visualisasi"],
        icons=["plus-circle", "bar-chart", "lightning", "pie-chart"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#fafafa"},
            "icon": {"color": "black", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"2px"},
            "nav-link-selected": {"background-color": "#3b55f6", "color": "white"},
        }
    )


# ==== DATA ====
if selected == "Data Crawling":
    st.title("Data Crawling")

    try:
        # Otomatis baca file tanpa perlu upload manual
        df = pd.read_csv("data/data_crawling.csv", sep=';', encoding='utf-8', on_bad_lines='skip')
        st.success("Data crawling berhasil dimuat.")
        st.dataframe(df.head(5)) # Tampilkan 5 baris pertama

        # Tambahkan tombol download jika ingin
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Data Crawling",
            data=csv_data,
            file_name="data_crawling.csv",
            mime="text/csv"
        )

    except FileNotFoundError:
        st.error("File 'data_crawling.csv' tidak ditemukan. Harap simpan di folder yang sama.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca data: {e}")

# ==== EVALUASI MODEL =====
elif selected == "Evaluasi Model":
    st.title("Evaluasi Model Klasifikasi")

    try:
        df = pd.read_csv("data/data_labeled.csv", sep=";", encoding="utf-8", on_bad_lines="skip")
        st.success("Data berhasil dimuat.")
    except Exception as e:
        st.error(f"Gagal membaca file CSV: {e}")
        st.stop()

    if "full_text" not in df.columns or "sentiment" not in df.columns:
        st.error("File harus memiliki kolom 'full_text' dan 'sentiment'")
        st.stop()

    st.dataframe(df[["full_text", "sentiment"]].head(5))

    # === PILIH METODE DAN TOMBOL ===
    st.markdown("### Pilih Metode Klasifikasi")
    selected_model = st.selectbox("Metode", ["Naive Bayes", "KNN", "SVM"])
    run_button = st.button("🔍 Jalankan Evaluasi")

    if run_button:
        # === PERSIAPAN DATA ===
        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform(df["full_text"].fillna("").astype(str))
        y = df["sentiment"].astype(str)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_dict = {
            "SVM": LinearSVC(),
            "Naive Bayes": MultinomialNB(),
            "KNN": KNeighborsClassifier(n_neighbors=5),
        }

        model = model_dict[selected_model]

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
            report = classification_report(y_test, y_pred)

            st.markdown(f"## Evaluasi Model: {selected_model}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Akurasi", f"{acc:.2f}")
            col2.metric("Precision", f"{prec:.2f}")
            col3.metric("Recall", f"{rec:.2f}")
            col4.metric("F1-Score", f"{f1:.2f}")

            st.markdown("### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()),
                        annot_kws={'size': 8})
            plt.xlabel("Prediksi")
            plt.ylabel("Aktual")
            st.pyplot(fig)

            st.markdown("### Classification Report")
            st.text(report)

        except Exception as e:
            st.error(f" Model {selected_model} gagal dievaluasi: {e}")

# ==== PREDIKSI ====
if selected == "Prediksi":
    st.title("Prediksi Analisis Sentimen")

    input_text = st.text_input("Masukkan kalimat komentar:")

    if st.button("Prediksi Sentimen"):
        if input_text.strip() == "":
            st.warning("Silakan masukkan komentar terlebih dahulu.")
        else:
            # Preprocessing input
            cleaned_text = preprocessing(input_text)

            # Load model & vectorizer
            try:
                model = joblib.load("model/svm_model.pkl")
                tfidf = joblib.load("model/vectorizer.pkl")
            except:
                st.error("Model atau vectorizer belum tersedia.")
                st.stop()

            # Transform input
            input_vector = tfidf.transform([cleaned_text])
            prediction = str(model.predict(input_vector)[0]).strip()

            # Pemetaan angka ke label teks
            label_mapping = {
                "0": "negatif",
                "1": "positif",
                "2": "netral"
            }
            label = label_mapping.get(prediction, "unknown")         

            # Tampilkan hasil prediksi
            if label == "positive":
                st.success("Prediksi Sentimen: **POSITIVE**")
            elif label == "negatif":
                st.error("Prediksi Sentimen: **NEGATIVE**")
            elif label == "netral":
                st.info("Prediksi Sentimen: **NEUTRAL**")
            else:
                st.warning(f"Hasil sentimen : {prediction}")

                            

# === VISUALISASI ===
elif selected == "Visualisasi":
    st.title("Visualisasi WordCloud")

    try:
        df = pd.read_csv("data/data_labeled.csv", sep=';', encoding='utf-8', on_bad_lines='skip')
        st.success(" Data berhasil dimuat dari file lokal.")
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

    # Normalisasi label sentimen
    df['sentiment_label'] = df['sentiment_label'].astype(str).str.strip().str.lower()

    # Daftar kategori sentimen
    kategori = ['positive', 'neutral', 'negative']

    # Fungsi warna acak untuk WordCloud
    def random_color_func(word=None, font_size=None, position=None, orientation=None, font_path=None, random_state=None):
        return "hsl({}, 100%, 40%)".format(random.randint(0, 360))

    # Buat dan tampilkan WordCloud untuk tiap label sentimen
    for label in kategori:
        text_data = df[df['sentiment_label'] == label]['full_text'].dropna().astype(str).str.cat(sep=' ')
        if text_data.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='cividis').generate(text_data)
            wordcloud.recolor(color_func=random_color_func)

            st.markdown(f"### WordCloud - {label.capitalize()}")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.warning(f"Tidak ada data untuk label '{label}'.")


    # =================4. PERBANDINGAN MODEL =================
    st.markdown("---")
    st.subheader("Perbandingan Model")

    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['full_text'].astype(str))
    y = df['sentiment_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_dict = {
        "SVM": LinearSVC(),
        "Naive Bayes": MultinomialNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    metrics = {"Model": [], "Akurasi": [], "Precision": [], "Recall": [], "F1-Score": []}

    for name, model in model_dict.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics["Model"].append(name)
            metrics["Akurasi"].append(accuracy_score(y_test, y_pred))
            metrics["Precision"].append(precision_score(y_test, y_pred, average='macro', zero_division=0))
            metrics["Recall"].append(recall_score(y_test, y_pred, average='macro', zero_division=0))
            metrics["F1-Score"].append(f1_score(y_test, y_pred, average='macro', zero_division=0))
        except Exception as e:
            st.warning(f" Model {name} gagal dievaluasi: {e}")

    df_metrics = pd.DataFrame(metrics)

    if not df_metrics.empty:
        # Tabel perbandingan metrik
        st.dataframe(df_metrics.set_index("Model").style.highlight_max(axis=0, color='lightgreen'))

    # Vectorize dan split data
    tfidf = TfidfVectorizer()
    df['full_text'] = df['full_text'].fillna('')
    X = tfidf.fit_transform(df['full_text'])
    y = df['sentiment_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifiers = {
        "SVM": LinearSVC(),
        "Naive Bayes": MultinomialNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    results = {}
    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        results[name] = {
            'accuracy': acc,
            'average_precision': prec,
            'average_recall': rec,
            'average_f1_score': f1
        }

    # Buat DataFrame
    comparison_df = pd.DataFrame({
        "Model": list(results.keys()),
        "Accuracy": [results[m]['accuracy'] for m in results],
        "Precision": [results[m]['average_precision'] for m in results],
        "Recall": [results[m]['average_recall'] for m in results],
        "F1-Score": [results[m]['average_f1_score'] for m in results],
    })

    # Plot bar
    fig, ax = plt.subplots(figsize=(10, 4))
    comparison_df.set_index("Model").plot(kind="bar", ax=ax, colormap="viridis")
    ax.set_ylabel("Skor")
    ax.set_ylim(0, 1.0)
    ax.set_title(" Grafik Akurasi, Precision, Recall, dan F1-Score")
    ax.grid(axis='y')
    plt.xticks(rotation=0)
    st.pyplot(fig)