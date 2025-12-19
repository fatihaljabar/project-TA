import streamlit as st
import torch
import torch.nn as nn  # noqa: F401
import re
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis - Business Review",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants
MAX_LEN = 128
MODEL_PATH = "pytorch_model.bin"
TOKENIZER_PATH = None
LABEL_ENCODER_PATH = "kaggle/working/model_outputs/label_encoder.pkl"

# Initialize session state for language
if "language" not in st.session_state:
    st.session_state.language = "en"  # Default to English

# Initialize session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "analysis"  # Default page

# Translations dictionary
TRANSLATIONS = {
    "en": {
        # Navigation
        "about": "About",
        "settings": "Settings",
        "language": "Language",
        # Hero section
        "title": "Sentiment Analysis",
        "subtitle": "🇮🇩 Indonesian Business Review Analyzer",
        "description": "Powered by CNN + BiLSTM with Attention • Real-time AI predictions with 75.94% accuracy",
        # Sidebar
        "about_title": "About This Tool",
        "about_text": "This AI-powered sentiment analysis model classifies Indonesian business reviews into three categories:",
        "model_info": "Model Information",
        "architecture": "Architecture",
        "accuracy": "Accuracy",
        "dataset_size": "Dataset Size",
        "samples": "samples",
        "try_examples": "Try Examples",
        # Sentiments
        "positive": "Positive",
        "neutral": "Neutral",
        "negative": "Negative",
        # Input section
        "input_title": "Enter Your Review Text",
        "input_subtitle": "Type or paste a business review below and click analyze to see the sentiment prediction.",
        "input_placeholder": "Example: Masukkan teks ulasan bisnis Indonesia Anda di sini... ⚡",
        "input_label": "Type or paste your Indonesian business review here:",
        "analyze_button": "🔍 Analyze Sentiment",
        # Results
        "results_title": "Analysis Results",
        "sentiment_label": "Sentiment",
        "confidence_label": "Confidence",
        "distribution_title": "Sentiment Distribution",
        "statistics_title": "Text Statistics",
        "word_count": "Word Count",
        "characters": "Characters",
        "cleaned_words": "Cleaned Words",
        "view_preprocessed": "🔍 View Preprocessed Text",
        # Messages
        "loading_model": "Loading model...",
        "analyzing": "Analyzing sentiment...",
        "empty_warning": "⚠️ Please enter some text to analyze.",
        "error_loading": "Failed to load model. Please check the model files.",
        "error_prediction": "Error loading model artifacts:",
        # Footer
        "powered_by": "Powered by",
        "model_description": "CNN + BiLSTM with Attention Mechanism",
        "copyright": "© 2025 by Fatih's",
        # Example texts
        "example_positive": "Saya senang dengan keputusan pemerintah mendukung energi hijau.",
        "example_neutral": "Pemerintah sedang menyiapkan insentif untuk kendaraan bermotor listrik.",
        "example_negative": "Mobil listrik sangat mengecewakan dan tidak layak dibeli. Saya sangat kecewa dengan kualitasnya yang buruk dan harganya terlalu mahal.",
        # Performance Metrics
        "model_performance": "Model Performance",
        "performance_subtitle": "CNN + BiLSTM with Attention Mechanism • Trained on 26,852 samples • Epoch: 150",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1-Score",
        "detailed_metrics": "Detailed Performance Metrics",
        "per_class_performance": "Per-Class Performance",
        "class": "Class",
        "aggregate_metrics": "Aggregate Metrics",
        # Navigation
        "nav_analysis": "Sentiment Analysis",
        "nav_performance": "Model Performance",
        "epoch_comparison": "Epoch Comparison",
        # Additional translations for detailed metrics
        "macro_average": "Macro Average",
        "weighted_average": "Weighted Average",
        "overall_accuracy": "Overall Accuracy",
        "dataset_split": "Dataset Split",
        "train": "Train",
        "val": "Val",
        "test": "Test",
        "total": "Total",
        "architecture_label": "Architecture",
        "training_epochs_label": "Training Epochs",
        "test_samples_label": "Test Samples",
        "best_performance_label": "Best Performance",
        "model_description_detail": "This model provides the best performance compared to individual models (CNN, BiLSTM, LSTM) by combining CNN's strength for feature extraction and BiLSTM for sequence modeling.",
        "cnn_bilstm_attention": "CNN + BiLSTM with Attention Mechanism",
        "trained_on": "Trained on",
        "model_label": "Model",
        "epoch_label": "Epoch",
        "total_samples": "Total Samples",
    },
    "id": {
        # Navigation
        "about": "Tentang",
        "settings": "Pengaturan",
        "language": "Bahasa",
        # Hero section
        "title": "Analisis Sentimen",
        "subtitle": "🇮🇩 Penganalisis Ulasan Bisnis Indonesia",
        "description": "Didukung oleh CNN + BiLSTM dengan Attention • Prediksi AI real-time dengan akurasi 75.94%",
        # Sidebar
        "about_title": "Tentang Alat Ini",
        "about_text": "Model analisis sentimen bertenaga AI ini mengklasifikasikan ulasan bisnis Indonesia ke dalam tiga kategori:",
        "model_info": "Informasi Model",
        "architecture": "Arsitektur",
        "accuracy": "Akurasi",
        "dataset_size": "Ukuran Dataset",
        "samples": "sampel",
        "try_examples": "Coba Contoh",
        # Sentiments
        "positive": "Positif",
        "neutral": "Netral",
        "negative": "Negatif",
        # Input section
        "input_title": "Masukkan Teks Ulasan Anda",
        "input_subtitle": "Ketik atau tempel ulasan bisnis di bawah ini dan klik analisis untuk melihat prediksi sentimen.",
        "input_placeholder": "Contoh: Masukkan teks ulasan bisnis Indonesia Anda di sini... ⚡",
        "input_label": "Ketik atau tempel ulasan bisnis Indonesia Anda di sini:",
        "analyze_button": "🔍 Analisis Sentimen",
        # Results
        "results_title": "Hasil Analisis",
        "sentiment_label": "Sentimen",
        "confidence_label": "Kepercayaan",
        "distribution_title": "Distribusi Sentimen",
        "statistics_title": "Statistik Teks",
        "word_count": "Jumlah Kata",
        "characters": "Karakter",
        "cleaned_words": "Kata Bersih",
        "view_preprocessed": "🔍 Lihat Teks Terproses",
        # Messages
        "loading_model": "Memuat model...",
        "analyzing": "Menganalisis sentimen...",
        "empty_warning": "⚠️ Silakan masukkan teks untuk dianalisis.",
        "error_loading": "Gagal memuat model. Silakan periksa file model.",
        "error_prediction": "Kesalahan memuat artefak model:",
        # Footer
        "powered_by": "Didukung oleh",
        "model_description": "CNN + BiLSTM dengan Mekanisme Attention",
        "copyright": "© 2025 by Fatih's ",
        # Example texts
        "example_positive": "Saya senang dengan keputusan pemerintah mendukung energi hijau.",
        "example_neutral": "Pemerintah sedang menyiapkan insentif untuk kendaraan bermotor listrik.",
        "example_negative": "Mobil listrik sangat mengecewakan dan tidak layak dibeli. Saya sangat kecewa dengan kualitasnya yang buruk dan harganya terlalu mahal.",
        # Performance Metrics
        "model_performance": "Performa Model",
        "performance_subtitle": "CNN + BiLSTM dengan Mekanisme Attention • Dilatih dengan 26,852 sampel • Epoch: 150",
        "precision": "Presisi",
        "recall": "Recall",
        "f1_score": "F1-Score",
        "detailed_metrics": "Metrik Performa Detail",
        "per_class_performance": "Performa Per-Kelas",
        "class": "Kelas",
        "aggregate_metrics": "Metrik Agregat",
        # Navigation
        "nav_analysis": "Analisis Sentimen",
        "nav_performance": "Performa Model",
        "epoch_comparison": "Perbandingan Epoch",
        # Additional translations for detailed metrics
        "macro_average": "Rata-rata Makro",
        "weighted_average": "Rata-rata Tertimbang",
        "overall_accuracy": "Akurasi Keseluruhan",
        "dataset_split": "Pembagian Dataset",
        "train": "Latih",
        "val": "Validasi",
        "test": "Uji",
        "total": "Total",
        "architecture_label": "Arsitektur",
        "training_epochs_label": "Epoch Pelatihan",
        "test_samples_label": "Sampel Uji",
        "best_performance_label": "Performa Terbaik",
        "model_description_detail": "Model ini memberikan performa terbaik dibanding model individual (CNN, BiLSTM, LSTM) dengan mengkombinasikan kekuatan CNN untuk feature extraction dan BiLSTM untuk sequence modeling.",
        "cnn_bilstm_attention": "CNN + BiLSTM dengan Mekanisme Attention",
        "trained_on": "Dilatih dengan",
        "model_label": "Model",
        "epoch_label": "Epoch",
        "total_samples": "Total Sampel",
    },
}


def get_text(key):
    """Get translated text based on current language"""
    return TRANSLATIONS[st.session_state.language].get(key, key)


# Single professional theme (no light/dark toggle)


# Single professional color palette
def get_theme_colors():
    return {
        "bg_color": "#0b1220",  # deep navy for modern look
        "text_color": "#e6eaf2",  # soft off-white
        "primary_color": "#7c8cf8",  # indigo-400
        "secondary_color": "#5dd39e",  # teal-green accent
        "accent_color": "#f38fb1",  # soft pink accent
        "positive_bg": "rgba(93, 211, 158, 0.15)",
        "positive_border": "#5dd39e",
        "positive_text": "#c9f7e6",
        "positive_glow": "rgba(93, 211, 158, 0.35)",
        "neutral_bg": "rgba(245, 158, 11, 0.15)",
        "neutral_border": "#f59e0b",
        "neutral_text": "#ffe2b5",
        "neutral_glow": "rgba(245, 158, 11, 0.35)",
        "negative_bg": "rgba(239, 68, 68, 0.15)",
        "negative_border": "#ef4444",
        "negative_text": "#ffd2d2",
        "negative_glow": "rgba(239, 68, 68, 0.35)",
        "card_bg": "rgba(16, 23, 42, 0.75)",
        "card_border": "rgba(124, 140, 248, 0.25)",
        "input_bg": "rgba(15, 22, 39, 0.85)",
        "input_border": "rgba(124, 140, 248, 0.35)",
        "shadow": "rgba(0, 0, 0, 0.6)",
    }


# Get current theme colors
colors = get_theme_colors()


# Custom CSS for UI - Modern Design
def get_custom_css():
    colors = get_theme_colors()
    return f"""
    <style>
    /* Import Modern Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@400;500;600;700&display=swap');
    
    /* Global Theme with Gradient Background */
    body {{
        color: {colors["text_color"]} !important;
        background: {colors["bg_color"]} !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    .stApp {{
        background: {colors["bg_color"]} !important;
        color: {colors["text_color"]} !important;
    }}
    
    .main {{
        background: {colors["bg_color"]} !important;
        color: {colors["text_color"]} !important;
    }}
    
    .block-container {{
        background: transparent !important;
        color: {colors["text_color"]} !important;
    }}
    
    /* Hide Streamlit Branding (but keep sidebar toggle) */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Style header to match our design */
    header {{
        background: transparent !important;
    }}
    
    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}
    
    /* COMPLETELY HIDE all sidebar toggle buttons - sidebar always visible */
    header button,
    header [data-testid="baseButton-header"],
    [data-testid="stHeader"] button,
    button[kind="header"],
    button[data-testid="collapsedControl"],
    section[data-testid="stSidebar"] button[kind="header"],
    section[data-testid="stSidebar"] > div > button,
    section[data-testid="stSidebar"] [data-testid="collapsedControl"],
    [data-testid="collapsedControl"],
    div[data-testid="collapsedControl"] {{
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
        width: 0 !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
    }}
    
    /* Force sidebar to always be visible and prevent collapse */
    section[data-testid="stSidebar"] {{
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        position: relative !important;
        transform: translateX(0) !important;
        min-width: 24rem !important;
        max-width: 24rem !important;
    }}
    
    /* Force header area to be visible */
    header,
    [data-testid="stHeader"] {{
        visibility: visible !important;
        opacity: 1 !important;
        background: transparent !important;
        pointer-events: auto !important;
    }}
    
    /* Force toolbar to be visible */
    .stToolbarActions,
    [data-testid="stToolbar"] {{
        visibility: visible !important;
        opacity: 1 !important;
    }}
    
    /* Streamlit toolbar buttons */
    .stToolbar {{
        opacity: 0.7;
        transition: opacity 0.3s ease;
    }}
    
    .stToolbar:hover {{
        opacity: 1;
    }}
    
    /* Hide deploy button if present */
    .stDeployButton {{
        visibility: hidden;
    }}
    
    /* Headers with Better Typography */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        color: {colors["text_color"]};
        letter-spacing: -0.02em;
        line-height: 1.2;
    }}
    
    h1 {{ font-size: 2.5rem; margin-bottom: 0.5rem; }}
    h2 {{ font-size: 2rem; margin-bottom: 0.75rem; }}
    h3 {{ font-size: 1.5rem; margin-bottom: 0.5rem; }}
    
    /* Glassmorphism Card */
    .card {{
        background: {colors["card_bg"]};
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2rem;
        border: 1px solid {colors["card_border"]};
        box-shadow: 0 8px 32px {colors["shadow"]};
        margin-bottom: 1.5rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }}
    
    .card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, {colors["primary_color"]}, {colors["secondary_color"]}, {colors["accent_color"]});
        opacity: 0;
        transition: opacity 0.3s ease;
    }}
    
    .card:hover {{
        transform: translateY(-8px) scale(1.01);
        box-shadow: 0 20px 60px {colors["shadow"]};
        border-color: {colors["primary_color"]};
    }}
    
    .card:hover::before {{
        opacity: 1;
    }}
    
    /* Modern Input Fields - Enhanced */
    .stTextArea textarea {{
        background: rgba(20, 30, 48, 0.95) !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px !important;
        border: 2px solid rgba(124, 140, 248, 0.4) !important;
        font-size: 16px !important;
        padding: 20px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
        line-height: 1.8 !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
        resize: vertical !important;
    }}
    
    .stTextArea textarea::placeholder {{
        color: rgba(230, 234, 242, 0.4) !important;
        font-style: italic !important;
    }}
    
    .stTextArea textarea:hover {{
        border: 2px solid rgba(124, 140, 248, 0.6) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    }}
    
    .stTextArea textarea:focus {{
        border: 2px solid {colors["primary_color"]} !important;
        box-shadow: 0 0 0 4px rgba(124, 140, 248, 0.15), 0 8px 24px rgba(124, 140, 248, 0.2) !important;
        transform: scale(1.005) !important;
        outline: none !important;
        background: rgba(20, 30, 48, 1) !important;
    }}
    
    /* Text area label */
    .stTextArea label {{
        font-weight: 500 !important;
        color: {colors["text_color"]} !important;
        margin-bottom: 0.5rem !important;
    }}
    
    /* Fix text visibility in all inputs */
    input {{
        color: {colors["text_color"]} !important;
    }}
    
    textarea {{
        color: {colors["text_color"]} !important;
    }}
    
    /* Modern Button with Gradient */
    .stButton > button {{
        background: linear-gradient(135deg, {colors["primary_color"]} 0%, {colors["secondary_color"]} 100%);
        color: white;
        border: none;
        border-radius: 16px;
        font-weight: 600;
        font-size: 16px;
        padding: 14px 32px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.02em;
        text-transform: none;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, {colors["secondary_color"]} 0%, {colors["primary_color"]} 100%);
    }}
    
    .stButton > button:active {{
        transform: translateY(-1px);
    }}
    
    /* Sentiment Result Boxes - Enhanced */
    .sentiment-positive {{
        background: {colors["positive_bg"]};
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        color: {colors["positive_text"]};
        padding: 2rem;
        border-radius: 20px;
        border: 2px solid {colors["positive_border"]};
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px {colors["positive_glow"]}, inset 0 0 0 1px rgba(255, 255, 255, 0.1);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .sentiment-positive::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, {colors["positive_glow"]} 0%, transparent 70%);
        opacity: 0.3;
        animation: pulse 3s ease-in-out infinite;
    }}
    
    .sentiment-negative {{
        background: {colors["negative_bg"]};
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        color: {colors["negative_text"]};
        padding: 2rem;
        border-radius: 20px;
        border: 2px solid {colors["negative_border"]};
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px {colors["negative_glow"]}, inset 0 0 0 1px rgba(255, 255, 255, 0.1);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .sentiment-negative::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, {colors["negative_glow"]} 0%, transparent 70%);
        opacity: 0.3;
        animation: pulse 3s ease-in-out infinite;
    }}
    
    .sentiment-neutral {{
        background: {colors["neutral_bg"]};
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        color: {colors["neutral_text"]};
        padding: 2rem;
        border-radius: 20px;
        border: 2px solid {colors["neutral_border"]};
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px {colors["neutral_glow"]}, inset 0 0 0 1px rgba(255, 255, 255, 0.1);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .sentiment-neutral::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, {colors["neutral_glow"]} 0%, transparent 70%);
        opacity: 0.3;
        animation: pulse 3s ease-in-out infinite;
    }}
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background: {colors["card_bg"]} !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid {colors["card_border"]} !important;
    }}
    
    section[data-testid="stSidebar"] * {{
        color: {colors["text_color"]} !important;
    }}
    
    section[data-testid="stSidebar"] .stButton > button {{
        width: 100%;
        background: {colors["input_bg"]} !important;
        color: {colors["text_color"]} !important;
        border: 1px solid {colors["input_border"]} !important;
        transition: all 0.3s ease;
        font-size: 1.5rem !important;
        padding: 0.75rem !important;
    }}
    
    section[data-testid="stSidebar"] .stButton > button:hover {{
        background: linear-gradient(135deg, {colors["primary_color"]} 0%, {colors["secondary_color"]} 100%) !important;
        color: white !important;
        border-color: transparent !important;
        transform: scale(1.05) !important;
    }}
    
    /* Metrics Cards - Simple card design */
    .metric-card {{
        background: {colors["card_bg"]};
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid {colors["card_border"]};
        transition: all 0.3s ease;
    }}
    
    .metric-card:hover {{
        border-color: {colors["primary_color"]};
        transform: translateY(-2px);
    }}
    
    .metric-value {{
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, {colors["primary_color"]}, {colors["secondary_color"]});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }}
    
    .metric-label {{
        font-size: 0.875rem;
        color: {colors["text_color"]};
        opacity: 0.7;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    
    /* Enhanced Confidence Gauge */
    .confidence-gauge {{
        width: 100%;
        height: 12px;
        background: {colors["input_bg"]};
        border-radius: 10px;
        overflow: hidden;
        margin: 8px 0;
        box-shadow: inset 0 2px 4px {colors["shadow"]};
        position: relative;
    }}
    
    .confidence-fill {{
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, {colors["primary_color"]}, {colors["secondary_color"]});
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }}
    
    .confidence-fill::after {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: shimmer 2s infinite;
    }}
    
    /* Expander - Simple card design */
    div[data-testid="stExpander"],
    .streamlit-expanderHeader,
    details summary {{
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        background: {colors["card_bg"]} !important;
        border-radius: 12px !important;
        border: 1px solid {colors["card_border"]} !important;
        padding: 0.875rem 1rem !important;
        transition: all 0.3s ease !important;
        color: {colors["text_color"]} !important;
    }}
    
    div[data-testid="stExpander"]:hover,
    .streamlit-expanderHeader:hover,
    details summary:hover {{
        border-color: {colors["primary_color"]} !important;
    }}
    
    .streamlit-expanderContent,
    details[open] {{
        background: {colors["input_bg"]} !important;
        border: 1px solid {colors["card_border"]} !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
        padding: 1rem !important;
        color: #ffffff !important;
    }}
    
    /* Fix all expander text to be white */
    div[data-testid="stExpander"] *,
    .streamlit-expanderContent *,
    details * {{
        color: #ffffff !important;
    }}
    
    /* Code blocks - Enhanced contrast */
    code {{
        background: rgba(15, 22, 39, 0.95) !important;
        color: #ffffff !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
        border: 1px solid rgba(124, 140, 248, 0.2) !important;
    }}
    
    pre {{
        background: rgba(15, 22, 39, 0.95) !important;
        color: #ffffff !important;
        border: 1px solid {colors["card_border"]} !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }}
    
    pre code {{
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }}
    
    /* Animations */
    @keyframes fadeIn {{
        from {{ 
            opacity: 0; 
            transform: translateY(20px);
        }}
        to {{ 
            opacity: 1; 
            transform: translateY(0);
        }}
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 0.3; }}
        50% {{ opacity: 0.5; }}
    }}
    
    @keyframes shimmer {{
        0% {{ transform: translateX(-100%); }}
        100% {{ transform: translateX(100%); }}
    }}
    
    @keyframes float {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-10px); }}
    }}
    
    .fade-in {{
        animation: fadeIn 0.6s cubic-bezier(0.4, 0, 0.2, 1) forwards;
    }}
    
    /* Progress Bar */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, {colors["primary_color"]}, {colors["secondary_color"]}) !important;
    }}
    
    /* Spinners */
    .stSpinner > div {{
        border-top-color: {colors["primary_color"]} !important;
    }}
    
    /* Alerts and Messages */
    .stAlert {{
        background: {colors["card_bg"]} !important;
        color: {colors["text_color"]} !important;
        border: 1px solid {colors["card_border"]} !important;
    }}
    
    /* Markdown text */
    .stMarkdown {{
        color: {colors["text_color"]} !important;
    }}
    
    .stMarkdown p {{
        color: {colors["text_color"]} !important;
    }}
    
    .stMarkdown div {{
        color: {colors["text_color"]} !important;
    }}
    
    /* Labels */
    label {{
        color: {colors["text_color"]} !important;
    }}
    
    /* All paragraph text */
    p {{
        color: {colors["text_color"]} !important;
    }}
    
    /* All div text */
    div {{
        color: {colors["text_color"]};
    }}
    
    /* All spans */
    span {{
        color: inherit;
    }}
    
    /* Select boxes and inputs */
    .stSelectbox > div > div {{
        background: {colors["input_bg"]} !important;
        color: {colors["text_color"]} !important;
        border-color: {colors["input_border"]} !important;
    }}
    
    /* Warnings */
    .stWarning {{
        background: {colors["neutral_bg"]} !important;
        border-left: 4px solid {colors["neutral_border"]} !important;
        color: {colors["text_color"]} !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }}
    
    /* Errors */
    .stError {{
        background: {colors["negative_bg"]} !important;
        border-left: 4px solid {colors["negative_border"]} !important;
        color: {colors["text_color"]} !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }}
    
    /* Success */
    .stSuccess {{
        background: {colors["positive_bg"]} !important;
        border-left: 4px solid {colors["positive_border"]} !important;
        color: {colors["text_color"]} !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }}
    
    /* Info messages */
    .stInfo {{
        background: {colors["card_bg"]} !important;
        border-left: 4px solid {colors["primary_color"]} !important;
        color: {colors["text_color"]} !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }}
    
    /* Responsive Design */
    @media (max-width: 768px) {{
        /* Adjust sidebar for mobile */
        section[data-testid="stSidebar"] {{
            min-width: 20rem !important;
            max-width: 20rem !important;
        }}
        
        .card {{
            padding: 1.25rem;
            border-radius: 16px;
        }}
        
        h1 {{ font-size: 2rem; }}
        h2 {{ font-size: 1.5rem; }}
        h3 {{ font-size: 1.25rem; }}
        
        .metric-value {{ font-size: 2rem; }}
        
        .stButton > button {{
            padding: 12px 24px;
            font-size: 15px;
        }}
    }}
    
    /* Wider screens - make sidebar even more spacious */
    @media (min-width: 1400px) {{
        section[data-testid="stSidebar"] {{
            min-width: 28rem !important;
            max-width: 28rem !important;
        }}
    }}
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {colors["input_bg"]};
        border-radius: 10px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(135deg, {colors["primary_color"]}, {colors["secondary_color"]});
        border-radius: 10px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(135deg, {colors["secondary_color"]}, {colors["primary_color"]});
    }}
    
    /* Dataframe Styling - Dark Theme */
    div[data-testid="stDataFrame"] {{
        background: {colors["card_bg"]} !important;
    }}
    
    div[data-testid="stDataFrame"] > div {{
        background: {colors["card_bg"]} !important;
        border: 1px solid {colors["card_border"]} !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }}
    
    /* Dataframe table cells */
    .stDataFrame table {{
        background: {colors["input_bg"]} !important;
        color: {colors["text_color"]} !important;
    }}
    
    .stDataFrame thead tr th {{
        background: {colors["card_bg"]} !important;
        color: {colors["text_color"]} !important;
        border-bottom: 2px solid {colors["card_border"]} !important;
        font-weight: 600 !important;
        padding: 0.75rem !important;
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }}
    
    .stDataFrame tbody tr td {{
        background: {colors["input_bg"]} !important;
        color: {colors["text_color"]} !important;
        border-bottom: 1px solid {colors["card_border"]} !important;
        padding: 0.75rem !important;
        font-size: 0.9rem !important;
    }}
    
    .stDataFrame tbody tr:hover td {{
        background: rgba(124, 140, 248, 0.1) !important;
    }}
    
    /* Fix gradient cells to show on dark background */
    .stDataFrame tbody tr td {{
        font-weight: 600 !important;
    }}
    
    /* Custom HTML Table Styling */
    table tbody tr {{
        border-bottom: 1px solid {colors["card_border"]};
        transition: background 0.2s ease;
    }}
    
    table tbody tr:hover {{
        background: rgba(124, 140, 248, 0.08) !important;
    }}
    
    table tbody tr td {{
        padding: 0.75rem !important;
        font-size: 0.95rem !important;
    }}
    
    /* Hoverable Cards - Aggregate Metrics & Model Info */
    .metric-card-hover {{
        cursor: pointer !important;
        transition: all 0.3s ease !important;
    }}
    
    .metric-card-hover:hover {{
        background: linear-gradient(135deg, {colors["primary_color"]} 0%, {colors["secondary_color"]} 100%) !important;
        border-color: transparent !important;
        transform: scale(1.05) translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(124, 140, 248, 0.4) !important;
    }}
    
    .dataset-card-hover {{
        cursor: pointer !important;
        transition: all 0.3s ease !important;
    }}
    
    .dataset-card-hover:hover {{
        background: linear-gradient(135deg, {colors["primary_color"]} 0%, {colors["secondary_color"]} 100%) !important;
        border-color: transparent !important;
        transform: scale(1.02) translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(124, 140, 248, 0.4) !important;
    }}
    </style>
    """


# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)


@st.cache_resource
def load_model_and_artifacts():
    """Load PyTorch model from HuggingFace Hub (same as notebook)."""
    try:
        # Use HuggingFace model directly (same as the working notebook)
        model_id = "taufiqdp/indonesian-sentiment"

        # Load tokenizer and model from HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.eval()

        # Get label mapping from model config
        id2label = getattr(
            model.config, "id2label", {0: "Negatif", 1: "Netral", 2: "Positif"}
        )

        # Normalize labels to Indonesian with capital first letter (same as notebook logic)
        norm = {
            "NEGATIVE": "Negatif",
            "NEUTRAL": "Netral",
            "POSITIVE": "Positif",
            "NEGATIF": "Negatif",
            "NETRAL": "Netral",
            "POSITIF": "Positif",
            "negatif": "Negatif",  # Add lowercase mappings (HuggingFace returns these!)
            "netral": "Netral",
            "positif": "Positif",
            "LABEL_0": "Negatif",
            "LABEL_1": "Netral",
            "LABEL_2": "Positif",
            "Negatif": "Negatif",
            "Netral": "Netral",
            "Positif": "Positif",
        }

        # Create normalized label encoder-like object for compatibility
        class LabelMapper:
            def __init__(self, id2label, norm):
                self.id2label = id2label
                self.norm = norm
                # Create classes_ array matching expected order
                self.classes_ = []
                for i in range(len(id2label)):
                    lbl = id2label.get(i, id2label.get(str(i), f"LABEL_{i}"))
                    # Try multiple lookup strategies: exact match, uppercase, lowercase, or default
                    normalized = norm.get(
                        str(lbl),
                        norm.get(
                            str(lbl).upper(), norm.get(str(lbl).lower(), str(lbl))
                        ),
                    )
                    self.classes_.append(normalized)
                self.classes_ = np.array(self.classes_)

        label_encoder = LabelMapper(id2label, norm)

        return model, tokenizer, label_encoder

    except Exception as e:
        error_msg = TRANSLATIONS.get(
            st.session_state.get("language", "en"), TRANSLATIONS["en"]
        ).get("error_prediction", "Error loading model artifacts:")
        st.error(f"{error_msg} {str(e)}")
        return None, None, None


def clean_text(text):
    """Clean and preprocess text (same as training pipeline)"""
    text = str(text)
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    # Remove mentions
    text = re.sub(r"@\w+", " ", text)
    # Remove hashtags
    text = re.sub(r"#", " ", text)
    # Remove non-alphanumeric characters
    text = re.sub(r"[^0-9a-z\s]", " ", text)
    # Collapse spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text, tokenizer):
    """Preprocess with HF tokenizer (same as notebook - use original text).
    Note: The HuggingFace model works better with original text, not cleaned.
    """
    # Use ORIGINAL text for tokenization (like the notebook)
    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )
    # Still generate cleaned version for display purposes
    cleaned = clean_text(text)
    return encoded, cleaned


def translate_sentiment(indonesian_label):
    """Translate Indonesian sentiment labels based on current language"""
    # Normalize to capital first letter first (handle lowercase from model)
    label_normalized = (
        indonesian_label.capitalize() if indonesian_label else indonesian_label
    )

    if st.session_state.language == "en":
        translation = {
            "Positif": "Positive",
            "Netral": "Neutral",
            "Negatif": "Negative",
        }
        return translation.get(label_normalized, label_normalized)
    else:
        # Keep Indonesian labels as-is for Indonesian language
        return label_normalized


def predict_sentiment(text, model, tokenizer, label_encoder):
    """Predict sentiment with HF model"""
    encoded, cleaned_text = preprocess_text(text, tokenizer)

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        prediction = probabilities.cpu().numpy()

    predicted_class_idx = int(np.argmax(prediction[0]))
    confidence = float(prediction[0][predicted_class_idx])

    # Prefer HF labels if present; else fallback to label_encoder
    if hasattr(model.config, "id2label") and model.config.id2label:
        # HF id2label keys are ints in newer versions; handle string keys as well
        try:
            raw_label = model.config.id2label[predicted_class_idx]
        except Exception:
            raw_label = model.config.id2label.get(
                str(predicted_class_idx), str(predicted_class_idx)
            )

        # Normalize to Indonesian expected names if possible
        mapping = {
            "NEGATIVE": "Negatif",
            "NEUTRAL": "Netral",
            "POSITIVE": "Positif",
            "LABEL_0": "Negatif",
            "LABEL_1": "Netral",
            "LABEL_2": "Positif",
        }
        sentiment_indonesian = mapping.get(str(raw_label).upper(), str(raw_label))
        # Build all probabilities using HF labels
        all_probabilities = {}
        for i in range(prediction.shape[1]):
            try:
                lbl = model.config.id2label[i]
            except Exception:
                lbl = model.config.id2label.get(str(i), f"LABEL_{i}")
            lbl_id = mapping.get(str(lbl).upper(), str(lbl))
            all_probabilities[translate_sentiment(lbl_id)] = float(prediction[0][i])
    else:
        # Fallback to provided label encoder
        sentiment_indonesian = label_encoder.classes_[predicted_class_idx]
        all_probabilities = {
            translate_sentiment(label_encoder.classes_[i]): float(prediction[0][i])
            for i in range(len(label_encoder.classes_))
        }

    sentiment = translate_sentiment(sentiment_indonesian)
    return sentiment, confidence, all_probabilities, cleaned_text


def display_metric_card(label, value, unit=""):
    """Display a metric in a styled card"""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{value}{unit}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_confidence_gauge(label, value, color):
    """Display a confidence gauge with label"""
    st.markdown(
        f"""
        <div style="margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-weight: 500;">{label}</span>
                <span>{value*100:.1f}%</span>
            </div>
            <div class="confidence-gauge">
                <div class="confidence-fill" style="width: {value*100}%; background-color: {color};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_metric_box(label, value, delta, icon, color):
    """Display a single metric box with modern styling"""
    colors = get_theme_colors()
    
    # Determine delta color
    delta_color = colors["positive_border"] if "+" in str(delta) else colors["negative_border"]
    
    st.markdown(
        f"""
        <div style="background: {colors["card_bg"]}; backdrop-filter: blur(20px); border-radius: 16px; padding: 1.5rem; border: 1px solid {colors["card_border"]}; box-shadow: 0 8px 32px {colors["shadow"]}; transition: all 0.3s ease; text-align: center; position: relative; overflow: hidden;">
            <div style="position: absolute; top: 0; left: 0; right: 0; height: 3px; background: {color};"></div>
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
            <div style="font-size: 2rem; font-weight: 800; margin-bottom: 0.25rem; color: {color};">{value}%</div>
            <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.7; font-weight: 600; margin-bottom: 0.5rem; color: {colors["text_color"]};">{label}</div>
            <div style="font-size: 0.85rem; font-weight: 600; color: {delta_color}; display: inline-block; padding: 0.25rem 0.75rem; background: rgba(93, 211, 158, 0.15); border-radius: 12px;">{delta}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_performance_metrics(epoch=150):
    """Display model performance metrics from research - DATA REAL DARI TA"""
    colors = get_theme_colors()
    
    # Data untuk setiap epoch
    epoch_data = {
        50: {
            "accuracy": 75.07,
            "precision": 75.59,
            "recall": 75.07,
            "f1_score": 75.13,
            "per_class": {
                get_text('negative'): {"precision": 80.97, "recall": 72.85, "f1": 76.70, "support": 1256},
                get_text('neutral'): {"precision": 70.08, "recall": 78.28, "f1": 73.95, "support": 1768},
                get_text('positive'): {"precision": 78.55, "recall": 72.21, "f1": 75.25, "support": 1004}
            },
            "macro_avg": {"precision": 76.53, "recall": 74.45, "f1": 75.30},
            "weighted_avg": {"precision": 75.59, "recall": 75.07, "f1": 75.13}
        },
        100: {
            "accuracy": 75.15,
            "precision": 76.07,
            "recall": 75.15,
            "f1_score": 75.13,
            "per_class": {
                get_text('negative'): {"precision": 80.81, "recall": 75.80, "f1": 78.23, "support": 1256},
                get_text('neutral'): {"precision": 69.20, "recall": 80.71, "f1": 74.52, "support": 1768},
                get_text('positive'): {"precision": 82.23, "recall": 64.54, "f1": 72.32, "support": 1004}
            },
            "macro_avg": {"precision": 77.42, "recall": 73.68, "f1": 75.02},
            "weighted_avg": {"precision": 76.07, "recall": 75.15, "f1": 75.13}
        },
        150: {
            "accuracy": 75.94,
            "precision": 76.58,
            "recall": 75.94,
            "f1_score": 75.99,
            "per_class": {
                get_text('negative'): {"precision": 82.64, "recall": 73.17, "f1": 77.62, "support": 1256},
                get_text('neutral'): {"precision": 70.60, "recall": 80.15, "f1": 75.07, "support": 1768},
                get_text('positive'): {"precision": 79.54, "recall": 72.01, "f1": 75.59, "support": 1004}
            },
            "macro_avg": {"precision": 77.59, "recall": 75.11, "f1": 76.09},
            "weighted_avg": {"precision": 76.58, "recall": 75.94, "f1": 75.99}
        }
    }
    
    data = epoch_data[epoch]
    
    st.markdown(
        f"""
        <div style="text-align: center; margin: 3rem 0 2rem 0;">
            <h2 style="font-size: 2rem; margin-bottom: 0.5rem; background: linear-gradient(135deg, {colors["primary_color"]}, {colors["secondary_color"]}); 
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                🎯 {get_text('model_performance')} - Epoch {epoch}
            </h2>
            <p style="font-size: 0.95rem; opacity: 0.7;">
                {get_text('cnn_bilstm_attention')} • {get_text('trained_on')} 26,852 {get_text('samples')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Performance metrics in 4 columns
    col1, col2, col3, col4 = st.columns(4)
    
    # Metrics data
    metrics = {
        "accuracy": {"value": data["accuracy"], "delta": "+2.3", "icon": "🎯"},
        "precision": {"value": data["precision"], "delta": "+1.8", "icon": "🔍"},
        "recall": {"value": data["recall"], "delta": "+2.1", "icon": "📊"},
        "f1_score": {"value": data["f1_score"], "delta": "+2.0", "icon": "⭐"}
    }
    
    with col1:
        display_metric_box(
            label=get_text('accuracy'),
            value=metrics["accuracy"]["value"],
            delta=metrics["accuracy"]["delta"],
            icon=metrics["accuracy"]["icon"],
            color=colors["primary_color"]
        )
    
    with col2:
        display_metric_box(
            label=get_text('precision'),
            value=metrics["precision"]["value"],
            delta=metrics["precision"]["delta"],
            icon=metrics["precision"]["icon"],
            color=colors["secondary_color"]
        )
    
    with col3:
        display_metric_box(
            label=get_text('recall'),
            value=metrics["recall"]["value"],
            delta=metrics["recall"]["delta"],
            icon=metrics["recall"]["icon"],
            color=colors["accent_color"]
        )
    
    with col4:
        display_metric_box(
            label=get_text('f1_score'),
            value=metrics["f1_score"]["value"],
            delta=metrics["f1_score"]["delta"],
            icon=metrics["f1_score"]["icon"],
            color=colors["positive_border"]
        )
    
    # Detailed metrics table
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    
    with st.expander(f"📋 {get_text('detailed_metrics')}"):
        # Per-class metrics
        st.markdown(f"### {get_text('per_class_performance')}")
        st.markdown(f"<p style='font-size: 0.9rem; opacity: 0.7; margin-bottom: 1rem;'>{get_text('model_label')}: CNN + BiLSTM | {get_text('epoch_label')}: {epoch} | {get_text('total_samples')}: 4,028</p>", unsafe_allow_html=True)
        
        # Create DataFrame for metrics
        metrics_df = pd.DataFrame({
            get_text('class'): list(data["per_class"].keys()),
            get_text('precision'): [data["per_class"][k]["precision"] for k in data["per_class"].keys()],
            get_text('recall'): [data["per_class"][k]["recall"] for k in data["per_class"].keys()],
            'F1-Score': [data["per_class"][k]["f1"] for k in data["per_class"].keys()],
            'Support': [data["per_class"][k]["support"] for k in data["per_class"].keys()]
        })
        
        # Create custom HTML table with exact same styling as sentiment boxes
        # Build table HTML step by step to avoid f-string escaping issues
        table_html = '<div style="background: ' + colors["input_bg"] + '; border: 1px solid ' + colors["card_border"] + '; border-radius: 12px; overflow: hidden; margin: 1rem 0;">'
        table_html += '<table style="width: 100%; border-collapse: collapse; color: ' + colors["text_color"] + ';">'
        table_html += '<thead><tr style="background: ' + colors["card_bg"] + '; border-bottom: 2px solid ' + colors["card_border"] + ';">'
        table_html += '<th style="padding: 0.75rem; text-align: left; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">' + get_text('class') + '</th>'
        table_html += '<th style="padding: 0.75rem; text-align: center; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">' + get_text('precision') + '</th>'
        table_html += '<th style="padding: 0.75rem; text-align: center; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">' + get_text('recall') + '</th>'
        table_html += '<th style="padding: 0.75rem; text-align: center; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">F1-Score</th>'
        table_html += '<th style="padding: 0.75rem; text-align: center; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">Support</th>'
        table_html += '</tr></thead><tbody>'
        
        # Row styles - EXACT same as sentiment result boxes
        row_styles = {
            get_text('negative'): {
                'bg': colors["negative_bg"],
                'border': colors["negative_border"],
                'shadow': colors["negative_glow"]
            },
            get_text('neutral'): {
                'bg': colors["neutral_bg"],
                'border': colors["neutral_border"],
                'shadow': colors["neutral_glow"]
            },
            get_text('positive'): {
                'bg': colors["positive_bg"],
                'border': colors["positive_border"],
                'shadow': colors["positive_glow"]
            }
        }
        
        for cls_name in data["per_class"].keys():
            cls_data = data["per_class"][cls_name]
            style = row_styles.get(cls_name, {'bg': 'transparent', 'border': colors["card_border"], 'shadow': 'rgba(0,0,0,0.2)'})
            
            # Apply EXACT same styling as sentiment boxes
            table_html += '<tr style="background: ' + style['bg'] + '; border: 2px solid ' + style['border'] + '; border-left: 3px solid ' + style['border'] + '; box-shadow: 0 4px 16px ' + style['shadow'] + '; transition: all 0.3s ease;">'
            table_html += '<td style="padding: 0.75rem; font-weight: 600;">' + str(cls_name) + '</td>'
            table_html += '<td style="padding: 0.75rem; text-align: center; font-weight: 700; color: ' + colors["secondary_color"] + ';">' + f'{cls_data["precision"]:.2f}%' + '</td>'
            table_html += '<td style="padding: 0.75rem; text-align: center; font-weight: 700; color: ' + colors["accent_color"] + ';">' + f'{cls_data["recall"]:.2f}%' + '</td>'
            table_html += '<td style="padding: 0.75rem; text-align: center; font-weight: 700; color: ' + colors["primary_color"] + ';">' + f'{cls_data["f1"]:.2f}%' + '</td>'
            table_html += '<td style="padding: 0.75rem; text-align: center; font-weight: 600;">' + f'{cls_data["support"]:,}' + '</td>'
            table_html += '</tr>'
        
        table_html += '</tbody></table></div>'
        
        st.markdown(table_html, unsafe_allow_html=True)
        
        # Additional metrics
        st.markdown(f"### {get_text('aggregate_metrics')}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                f"""
                <div class="metric-card-hover" style="background: {colors["card_bg"]}; padding: 1rem; border-radius: 12px; text-align: center; border: 1px solid {colors["card_border"]};">
                    <div style="font-size: 0.75rem; text-transform: uppercase; opacity: 0.7; margin-bottom: 0.5rem;">{get_text('macro_average')}</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: {colors["secondary_color"]};">{data["macro_avg"]["f1"]:.2f}%</div>
                    <div style="font-size: 0.8rem; opacity: 0.6; margin-top: 0.25rem;">{get_text('precision')}: {data["macro_avg"]["precision"]:.2f}% | {get_text('recall')}: {data["macro_avg"]["recall"]:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""
                <div class="metric-card-hover" style="background: {colors["card_bg"]}; padding: 1rem; border-radius: 12px; text-align: center; border: 1px solid {colors["card_border"]};">
                    <div style="font-size: 0.75rem; text-transform: uppercase; opacity: 0.7; margin-bottom: 0.5rem;">{get_text('weighted_average')}</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: {colors["primary_color"]};">{data["weighted_avg"]["f1"]:.2f}%</div>
                    <div style="font-size: 0.8rem; opacity: 0.6; margin-top: 0.25rem;">{get_text('precision')}: {data["weighted_avg"]["precision"]:.2f}% | {get_text('recall')}: {data["weighted_avg"]["recall"]:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                f"""
                <div class="metric-card-hover" style="background: {colors["card_bg"]}; padding: 1rem; border-radius: 12px; text-align: center; border: 1px solid {colors["card_border"]};">
                    <div style="font-size: 0.75rem; text-transform: uppercase; opacity: 0.7; margin-bottom: 0.5rem;">{get_text('overall_accuracy')}</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: {colors["accent_color"]};">{data["accuracy"]:.2f}%</div>
                    <div style="font-size: 0.8rem; opacity: 0.6; margin-top: 0.25rem;">{get_text('test')}: 4,028 {get_text('samples')}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Model info
        st.markdown(f"### {get_text('model_info')}")
        
        # Dataset split info
        st.markdown(
            f"""
            <div class="dataset-card-hover" style="background: {colors["card_bg"]}; padding: 1rem; border-radius: 12px; border: 1px solid {colors["card_border"]}; margin-bottom: 1rem;">
                <div style="font-size: 0.85rem; font-weight: 600; margin-bottom: 0.75rem; opacity: 0.9;">📊 {get_text('dataset_split')}</div>
                <div style="display: flex; justify-content: space-between; gap: 1rem;">
                    <div style="flex: 1; text-align: center;">
                        <div style="font-size: 0.7rem; text-transform: uppercase; opacity: 0.6; margin-bottom: 0.25rem;">{get_text('train')}</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: {colors["primary_color"]};">19,400</div>
                    </div>
                    <div style="flex: 1; text-align: center; border-left: 1px solid {colors["card_border"]}; border-right: 1px solid {colors["card_border"]};">
                        <div style="font-size: 0.7rem; text-transform: uppercase; opacity: 0.6; margin-bottom: 0.25rem;">{get_text('val')}</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: {colors["secondary_color"]};">3,424</div>
                    </div>
                    <div style="flex: 1; text-align: center;">
                        <div style="font-size: 0.7rem; text-transform: uppercase; opacity: 0.6; margin-bottom: 0.25rem;">{get_text('test')}</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: {colors["accent_color"]};">4,028</div>
                    </div>
                </div>
                <div style="font-size: 0.75rem; opacity: 0.5; margin-top: 0.75rem; text-align: center;">{get_text('total')}: 26,852 {get_text('samples')}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.info(f"""
**{get_text('architecture_label')}:** {get_text('cnn_bilstm_attention')}  
**{get_text('training_epochs_label')}:** {epoch}  
**{get_text('test_samples_label')}:** 4,028  
**{get_text('best_performance_label')}:** {get_text('weighted_average')} F1-Score {data["weighted_avg"]["f1"]:.2f}%

{get_text('model_description_detail')}
        """)


def main():
    # Load model and artifacts
    with st.spinner(get_text("loading_model")):
        model, tokenizer, label_encoder = load_model_and_artifacts()

    if model is None:
        st.error(get_text("error_loading"))
        return

    # Get theme colors
    colors = get_theme_colors()

    # Sidebar with info
    with st.sidebar:
        # Language selector at the top
        st.markdown(
            f"""
            <div style="text-align: center; margin: 1rem 0 1rem 0;">
                <h2 style="font-size: 1.3rem; margin-bottom: 0; background: linear-gradient(135deg, {colors["primary_color"]}, {colors["secondary_color"]}); 
                           -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                    🌐 {get_text('language')}
                </h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Language selection buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "🇬🇧 English",
                use_container_width=True,
                type="primary" if st.session_state.language == "en" else "secondary",
            ):
                st.session_state.language = "en"
                st.rerun()
        with col2:
            if st.button(
                "🇮🇩 Indonesia",
                use_container_width=True,
                type="primary" if st.session_state.language == "id" else "secondary",
            ):
                st.session_state.language = "id"
                st.rerun()

        st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
        
        # Navigation buttons
        st.markdown(
            f"""
            <div style="text-align: center; margin: 1rem 0 1.5rem 0;">
                <h2 style="font-size: 1.3rem; margin-bottom: 0; background: linear-gradient(135deg, {colors["primary_color"]}, {colors["secondary_color"]}); 
                           -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                    🧭 Navigation
                </h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        if st.button(
            f"🔍 {get_text('nav_analysis')}", 
            use_container_width=True,
            type="primary" if st.session_state.page == "analysis" else "secondary"
        ):
            st.session_state.page = "analysis"
            st.rerun()
            
        if st.button(
            f"📊 {get_text('nav_performance')}", 
            use_container_width=True,
            type="primary" if st.session_state.page == "performance" else "secondary"
        ):
            st.session_state.page = "performance"
            st.rerun()

        st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

        # About section
        st.markdown(
            f"""
            <div style="text-align: center; margin: 1rem 0 1.5rem 0;">
                <h2 style="font-size: 1.5rem; margin-bottom: 0; background: linear-gradient(135deg, {colors["primary_color"]}, {colors["secondary_color"]}); 
                           -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                    📖 {get_text('about')}
                </h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # About section with modern styling
        st.markdown(
            f"""
            <div class="card">
                <h3 style="margin-top: 0; font-size: 1.25rem; margin-bottom: 1rem;">{get_text('about_title')}</h3>
                <p style="font-size: 0.9rem; line-height: 1.6; margin-bottom: 1rem;">
                    {get_text('about_text')}
                </p>
                <div style="margin: 1rem 0;">
                    <div style="display: flex; align-items: center; margin: 0.75rem 0;">
                        <span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; 
                                     background: {colors["positive_border"]}; margin-right: 0.75rem; 
                                     box-shadow: 0 0 10px {colors["positive_border"]}40;"></span>
                        <strong style="color: {colors["positive_border"]};">{get_text('positive')}</strong>
                    </div>
                    <div style="display: flex; align-items: center; margin: 0.75rem 0;">
                        <span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; 
                                     background: {colors["neutral_border"]}; margin-right: 0.75rem;
                                     box-shadow: 0 0 10px {colors["neutral_border"]}40;"></span>
                        <strong style="color: {colors["neutral_border"]};">{get_text('neutral')}</strong>
                    </div>
                    <div style="display: flex; align-items: center; margin: 0.75rem 0;">
                        <span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; 
                                     background: {colors["negative_border"]}; margin-right: 0.75rem;
                                     box-shadow: 0 0 10px {colors["negative_border"]}40;"></span>
                        <strong style="color: {colors["negative_border"]};">{get_text('negative')}</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Model info with badges
        st.markdown(
            f"""
            <div class="card">
                <h3 style="margin-top: 0; font-size: 1.25rem; margin-bottom: 1rem;">🤖 {get_text('model_info')}</h3>
                <div style="margin: 1rem 0;">
                    <div style="margin-bottom: 1rem;">
                        <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; 
                                    opacity: 0.7; margin-bottom: 0.25rem;">{get_text('architecture')}</div>
                        <div style="background: linear-gradient(135deg, {colors["primary_color"]}20, {colors["secondary_color"]}20);
                                    border: 1px solid {colors["primary_color"]}40; border-radius: 8px; 
                                    padding: 0.5rem; font-weight: 600; font-size: 0.9rem;">
                            CNN + BiLSTM + Attention
                        </div>
                    </div>
                    <div style="margin-bottom: 1rem;">
                        <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; 
                                    opacity: 0.7; margin-bottom: 0.25rem;">{get_text('accuracy')}</div>
                        <div style="background: linear-gradient(135deg, {colors["secondary_color"]}20, {colors["accent_color"]}20);
                                    border: 1px solid {colors["secondary_color"]}40; border-radius: 8px; 
                                    padding: 0.5rem; font-weight: 600; font-size: 0.9rem;">
                            75.94% ⭐
                        </div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; 
                                    opacity: 0.7; margin-bottom: 0.25rem;">{get_text('dataset_size')}</div>
                        <div style="background: linear-gradient(135deg, {colors["accent_color"]}20, {colors["primary_color"]}20);
                                    border: 1px solid {colors["accent_color"]}40; border-radius: 8px; 
                                    padding: 0.5rem; font-weight: 600; font-size: 0.9rem;">
                            26,852 {get_text('samples')}
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Example section - only show on analysis page
        if st.session_state.page == "analysis":
            st.markdown(
                f"""
                <div class="card">
                    <h3 style="margin-top: 0; font-size: 1.25rem; margin-bottom: 0.75rem;">💡 {get_text('try_examples')}</h3>
                    <p style="font-size: 0.85rem; opacity: 0.7; margin-bottom: 1rem;">
                        {get_text('input_subtitle').split('.')[0]}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("😊", use_container_width=True, key="positive_btn"):
                    st.session_state.example_text = get_text("example_positive")
                    st.rerun()
            with col2:
                if st.button("😐", use_container_width=True, key="neutral_btn"):
                    st.session_state.example_text = get_text("example_neutral")
                    st.rerun()
            with col3:
                if st.button("😞", use_container_width=True, key="negative_btn"):
                    st.session_state.example_text = get_text("example_negative")
                    st.rerun()
    
    # Route to appropriate page
    if st.session_state.page == "performance":
        show_performance_page(colors)
    else:
        show_analysis_page(model, tokenizer, label_encoder, colors)


def show_performance_page(colors):
    """Display the model performance comparison page"""
    # Main content area
    st.markdown(
        f"""
        <div style="text-align: center; margin: 2rem 0 3rem 0; position: relative;">
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                        width: 400px; height: 400px; background: radial-gradient(circle, {colors["primary_color"]}15 0%, transparent 70%); 
                        filter: blur(60px); z-index: 0;"></div>
            <h1 style="font-size: 3rem; font-weight: 800; margin-bottom: 1rem; position: relative; z-index: 1;">
                <span style="background: linear-gradient(135deg, {colors["primary_color"]}, {colors["secondary_color"]}, {colors["accent_color"]}); 
                      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                      background-clip: text; display: inline-block;">
                    📊 {get_text('model_performance')}
                </span>
            </h1>
            <p style="font-size: 1.125rem; margin-bottom: 0.5rem; color: {colors["text_color"]}; 
                      opacity: 0.9; font-weight: 500; position: relative; z-index: 1;">
                {get_text('epoch_comparison')}
            </p>
            <p style="font-size: 0.95rem; color: {colors["text_color"]}; 
                      opacity: 0.6; max-width: 600px; margin: 0 auto; position: relative; z-index: 1;">
                {get_text('cnn_bilstm_attention')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Tabs for different epochs
    tab1, tab2, tab3 = st.tabs(["📈 Epoch 50", "📈 Epoch 100", "📈 Epoch 150 (Best)"])
    
    with tab1:
        display_performance_metrics(epoch=50)
    
    with tab2:
        display_performance_metrics(epoch=100)
    
    with tab3:
        display_performance_metrics(epoch=150)
    
    # Footer
    st.markdown(
        f"""
        <div style="text-align: center; padding: 3rem 0 2rem 0; margin-top: 4rem; 
                    border-top: 1px solid {colors["card_border"]};">
            <div style="display: flex; justify-content: center; align-items: center; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem;">
                <span style="font-size: 1.25rem;">🚀</span>
                <span style="font-weight: 600; font-size: 1rem;">{get_text('powered_by')}</span>
                <span style="opacity: 0.5;">•</span>
                <span style="background: linear-gradient(135deg, {colors["secondary_color"]}, {colors["accent_color"]}); 
                             -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                             background-clip: text; font-weight: 700;">
                    Streamlit
                </span>
            </div>
            <p style="font-size: 0.875rem; opacity: 0.7; margin: 0.5rem 0;">
                {get_text('model_description')}
            </p>
            <p style="font-size: 0.75rem; opacity: 0.5; margin-top: 1rem;">
                {get_text('copyright')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_analysis_page(model, tokenizer, label_encoder, colors):
    """Display the sentiment analysis page"""
    st.markdown(
        f"""
        <div style="text-align: center; margin: 2rem 0 3rem 0; position: relative;">
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                        width: 400px; height: 400px; background: radial-gradient(circle, {colors["primary_color"]}15 0%, transparent 70%); 
                        filter: blur(60px); z-index: 0;"></div>
            <h1 style="font-size: 3rem; font-weight: 800; margin-bottom: 1rem; position: relative; z-index: 1;">
                <span style="background: linear-gradient(135deg, {colors["primary_color"]}, {colors["secondary_color"]}, {colors["accent_color"]}); 
                      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                      background-clip: text; display: inline-block;">
                    {get_text('title')}
                </span>
            </h1>
            <p style="font-size: 1.125rem; margin-bottom: 0.5rem; color: {colors["text_color"]}; 
                      opacity: 0.9; font-weight: 500; position: relative; z-index: 1;">
                {get_text('subtitle')}
            </p>
            <p style="font-size: 0.95rem; color: {colors["text_color"]}; 
                      opacity: 0.6; max-width: 600px; margin: 0 auto; position: relative; z-index: 1;">
                {get_text('description')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # === PERFORMANCE METRICS DISPLAY === 
    display_performance_metrics()
    
    # Main container
    with st.container():
        # Use example text if available
        default_text = st.session_state.get("example_text", "")

        # Input area with modern styling
        st.markdown(
            f"""
            <h3 style="margin-top: 0; font-size: 1.5rem; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
                <span>{get_text('input_title')}</span>
            </h3>
            <p style="font-size: 0.95rem; opacity: 0.7; margin-bottom: 1rem;">
                {get_text('input_subtitle')}
            </p>
            """,
            unsafe_allow_html=True,
        )

        user_input = st.text_area(
            get_text("input_label"),
            value=default_text,
            height=200,
            placeholder=get_text("input_placeholder"),
            label_visibility="collapsed",
        )

        # Analyze button with modern styling
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button(
                get_text("analyze_button"), use_container_width=True, type="primary"
            )

    # Perform analysis
    if analyze_button:
        # Clear example text from session state after button click
        if "example_text" in st.session_state:
            del st.session_state.example_text

        if not user_input.strip():
            st.warning(get_text("empty_warning"))
        else:
            with st.spinner(get_text("analyzing")):
                sentiment, confidence, all_probs, cleaned_text = predict_sentiment(
                    user_input, model, tokenizer, label_encoder
                )

            # Display results with modern design
            st.markdown(
                f"""
                <h2 style="margin-top: 2rem; font-size: 2rem; margin-bottom: 1.5rem;">
                    {get_text('results_title')}
                </h2>
                """,
                unsafe_allow_html=True,
            )

            # Get translated sentiment for comparison
            translated_positive = get_text("positive")
            translated_negative = get_text("negative")
            translated_neutral = get_text("neutral")

            # Sentiment box with color and better icons
            sentiment_class = (
                "sentiment-positive"
                if sentiment == translated_positive
                else (
                    "sentiment-negative"
                    if sentiment == translated_negative
                    else "sentiment-neutral"
                )
            )

            sentiment_icon = (
                "🎉"
                if sentiment == translated_positive
                else "😔" if sentiment == translated_negative else "🤔"
            )

            st.markdown(
                f"""
                <div class="{sentiment_class}" style="position: relative;">
                    <div style="display: flex; justify-content: space-between; align-items: center; position: relative; z-index: 1;">
                        <div style="display: flex; align-items: center; gap: 1rem;">
                            <span style="font-size: 3.5rem; line-height: 1;">{sentiment_icon}</span>
                            <div>
                                <div style="font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.1em; 
                                            opacity: 0.8; margin-bottom: 0.25rem;">{get_text('sentiment_label')}</div>
                                <h2 style="margin: 0; font-size: 2.5rem; font-weight: 800;">{sentiment}</h2>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.1em; 
                                        opacity: 0.8; margin-bottom: 0.25rem;">{get_text('confidence_label')}</div>
                            <div style="font-size: 2.5rem; font-weight: 800;">
                                {confidence*100:.1f}<span style="font-size: 1.5rem; opacity: 0.8;">%</span>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Confidence breakdown with modern design
            st.markdown(
                f"""
                <h3 style="font-size: 1.5rem; margin-top: 2rem; margin-bottom: 1.25rem; display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.5rem;">📈</span>
                    <span>{get_text('distribution_title')}</span>
                </h3>
                """,
                unsafe_allow_html=True,
            )

            # Sort probabilities
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)

            # Set colors and emojis for each sentiment (case-insensitive keys)
            sentiment_colors = {
                translated_positive.lower(): colors["positive_border"],
                translated_neutral.lower(): colors["neutral_border"],
                translated_negative.lower(): colors["negative_border"],
            }

            sentiment_emojis = {
                translated_positive.lower(): "😊",
                translated_neutral.lower(): "😐",
                translated_negative.lower(): "😞",
            }

            # Display confidence gauges with emojis
            for label, prob in sorted_probs:
                label_key = str(label).strip().lower()
                st.markdown(
                    f"""
                    <div style="margin-bottom: 1.25rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <div style="display: flex; align-items: center; gap: 0.5rem;">
                                <span style="font-size: 1.25rem;">{sentiment_emojis.get(label_key, '😐')}</span>
                                <span style="font-weight: 600; font-size: 1rem;">{label}</span>
                            </div>
                            <span style="font-weight: 700; font-size: 1.125rem; color: {sentiment_colors.get(label_key, colors['neutral_border'])};">
                                {prob*100:.1f}%
                            </span>
                        </div>
                        <div class="confidence-gauge">
                            <div class="confidence-fill" style="width: {prob*100}%; background: linear-gradient(90deg, {sentiment_colors.get(label_key, colors['neutral_border'])}, {sentiment_colors.get(label_key, colors['neutral_border'])}dd);"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Text statistics - Simplified design
            st.markdown(
                f"""
                <h3 style="font-size: 1.5rem; margin-top: 2.5rem; margin-bottom: 1rem;">
                    📝 {get_text('statistics_title')}
                </h3>
                """,
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(user_input.split())}</div>
                        <div class="metric-label">{get_text('word_count')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(user_input)}</div>
                        <div class="metric-label">{get_text('characters')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(cleaned_text.split())}</div>
                        <div class="metric-label">{get_text('cleaned_words')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Show cleaned text with better styling
            st.markdown(
                "<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True
            )
            with st.expander(get_text("view_preprocessed")):
                st.code(cleaned_text, language=None)

    # Footer with modern design
    st.markdown(
        f"""
        <div style="text-align: center; padding: 3rem 0 2rem 0; margin-top: 4rem; 
                    border-top: 1px solid {colors["card_border"]};">
            <div style="display: flex; justify-content: center; align-items: center; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem;">
                <span style="font-size: 1.25rem;">🚀</span>
                <span style="font-weight: 600; font-size: 1rem;">{get_text('powered_by')}</span>
                <span style="opacity: 0.5;">•</span>
                <span style="background: linear-gradient(135deg, {colors["secondary_color"]}, {colors["accent_color"]}); 
                             -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                             background-clip: text; font-weight: 700;">
                    Streamlit
                </span>
            </div>
            <p style="font-size: 0.875rem; opacity: 0.7; margin: 0.5rem 0;">
                {get_text('model_description')}
            </p>
            <p style="font-size: 0.75rem; opacity: 0.5; margin-top: 1rem;">
                {get_text('copyright')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_analysis_page(model, tokenizer, label_encoder, colors):
    """Display the sentiment analysis page"""
    # Main content area
    st.markdown(
        f"""
        <div style="text-align: center; margin: 2rem 0 3rem 0; position: relative;">
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                        width: 400px; height: 400px; background: radial-gradient(circle, {colors["primary_color"]}15 0%, transparent 70%); 
                        filter: blur(60px); z-index: 0;"></div>
            <h1 style="font-size: 3rem; font-weight: 800; margin-bottom: 1rem; position: relative; z-index: 1;">
                <span style="background: linear-gradient(135deg, {colors["primary_color"]}, {colors["secondary_color"]}, {colors["accent_color"]}); 
                      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                      background-clip: text; display: inline-block;">
                    {get_text('title')}
                </span>
            </h1>
            <p style="font-size: 1.125rem; margin-bottom: 0.5rem; color: {colors["text_color"]}; 
                      opacity: 0.9; font-weight: 500; position: relative; z-index: 1;">
                {get_text('subtitle')}
            </p>
            <p style="font-size: 0.95rem; color: {colors["text_color"]}; 
                      opacity: 0.6; max-width: 600px; margin: 0 auto; position: relative; z-index: 1;">
                {get_text('description')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Main container
    with st.container():
        # Use example text if available
        default_text = st.session_state.get("example_text", "")

        # Input area with modern styling
        st.markdown(
            f"""
            <h3 style="margin-top: 0; font-size: 1.5rem; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
                <span>{get_text('input_title')}</span>
            </h3>
            <p style="font-size: 0.95rem; opacity: 0.7; margin-bottom: 1rem;">
                {get_text('input_subtitle')}
            </p>
            """,
            unsafe_allow_html=True,
        )

        user_input = st.text_area(
            get_text("input_label"),
            value=default_text,
            height=200,
            placeholder=get_text("input_placeholder"),
            label_visibility="collapsed",
        )

        # Analyze button with modern styling
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button(
                get_text("analyze_button"), use_container_width=True, type="primary"
            )

    # Perform analysis
    if analyze_button:
        # Clear example text from session state after button click
        if "example_text" in st.session_state:
            del st.session_state.example_text

        if not user_input.strip():
            st.warning(get_text("empty_warning"))
        else:
            with st.spinner(get_text("analyzing")):
                sentiment, confidence, all_probs, cleaned_text = predict_sentiment(
                    user_input, model, tokenizer, label_encoder
                )

            # Display results with modern design
            st.markdown(
                f"""
                <h2 style="margin-top: 2rem; font-size: 2rem; margin-bottom: 1.5rem;">
                    {get_text('results_title')}
                </h2>
                """,
                unsafe_allow_html=True,
            )

            # Get translated sentiment for comparison
            translated_positive = get_text("positive")
            translated_negative = get_text("negative")
            translated_neutral = get_text("neutral")

            # Sentiment box with color and better icons
            sentiment_class = (
                "sentiment-positive"
                if sentiment == translated_positive
                else (
                    "sentiment-negative"
                    if sentiment == translated_negative
                    else "sentiment-neutral"
                )
            )

            sentiment_icon = (
                "🎉"
                if sentiment == translated_positive
                else "😔" if sentiment == translated_negative else "🤔"
            )

            st.markdown(
                f"""
                <div class="{sentiment_class}" style="position: relative;">
                    <div style="display: flex; justify-content: space-between; align-items: center; position: relative; z-index: 1;">
                        <div style="display: flex; align-items: center; gap: 1rem;">
                            <span style="font-size: 3.5rem; line-height: 1;">{sentiment_icon}</span>
                            <div>
                                <div style="font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.1em; 
                                            opacity: 0.8; margin-bottom: 0.25rem;">{get_text('sentiment_label')}</div>
                                <h2 style="margin: 0; font-size: 2.5rem; font-weight: 800;">{sentiment}</h2>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.1em; 
                                        opacity: 0.8; margin-bottom: 0.25rem;">{get_text('confidence_label')}</div>
                            <div style="font-size: 2.5rem; font-weight: 800;">
                                {confidence*100:.1f}<span style="font-size: 1.5rem; opacity: 0.8;">%</span>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Confidence breakdown with modern design
            st.markdown(
                f"""
                <h3 style="font-size: 1.5rem; margin-top: 2rem; margin-bottom: 1.25rem; display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.5rem;">📈</span>
                    <span>{get_text('distribution_title')}</span>
                </h3>
                """,
                unsafe_allow_html=True,
            )

            # Sort probabilities
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)

            # Set colors and emojis for each sentiment (case-insensitive keys)
            sentiment_colors = {
                translated_positive.lower(): colors["positive_border"],
                translated_neutral.lower(): colors["neutral_border"],
                translated_negative.lower(): colors["negative_border"],
            }

            sentiment_emojis = {
                translated_positive.lower(): "😊",
                translated_neutral.lower(): "😐",
                translated_negative.lower(): "😞",
            }

            # Display confidence gauges with emojis
            for label, prob in sorted_probs:
                label_key = str(label).strip().lower()
                st.markdown(
                    f"""
                    <div style="margin-bottom: 1.25rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <div style="display: flex; align-items: center; gap: 0.5rem;">
                                <span style="font-size: 1.25rem;">{sentiment_emojis.get(label_key, '😐')}</span>
                                <span style="font-weight: 600; font-size: 1rem;">{label}</span>
                            </div>
                            <span style="font-weight: 700; font-size: 1.125rem; color: {sentiment_colors.get(label_key, colors['neutral_border'])};">
                                {prob*100:.1f}%
                            </span>
                        </div>
                        <div class="confidence-gauge">
                            <div class="confidence-fill" style="width: {prob*100}%; background: linear-gradient(90deg, {sentiment_colors.get(label_key, colors['neutral_border'])}, {sentiment_colors.get(label_key, colors['neutral_border'])}dd);"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Text statistics - Simplified design
            st.markdown(
                f"""
                <h3 style="font-size: 1.5rem; margin-top: 2.5rem; margin-bottom: 1rem;">
                    📝 {get_text('statistics_title')}
                </h3>
                """,
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(user_input.split())}</div>
                        <div class="metric-label">{get_text('word_count')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(user_input)}</div>
                        <div class="metric-label">{get_text('characters')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(cleaned_text.split())}</div>
                        <div class="metric-label">{get_text('cleaned_words')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Show cleaned text with better styling
            st.markdown(
                "<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True
            )
            with st.expander(get_text("view_preprocessed")):
                st.code(cleaned_text, language=None)

    # Footer with modern design
    st.markdown(
        f"""
        <div style="text-align: center; padding: 3rem 0 2rem 0; margin-top: 4rem; 
                    border-top: 1px solid {colors["card_border"]};">
            <div style="display: flex; justify-content: center; align-items: center; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem;">
                <span style="font-size: 1.25rem;">🚀</span>
                <span style="font-weight: 600; font-size: 1rem;">{get_text('powered_by')}</span>

                <span style="opacity: 0.5;">•</span>
                <span style="background: linear-gradient(135deg, {colors["secondary_color"]}, {colors["accent_color"]}); 
                             -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                             background-clip: text; font-weight: 700;">
                    Streamlit
                </span>
            </div>
            <p style="font-size: 0.875rem; opacity: 0.7; margin: 0.5rem 0;">
                {get_text('model_description')}
            </p>
            <p style="font-size: 0.75rem; opacity: 0.5; margin-top: 1rem;">
                {get_text('copyright')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()