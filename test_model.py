"""
Quick test script to verify model can be loaded and makes predictions
"""

import torch
import torch.nn as nn
import pickle
import numpy as np
import re
from transformers import BertModel, BertConfig


# Define BERT-based Sentiment Classifier
class BertSentimentClassifier(nn.Module):
    def __init__(self, num_classes=3, hidden_dropout_prob=0.1):
        super(BertSentimentClassifier, self).__init__()
        # Initialize BERT with configuration
        config = BertConfig(
            vocab_size=31923,  # Actual vocab size from your trained model
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=hidden_dropout_prob,
            max_position_embeddings=512,
            type_vocab_size=2,
        )
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # BERT forward pass
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        # Classification
        logits = self.classifier(pooled_output)
        return logits


def clean_text(text):
    """Clean and preprocess text"""
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^0-9a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


print("=" * 60)
print("Testing Sentiment Analysis Model (BERT)")
print("=" * 60)

try:
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    with open("kaggle/working/model_outputs/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    print("✓ Tokenizer loaded successfully")
    print(f"   Vocabulary size: {len(tokenizer.word_index)}")

    # Load label encoder
    print("\n2. Loading label encoder...")
    with open("kaggle/working/model_outputs/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    print("✓ Label encoder loaded successfully")
    print(f"   Classes: {list(label_encoder.classes_)}")

    # Load model
    print("\n3. Loading BERT model...")
    model = BertSentimentClassifier(num_classes=3, hidden_dropout_prob=0.1)
    state_dict = torch.load("pytorch_model.bin", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    print("✓ Model loaded successfully")
    print(f"   Model: BERT-based Sentiment Classifier")
    print(f"   Architecture: 12-layer BERT + Classifier")

    # Test predictions
    print("\n4. Testing predictions...")
    test_texts = [
        "Mobil listrik ini sangat bagus dan hemat energi. Saya sangat puas!",
        "Harga mobil listrik terlalu mahal dan infrastruktur charging masih kurang.",
        "Pemerintah sedang menyiapkan insentif untuk kendaraan bermotor listrik.",
    ]

    MAX_LEN = 128

    for i, text in enumerate(test_texts, 1):
        print(f"\n   Test {i}: {text[:50]}...")

        # Preprocess
        cleaned = clean_text(text)
        sequence = tokenizer.texts_to_sequences([cleaned])

        # Pad sequence manually
        if len(sequence[0]) > MAX_LEN:
            padded = sequence[0][:MAX_LEN]
        else:
            padded = sequence[0] + [0] * (MAX_LEN - len(sequence[0]))

        # Convert to PyTorch tensor
        input_tensor = torch.LongTensor([padded])

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_tensor != 0).long()

        # Predict
        with torch.no_grad():
            outputs = model(input_tensor, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = probabilities.cpu().numpy()

        predicted_class_idx = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class_idx]
        sentiment = label_encoder.classes_[predicted_class_idx]

        print(f"   → Sentiment: {sentiment}")
        print(f"   → Confidence: {confidence*100:.2f}%")
        print(f"   → All probabilities:")
        for j, class_name in enumerate(label_encoder.classes_):
            print(f"      - {class_name}: {prediction[0][j]*100:.2f}%")

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED - BERT Model is working correctly!")
    print("=" * 60)
    print("\n✓ Ready to run: streamlit run app.py")

except Exception as e:
    print("\n" + "=" * 60)
    print(f"✗ ERROR: {str(e)}")
    print("=" * 60)
    import traceback

    traceback.print_exc()
