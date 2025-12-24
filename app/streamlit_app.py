import streamlit as st
import torch
import numpy as np
import os
import sys
import io
import gdown

# Add parent directory to path for src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import BertTokenizer
from src.model_bert_attn import BertWithAttention
# from src.model_bilstm_attn import BiLSTMAttention   # 🔒 BiLSTM import commented
from src.explain import plot_attention, highlight_text, lime_explain
from src.data_utils import encode_texts
from PIL import Image

# Load logo if available
logo_path = os.path.join(os.path.dirname(__file__), 'logo.png')
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image(logo, width=80)
    with col2:
        st.title("Context-Aware Fake News Detection")
else:
    st.title("📰 Context-Aware Fake News Detection")

# Choose model
# model_type = st.selectbox("Choose Model", ("BERT+Attention", "BiLSTM+Attention"))  # 🔒 Removed BiLSTM option
#model_type = st.selectbox("Choose Model", ("BERT+Attention",))  # ✅ Only BERT for now
model_type = "BERT+Attention" 
news = st.text_area("Paste news article here:")

# ========== BERT+Attention ==========
if model_type == "BERT+Attention":
    st.write("🔍 BERT+Attention selected.")

    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertWithAttention()
        model_path = os.path.join(os.path.dirname(__file__), '../saved_models/best_model.pt')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        device = torch.device("cpu")  # or "cuda" if available
        model.to(device)

        st.success("✅ BERT model loaded.")
    except Exception as e:
        st.error(f"❌ Failed to load BERT model: {e}")
        st.stop()

    if st.button("Check", key="check_bert"):
        if news.strip():
            encoding = tokenizer(news, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            with torch.no_grad():
                logits, attn_weights = model(input_ids, attention_mask)
                prediction = torch.argmax(logits, dim=1).item()

            label = "📰 Prediction: REAL ✅" if prediction == 0 else "📰 Prediction: FAKE ❌"
            st.success(label)

            st.markdown("#### 🎯 Attention Heatmap")
            fig = plot_attention(news, attn_weights.squeeze(0))  # Squeeze batch dim
            st.pyplot(fig)

            # Save & download heatmap
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            st.download_button(
                "📥 Download Heatmap",
                buf.getvalue(),
                file_name="bert_attention_heatmap.png",
                mime="image/png"
            )

            st.markdown("#### ✨ Attention-Weighted Text")
            st.write(highlight_text(news, attn_weights.squeeze(0)))
        else:
            st.warning("⚠️ Please enter a news article.")

    if st.button("Explain with LIME", key="lime_bert"):
        if news.strip():
            try:
                explanation = lime_explain(news, model, tokenizer)
                st.markdown("#### 🧠 LIME Explanation")
                st.pyplot(explanation.as_pyplot_figure())
            except Exception as e:
                st.error(f"❌ LIME explanation failed: {e}")
        else:
            st.warning("⚠️ Please enter a news article.")

# ========== BiLSTM+Attention (DISABLED) ==========
# elif model_type == "BiLSTM+Attention":
#     st.write("🔍 BiLSTM+Attention selected.")
# 
#     vocab_path = os.path.join(os.path.dirname(__file__), '../saved_models/vocab.npy')
#     lstm_model_path = os.path.join(os.path.dirname(__file__), '../saved_models/bilstm_model.pt')
# 
#     try:
#         vocab = np.load(vocab_path, allow_pickle=True).item()
#         st.success("✅ Vocabulary loaded.")
#     except Exception as e:
#         st.error(f"❌ Error loading vocabulary: {e}")
#         st.stop()
# 
#     try:
#         model = BiLSTMAttention(vocab_size=len(vocab), embed_dim=128, hidden_dim=64, n_classes=2, pad_idx=0)
#         model.load_state_dict(torch.load(lstm_model_path, map_location=torch.device('cpu')))
#         model.eval()
#         st.success("✅ BiLSTM model loaded.")
#     except Exception as e:
#         st.error(f"❌ Error loading LSTM model: {e}")
#         st.stop()
# 
#     if st.button("Check", key="check_lstm"):
#         if news.strip():
#             x = encode_texts([news], vocab, 128)
#             x_tensor = torch.tensor(x, dtype=torch.long)
# 
#             with torch.no_grad():
#                 logits, attn_weights = model(x_tensor)
#                 prediction = torch.argmax(logits, dim=1).item()
# 
#             label = "📰 Prediction: REAL ✅" if prediction == 0 else "📰 Prediction: FAKE ❌"
#             st.success(label)
# 
#             st.markdown("#### 🎯 Attention Heatmap")
#             fig = plot_attention(news, attn_weights.squeeze(0))
#             st.pyplot(fig)
# 
#             buf = io.BytesIO()
#             fig.savefig(buf, format="png")
#             st.download_button(
#                 "📥 Download Heatmap",
#                 buf.getvalue(),
#                 file_name="lstm_attention_heatmap.png",
#                 mime="image/png"
#             )
# 
#             st.markdown("#### ✨ Attention-Weighted Text")
#             st.write(highlight_text(news, attn_weights.squeeze(0)))
#         else:
#             st.warning("⚠️ Please enter a news article.")
