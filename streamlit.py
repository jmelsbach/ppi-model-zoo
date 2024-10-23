import streamlit as st
import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer
import re

# Load models and tokenizers
MODEL_DICT = {
    "SYNTERACT": {
        "model": BertForSequenceClassification.from_pretrained('GleghornLab/SYNTERACT'),
        "tokenizer": BertTokenizer.from_pretrained('GleghornLab/SYNTERACT')
    }
}

# Set device
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

# Move model to device
for key, value in MODEL_DICT.items():
    value['model'].to(device)
    value['model'].eval()

# Streamlit interface
st.title('Protein-Protein Interaction Prediction')

# Sidebar for model selection
selected_model_name = st.sidebar.selectbox(
    'Select Model', list(MODEL_DICT.keys()))

# Text input for protein sequences
sequence_a = st.text_area("Input Sequence A (FASTA format)",
                          value='MEKSCSIGNGREQYGWGHGEQCGTQFLECVYRNASMYSVLGDLITYVVFLGATCYAILFGFRLLLSCVRIVLKVVIALFVIRLLLALGSVDITSVSYSG')
sequence_b = st.text_area("Input Sequence B (FASTA format)",
                          value='MRLTLLALIGVLCLACAYALDDSENNDQVVGLLDVADQGANHANDGAREARQLGGWGGGWGGRGGWGGRGGWGGRGGWGGRGGWGGRGGWGGGWGGRGGWGGRGGGWYGR')

if st.button('Predict Interaction'):
    # Preprocess sequences
    # replace rare amino acids with X
    sequence_a = ' '.join(list(re.sub(r'[UZOB]', 'X', sequence_a)))
    # replace rare amino acids with X
    sequence_b = ' '.join(list(re.sub(r'[UZOB]', 'X', sequence_b)))
    example = sequence_a + ' [SEP] ' + sequence_b  # add SEP token

    # Load selected model and tokenizer
    model = MODEL_DICT[selected_model_name]['model']
    tokenizer = MODEL_DICT[selected_model_name]['tokenizer']

    # Tokenize the sequences
    example = tokenizer(example, return_tensors='pt', padding=False).to(device)

    # Make prediction
    with torch.no_grad():
        # get logits from model
        logits = model(**example).logits.cpu().detach()

    # use softmax to get "confidence" in the prediction
    probability = F.softmax(logits, dim=-1)
    # 0 for no interaction, 1 for interaction
    prediction = probability.argmax(dim=-1).item()

    # Display prediction
    if prediction == 1:
        st.success(
            f"The model predicts an **interaction** with confidence {probability[0][1].item():.2f}")
    else:
        st.error(
            f"The model predicts **no interaction** with confidence {probability[0][0].item():.2f}")
