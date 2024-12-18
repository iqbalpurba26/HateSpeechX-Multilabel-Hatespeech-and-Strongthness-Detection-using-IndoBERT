import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

def load_models():
    hate_category_model_name = "iqbalpurba26/IndoBERT-multilabel-hatespeech"
    hate_category_tokenizer = AutoTokenizer.from_pretrained(hate_category_model_name)
    hate_category_model = TFAutoModelForSequenceClassification.from_pretrained(hate_category_model_name)
    
    hate_strength_model_name = "iqbalpurba26/IndoBERT-strengthness-hatespeech-detection"
    hate_strength_tokenizer = AutoTokenizer.from_pretrained(hate_strength_model_name)
    hate_strength_model = TFAutoModelForSequenceClassification.from_pretrained(hate_strength_model_name)
    
    return (hate_category_tokenizer, hate_category_model,
            hate_strength_tokenizer, hate_strength_model)

def predict_hate_speech_category(text, tokenizer, model, threshold=0.5):
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=500)
    
    preds = model.predict(inputs).logits
    preds_probs = tf.sigmoid(preds).numpy()
    preds_label = (preds_probs > threshold).astype(int)
    
    categories = ['abusive', 'hs_individual', 'hs_group', 'hs_religion', 'hs_race', 'hs_other']
    
    output = {}
    if categories:
        for i, name in enumerate(categories):
            output[name] = {'probability': preds_probs[0][i], 'label': int(preds_label[0][i])}
    else:
        output = {'probability': preds_probs[0], 'label': int(preds_label[0])}
    
    return output

def predict_hate_speech_strength(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=500)
    preds = model.predict(inputs).logits
    preds_probs = tf.nn.softmax(preds).numpy()

    strength_categories = ['hs_weak', 'hs_moderate', 'hs_strong']
    max_index = preds_probs[0].argmax()
    return strength_categories[max_index]

def display_results(categories, strength=None):
    st.markdown("""
    <style>
        .indicator-container {
            display: flex;
            align-items: center;
            margin: 10px 0;
            font-size: 16px;
        }
        .indicator {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
            display: inline-block;
        }
        .indicator-text {
            font-size: 16px;
        }
        .detected {
            background-color: #ff4b4b;
        }
        .not-detected {
            background-color: #808080;
        }
        .strength-text {
            margin-top: 10px;
            font-size: 18px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.subheader("Detection Results:")
    
    sorted_categories = sorted(
        categories.items(),
        key=lambda x: x[1]['probability'],
        reverse=True
    )
    
    midpoint = len(sorted_categories) // 2
    left_col_categories = sorted_categories[:midpoint]
    right_col_categories = sorted_categories[midpoint:]
    
    col1, col2 = st.columns(2)

    with col1:
        for label, result in left_col_categories:
            prob_percentage = f"{result['probability']*100:.1f}%"
            indicator_class = "detected" if result['label'] == 1 else "not-detected"
            
            st.markdown(
                f"""
                <div class="indicator-container">
                    <div class="indicator {indicator_class}"></div>
                    <span class="indicator-text">
                        <strong>{label}</strong> ({prob_percentage})
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    with col2:
        for label, result in right_col_categories:
            prob_percentage = f"{result['probability']*100:.1f}%"
            indicator_class = "detected" if result['label'] == 1 else "not-detected"
            
            st.markdown(
                f"""
                <div class="indicator-container">
                    <div class="indicator {indicator_class}"></div>
                    <span class="indicator-text">
                        <strong>{label}</strong> ({prob_percentage})
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
            
    if strength:
        strength_cleaned = strength.replace("hs_", "").capitalize()
        st.markdown(f"""
        <div class="strength-text">
            <strong>Hate Speech Strengthness:</strong> {strength_cleaned}
        </div>
        """, unsafe_allow_html=True)

def main():
    st.title("Hate Speech Detection")

    (category_tokenizer, category_model,
     strength_tokenizer, strength_model) = load_models()

    st.write("Mau tahu perkataan yang kamu terima termasuk hate speech atau engga? Yuk mari kita cek bareng-bareng:)")
    input_text = st.text_area("", height=100)

    if st.button("Submit"):
        if input_text:
            with st.spinner("Menganalisis teks..."):
                categories = predict_hate_speech_category(
                    input_text,
                    category_tokenizer,
                    category_model
                )
                
                relevant_categories = ['hs_individual', 'hs_group', 'hs_religion', 'hs_race', 'hs_other']
                should_check_strength = any(categories[cat]['label'] == 1 for cat in relevant_categories)

                strength = None
                if should_check_strength:
                    strength = predict_hate_speech_strength(
                        input_text,
                        strength_tokenizer,
                        strength_model
                    )

            display_results(categories, strength)
        else:
            st.warning("Mohon masukkan teks terlebih dahulu.")

if __name__== "__main__":
    main()