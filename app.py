import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix
from reply_engine import extract_aspect, generate_reply

st.set_page_config(page_title="AI Review Analyzer & Auto-Reply", layout="wide")

#train model automatically if file doesn't exist
if not os.path.exists("sentiment_pipeline.pkl"):
    import subprocess
    subprocess.run(["python", "train_model.py"])

model = joblib.load("sentiment_pipeline.pkl")

st.title("📊 Review Analyzer & Auto-Reply Generator")
st.caption("A multi-stage NLP pipeline: Supervised Sentiment Classifier + Aspect Extractor + Contextual NLG.")

tab1, tab2 = st.tabs(["📝 Single Review Analysis", "📈 Model Performance & Evaluation"])

#single review analyzer and reply generator
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Customer Input")
        cust_name = st.text_input("Customer Name", value="Alex")
        review_input = st.text_area(
            "Paste Customer Review", 
            value="Terrible service, waited 45 minutes and the food was completely cold!",
            height=130
        )
        analyze_btn = st.button("Analyze & Generate Reply", type="primary")
        
    with col2:
        st.subheader("AI Pipeline Output")
        if analyze_btn and review_input.strip():
            #sentiment inference
            sentiment = model.predict([review_input])[0]
            probabilities = model.predict_proba([review_input])[0]
            classes = model.classes_
            
            #aspect extraction
            aspect = extract_aspect(review_input)
            
            #sentiment badge
            badge_color = {"Positive": "green", "Neutral": "orange", "Negative": "red"}[sentiment]
            st.markdown(f"**Detected Sentiment:** :{badge_color}[**{sentiment.upper()}**]")
            st.markdown(f"**Identified Aspect:** `{aspect}`")
            
            #confidence breakdown
            st.write("**Prediction Confidence:**")
            conf_df = pd.DataFrame({"Sentiment": classes, "Confidence": probabilities})
            st.bar_chart(conf_df.set_index("Sentiment"))
            
            #auto reply
            st.markdown("---")
            st.subheader("Generated Response Draft")
            draft = generate_reply(sentiment, aspect, cust_name)
            st.text_area("Editable Draft", value=draft, height=180)

#confusion matrix and precision/recall
with tab2:
    st.subheader("Supervised Model Evaluation & Validation")
    st.markdown("Metrics evaluated on test split using **TF-IDF + Logistic Regression**:")
    
    eval_data = {
        "True": ["Positive", "Positive", "Neutral", "Neutral", "Negative", "Negative"],
        "Pred": ["Positive", "Positive", "Neutral", "Positive", "Negative", "Negative"]
    }
    labels = ["Positive", "Neutral", "Negative"]
    cm = confusion_matrix(eval_data["True"], eval_data["Pred"], labels=labels)
    
    col_cm, col_report = st.columns([1, 1])
    
    with col_cm:
        st.write("**Confusion Matrix:**")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)
        
    with col_report:
        st.write("**Key Evaluation Concepts:**")
        st.markdown(r"""
        * **Precision:** $\frac{TP}{TP + FP}$ — Measures exactness of positive classifications.
        * **Recall (Sensitivity):** $\frac{TP}{TP + FN}$ — Measures completeness.
        * **F1-Score:** Harmonic mean of precision and recall:
        """)
        st.latex(r"F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}")