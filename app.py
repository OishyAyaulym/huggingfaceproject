import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import random

model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner=pipeline("ner",model=model, tokenizer=tokenizer, grouped_entities=True)
st.title("💬 Атауын тану (Named Entity Recognition)")
st.write("Мәтінді енгізіңіз - модель атауларды (адам, ұйым, орын, т.б.) автоматты түрде таниды.")
user_input=st.text_area("Мәтінді осында жазыңыз:")
if st.button("Анализ жасау"):
    if not user_input.strip():
        st.warning("Мәтін енгізіңіз!")
    else:
        with st.spinner("Талдау жүріп жатыр..."):
            results=ner(user_input)

        highlighted_text = user_input
        for ent in results:
            word = ent['word'].replace("##", "")
            color = f"#{random.randint(0, 0xFFFFFF):06x}"
            highlighted_text = highlighted_text.replace(
                word,
                f"<span style='background-color:{color}; padding:4px; border-radius:4px'>{word} ({ent['entity_group']})</span>",
                1
            )

        st.subheader("🔍 Нәтиже:")
        st.markdown(highlighted_text, unsafe_allow_html=True)