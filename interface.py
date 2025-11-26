import streamlit as st
from utils.model_class import Model

def sentiments_eval(model_key:str, phrase:str)->str:
    model = Model(model_key=model_key)
    model.model_load()
    message = model.model_inf(phrase)
    model.unload()
    return message
    

def translator(model_key:str, phrase:str, language:str)->str:
    model = Model(model_key=model_key)
    model.model_load()
    message = model.model_inf(input_text=phrase,mother_lang=language)
    model.unload()
    return message

def language_detector(model_key:str, phrase:str)->str:
    model = Model(model_key=model_key)
    model.model_load()
    message = model.model_inf(input_text=phrase)
    model.unload()
    return message

def answer(model_key:str, phrase:str, sentiment:str)->str:
    model = Model(model_key=model_key)
    model.model_load()
    message = model.model_inf(input_text=phrase,sentiment_label=sentiment)
    model.unload()
    return message

st.set_page_config(
    page_title="Chatbot",
    page_icon="🤖",
    layout="centered"
)


st.title("Yaghmood's tourism agency")


st.markdown("This tool analyzes customer feedback, detects sentiment, checks relevance, and generates an appropriate response.")


user_input = st.text_area(
    label="Enter your text:",
    placeholder="Type your feedback here...",
    height=120,
    max_chars=500
)



if st.button("Analyze Sentiment", type="primary"):
    if user_input.strip():

        loading_placeholder = st.empty()
        result_placeholder = st.empty()

        eng_comment = user_input
        with st.spinner("Detecting Language..."):
            lang = language_detector(model_key="xlm_rbld",phrase=user_input)

        if lang!="eng_Latn":
            with st.spinner("Translating..."):
                eng_comment = translator(model_key="fb_nllb_1.3",phrase=user_input,language=lang)
        with st.spinner("Analyzing Sentiment..."):
            result = sentiments_eval(model_key="rsa",phrase=eng_comment)
                
            loading_placeholder.empty()

            st.success("✅ Analysis complete!")
            st.write("### Sentiment Result:")
            
            st.markdown(f"#### 😐 {result}")
    else:
        st.warning("⚠️ Please enter some text first!")
    if result not in ['Positive', 'Very Positive']:
        with st.spinner("Checking the context..."):
                context = answer(model_key="mistral_gptq_4b",phrase=user_input, sentiment=result)
        if context == "irrelevant":
            st.write(f"### The user complaint is irrelevant to the agency context.")
        else:
            st.write(f'## Typical appology:')
            st.markdown(f"### {context}")
