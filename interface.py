import streamlit as st
from utils.model_class import translator, language_detector, sentiments_eval, answer


st.set_page_config(
    page_title="My App",
    page_icon="🤖",
    layout="centered"
)

# Title
st.title("🤖 My Streamlit App")

# Optional: Add a subtitle or description
st.markdown("Enter your text below and click the button to process it.")

# Input field
user_input = st.text_input(
    label="Enter your text:",
    placeholder="Type something here...",
    max_chars=500
)


# Button
if st.button("Analyze Sentiment", type="primary"):
    if user_input.strip():
        # Create a placeholder for loading message
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
    if result in ['Positive', 'Very Positive']:
        st.write(f'## Comment is good, not important.')
    else:
        with st.spinner("Checking the bullshit..."):
                context = answer(model_key="mistral_gptq_4b",phrase=user_input, sentiment=result)
        if context == "irrelevant":
            st.write(f"### The BS is irrelevant")
        else:
            st.write(f'## Typical appology:')
            st.markdown(f"### {context}")

# Optional: Add a footer or additional info
st.markdown("---")
st.caption("Built with Streamlit 🎈")