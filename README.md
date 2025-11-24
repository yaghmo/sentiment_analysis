# Yaghmood Feedback Sentiment & Relevance Classifier

This project is an experimental chatbot designed to analyze customer feedback, determine the sentiment, check whether the complaint is relevant to the agency’s context, and optionally generate a short and polite response. It combines multiple AI models (sentiment classifier, language detector, translator, and a Mistral-7B assistant) to create a complete multi-step triage pipeline.

> **Important:**  
> All “agency context” logic is purely experimental and **does not reflect any real business decisions, operations, or the actual Yaghmood company**.  
> This project is intended for research, prototyping, and AI behavior testing only.

---

## 🚀 Highlights

- 🔎 Automatic sentiment detection using a robust multi-class classifier  
- 🌍 Automatic language detection & translation to English for multilingual feedback  
- 🎯 Context-aware relevance judgment using a Mistral 7B GPTQ model  
- 🧠 Fully customizable system prompt (edit `utils/system_promt.txt` to change assistant behavior instantly)  
- 🤖 End-to-end chatbot pipeline combining sentiment, relevance, and apology generation  
- ⚡ GPU-accelerated inference (recommended: **6GB VRAM or more**)  
- 🖥️ Simple Streamlit UI for testing and demonstration  

---

## 🧩 Models Used

All models run on GPU:

| Model Name | Type | Purpose |
|------------|------|---------|
| `tabularisai/robust-sentiment-analysis` | Classification | Classifies sentiment into 5 categories |
| `facebook/nllb-200-1.3B` | Seq2Seq | Translates any supported language to English |
| `papluca/xlm-roberta-base-language-detection` | Classification | Detects input language |
| `RedHatAI/Mistral-7B-Instruct-v0.3-GPTQ-4bit` | GPTQ | Determines relevance & generates responses |

---

## 📦 Installation

It’s recommended to use a dedicated environment (conda or Python venv).

### 1. Create a virtual environment (example with conda)

```
conda create -n env_name python=3.10
conda activate env_name
```
### 2. Install all dependencies
From the project root:
```
pip install -r requirements.txt
```


## 🧪 Usage

After installing dependencies, launch the interface with:

```
streamlit run interface.py
```

A web application will open where you can input user feedback.
The system will then:

1. Detect language

2. Translate if necessary (refer to the doccumentation to know the supported languages)

3. Analyze sentiment

4. Check contextual relevance

5. Output either:
    - irrelevant, or
    - a short apology message


## 📸 Examples
Example 1 — Relevant & Negative:
i am very angry the flight to sweden from france at 8am was delayed by 2 hours and the hotel is now full while i already booked 2 days!

Expected:
A short apology response with contact instructions.

![Relevant and Negative](image.png)

Example 2 — Negative but Irrelevant (cars):
i hate this agency so much i cant even find cars to rent

Expected:
irrelevant

![Negative but Irrelevant](image-1.png)

Example 3 — Positive using French language:
![Positive](image-2.png)


## 📚 Citations

Please cite the following models and authors if you use this project in academic, research, or public work.

Sentiment Analysis

Robust Sentiment Analysis — tabularisai/robust-sentiment-analysis
```
@misc{tabularisai_robust_sentiment_analysis,
  title={Robust Sentiment Analysis},
  author={TabularisAI},
  year={2023},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/tabularisai/robust-sentiment-analysis}}
}
```
Language Detection

XLM-RoBERTa Base Language Detection — papluca/xlm-roberta-base-language-detection
```
@misc{papluca_xlm_roberta_langdetect,
  title={XLM-RoBERTa Base Language Detection},
  author={Papluca},
  year={2022},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/papluca/xlm-roberta-base-language-detection}}
}
```

Translation

NLLB-200 1.3B — facebook/nllb-200-1.3B
```
@inproceedings{nllbteam2022nllb,
  author    = {NLLB Team and Meta AI},
  title     = {No Language Left Behind: Scaling Human-Centered Machine Translation},
  booktitle = {arXiv preprint arXiv:2207.04672},
  year      = {2022},
  url       = {https://huggingface.co/facebook/nllb-200-1.3B}
}
```

Context Understanding / Assistant Model

Mistral-7B-Instruct-v0.3 — RedHatAI/Mistral-7B-Instruct-v0.3-GPTQ-4bit

```
@misc{mistral_instruct_v03,
  title={Mistral 7B Instruct v0.3},
  author={Mistral AI},
  year={2024},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3}}
}
```