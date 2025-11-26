from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from auto_gptq import AutoGPTQForCausalLM
import torch
import json


class Model:
    def __init__(self, model_key:str, cfg_file:str="utils/model_cfg.json"):
        #model : is the name of model, cfg_file is the json file model_cfg
        self.model_key = model_key
        self.cfg_file = cfg_file
        self.cfg = self._load_cfg()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._size = self._get_model_size()
        self._model = None
        self.labels = None
        self.loaded = False

    def _load_cfg(self)->dict:
        with open(self.cfg_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data[self.model_key]

    def _get_model_size(self)->int:
        return 0 #cha9wa

    def model_load(self):
        if not self.loaded:
            self.model_name = self.cfg["model_name"]
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.cfg["AMFS"] == "Classification":
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self._device)
                self.labels = self._model.config.id2label
            elif self.cfg["AMFS"] == "2SeqLM":
                self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self._device)
            elif self.cfg["AMFS"] == "GPTQ":
                self._model = AutoGPTQForCausalLM.from_quantized(
                    self.model_name,
                    use_safetensors=True,
                    trust_remote_code=True,
                    inject_fused_attention=False,
                    inject_fused_mlp=False,
                ).to(self._device)
            self.loaded = True

    def model_inf(self, input_text:str, mother_lang:str = "eng_Latn",sentiment_label: str = None)->str:
        match self.model_key:
            case "rsa":
                inputs = self.tokenizer(input_text, return_tensors="pt").to(self._device)
                with torch.no_grad():
                    outputs = self._model(**inputs)
                logits = outputs.logits
                pred = torch.argmax(logits, dim=1).item()
                return self.labels[pred]

            case "fb_nllb_1.3" | "fb_nllb_d_0.6":
                self.tokenizer.src_lang = mother_lang
                inputs = self.tokenizer(input_text, return_tensors="pt").to(self._device)        
                with torch.no_grad():    
                    translated = self._model.generate(**inputs,forced_bos_token_id = self.tokenizer.convert_tokens_to_ids("eng_Latn"))
                return self.tokenizer.decode(translated[0], skip_special_tokens=True)

            case "xlm_rbld":
                with open("utils/lang_map.json", "r", encoding="utf-8") as f:
                    FT_TO_NLLB = json.load(f)
                inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True).to(self._device)
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    pred_id = int(torch.argmax(probs, dim=-1))
                return FT_TO_NLLB.get(self.labels[pred_id],"unkown")
            
            case "mistral_gptq_4b":
                with open("utils/system_prompt.txt", "r", encoding="utf-8") as f:
                    SYSTEM_PROMPT  = f.read()
                user_content = (
                    SYSTEM_PROMPT
                    + "\n\nNow analyze the following input and respond according to the rules above.\n\n"
                    + f'feedback_text: "{input_text}"\n'
                    + f'sentiment_label: "{sentiment_label}"\n\n'
                    + "Remember: respond with EXACTLY 'irrelevant' if it is out of scope or not negative, "
                    "otherwise respond with a short apology message as described."
                )
                messages = [
                    {"role": "user", "content": user_content}
                ]
                prompt_text = self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self._device)

                with torch.inference_mode():
                    output = self._model.generate(**inputs,max_new_tokens=128,do_sample=False,pad_token_id=self.tokenizer.eos_token_id)
                    generated_ids = output[0][inputs["input_ids"].shape[-1]:]
                    response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                    lower = response.lower().strip()
                if lower.startswith("irrelevant"):
                    return "irrelevant"
                return response

            case _:
                raise NotImplementedError(f"Inference not implemented for {self.model_key}")
    
    def unload(self):
        if self._model:
            del self._model
            self._model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        self.loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        import gc
        gc.collect()
    def __del__(self):
        self.unload()
