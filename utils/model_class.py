from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
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
            self.loaded = True
        

    def model_inf(self, input_text:str, mother_lang:str = "eng_Latn")->str:
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
                translated = self._model.generate(**inputs,forced_bos_token_id = self.tokenizer.convert_tokens_to_ids("eng_Latn"))
                return self.tokenizer.decode(translated[0], skip_special_tokens=True)

            case "xlm_rbld":
                with open("utils/lang_map.json", "r", encoding="utf-8") as f:
                    FT_TO_NLLB = json.load(f)
                self.labels = self._model.config.id2label
                inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True).to(self._device)
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    pred_id = int(torch.argmax(probs, dim=-1))
                return FT_TO_NLLB.get(self.labels[pred_id],"unkown")

            case _:
                raise NotImplementedError(f"Inference not implemented for {self.model_key}")

def sentiments_eval(model_key:str, phrase:str)->str:
    model = Model(model_key=model_key)
    model.model_load()
    return model.model_inf(phrase)

def translator(model_key:str, phrase:str, language:str)->str:
    model = Model(model_key=model_key)
    model.model_load()
    return model.model_inf(phrase,mother_lang=language)

def language_detector(model_key:str, phrase:str)->str:
    model = Model(model_key=model_key)
    model.model_load()
    return model.model_inf(phrase)

def get_context():
    ...

def urgent_notif():
    ...

def typical_answer():
    ...
