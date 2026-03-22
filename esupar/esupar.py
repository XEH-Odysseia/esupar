#! /usr/bin/python -i
# coding=utf-8

MODELS={
  "ain":"KoichiYasuoka/roberta-base-ainu-upos",
  "cop":"KoichiYasuoka/roberta-base-coptic-upos",
  "de":"KoichiYasuoka/bert-base-german-upos",
  "de_base":"KoichiYasuoka/bert-base-german-upos",
  "de_large":"KoichiYasuoka/bert-large-german-upos",
  "en":"KoichiYasuoka/roberta-base-english-upos",
  "en_base":"KoichiYasuoka/roberta-base-english-upos",
  "en_large":"KoichiYasuoka/roberta-large-english-upos",
  "ja":"KoichiYasuoka/bert-base-japanese-upos",
  "ja_base":"KoichiYasuoka/bert-base-japanese-upos",
  "ja_large":"KoichiYasuoka/bert-large-japanese-upos",
  "ja_luw":"KoichiYasuoka/bert-base-japanese-luw-upos",
  "ja_luw_small":"KoichiYasuoka/roberta-small-japanese-char-luw-upos",
  "ja_luw_base":"KoichiYasuoka/bert-base-japanese-luw-upos",
  "ja_luw_large":"KoichiYasuoka/bert-large-japanese-luw-upos",
  "ko":"KoichiYasuoka/roberta-base-korean-upos",
  "ko_base":"KoichiYasuoka/roberta-base-korean-upos",
  "ko_large":"KoichiYasuoka/roberta-large-korean-upos",
  "ko_morph":"KoichiYasuoka/roberta-base-korean-morph-upos",
  "ko_morph_base":"KoichiYasuoka/roberta-base-korean-morph-upos",
  "ko_morph_large":"KoichiYasuoka/roberta-large-korean-morph-upos",
  "lzh":"KoichiYasuoka/roberta-classical-chinese-base-upos",
  "lzh_base":"KoichiYasuoka/roberta-classical-chinese-base-upos",
  "lzh_large":"KoichiYasuoka/roberta-classical-chinese-large-upos",
  "sr":"KoichiYasuoka/gpt2-small-serbian-upos",
  "sr_large":"KoichiYasuoka/gpt2-large-serbian-upos",
  "th":"KoichiYasuoka/roberta-base-thai-spm-upos",
  "vi":"KoichiYasuoka/bert-base-vietnamese-upos",
  "zh":"KoichiYasuoka/chinese-bert-wwm-ext-upos",
  "zh_bert":"KoichiYasuoka/chinese-bert-wwm-ext-upos",
  "zh_base":"KoichiYasuoka/chinese-roberta-base-upos",
  "zh_large":"KoichiYasuoka/chinese-roberta-large-upos"
}

class Esupar(object):
    def __init__(
        self,
        model,
        lemma=None,
        use_adapter=False,
        adapter_path=None,
        adapter_name="historical_ko",
    ):
        import os, numpy
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        from huggingface_hub import hf_hub_download
        from esupar.supar import Parser

        self.tokenizer = AutoTokenizer.from_pretrained(model)

        try:
            self.tokenizerfast = self.tokenizer.is_fast
        except:
            self.tokenizerfast = (str(type(self.tokenizer)).find("TokenizerFast") > 0)

        if not self.tokenizerfast:
            try:
                if self.tokenizer.word_tokenizer_type == "mecab":
                    from esupar.mecab import BertMecabTokenizerFast
                    self.tokenizer = BertMecabTokenizerFast.from_pretrained(model)
                    self.tokenizerfast = True
            except:
                pass

        # Avoid issues with safetensors auto-conversion
        self.tagger = AutoModelForTokenClassification.from_pretrained(
            model,
            use_safetensors=False
        )

        self.use_adapter = use_adapter
        self.adapter_path = adapter_path
        self.adapter_name = adapter_name

        # 1) local file first
        f = os.path.join(model, "esupar.model")
        if os.path.isfile(f):
            self.parser = Parser.load(
                f,
                safe_tensor=True,
                use_adapter=use_adapter,
                adapter_path=adapter_path,
                adapter_name=adapter_name,
            )
        else:
            # 2) Directly download from HF Hub
            try:
                f = hf_hub_download(repo_id=model, filename="esupar.model")
                self.parser = Parser.load(
                    f,
                    safe_tensor=True,
                    use_adapter=use_adapter,
                    adapter_path=adapter_path,
                    adapter_name=adapter_name,
                )
            except Exception:
                f = None

        # 3) fallback: supar.model
        if not f:
            f = os.path.join(model, "supar.model")
            if os.path.isfile(f):
                self.parser = Parser.load(
                    f,
                    use_adapter=use_adapter,
                    adapter_path=adapter_path,
                    adapter_name=adapter_name,
                )
            else:
                f = hf_hub_download(repo_id=model, filename="supar.model")
                self.parser = Parser.load(
                    f,
                    use_adapter=use_adapter,
                    adapter_path=adapter_path,
                    adapter_name=adapter_name,
                )

        # Keep the original code
        try:
            for x in self.parser.transform.flattened_fields:
                if x.fn:
                    x.fn = lambda t: " " + t
        except:
            pass

        x = self.tagger.config.id2label
        self.labelmatrix = numpy.full((len(x), len(x)), numpy.nan)
        d = numpy.array([numpy.nan if x[i].startswith("I-") else 0 for i in range(len(x))])
        for i in range(len(x)):
            if x[i].startswith("B-"):
                try:
                    self.labelmatrix[i, self.tagger.config.label2id["I-" + x[i][2:]]] = 0
                except:
                    self.labelmatrix[i] = 0
            else:
                self.labelmatrix[i] = d
                if x[i].startswith("I-"):
                    self.labelmatrix[i, i] = 0

        if not lemma in {"copy", "tradify", "simplify", "hangul", "ainu", "none"}:
            try:
                lemma = self.tagger.config.task_specific_params["esupar_lemmatize"]
            except:
                pass

        if lemma == "copy":
            self.lemma = lambda x: x
        elif lemma == "tradify":
            from esupar.tradify import tradify
            self.lemma = lambda x: "".join(tradify[c] if c in tradify else c for c in x)
        elif lemma == "simplify":
            from esupar.simplify import simplify
            self.lemma = lambda x: "".join(simplify[c] if c in simplify else c for c in x)
        elif lemma == "hangul":
            from esupar.hangul import hangul
            self.hangul = {}
            for k, v in hangul.items():
                for c in "".join(v).replace("(", "").replace(")", ""):
                    self.hangul[c] = k
            self.lemma = lambda x: "".join(self.hangul[c] if c in self.hangul else c for c in x)
        elif lemma == "ainu":
            from esupar.ainu import Lemmatize
            self.lemma = Lemmatize()
        else:
            self.lemma = lambda x: "_"

def load(model="ja",
         lemma=None,
         use_adapter=False,
         adapter_path=None,
         adapter_name="historical_ko"):
  if model in MODELS:
    model=MODELS[model]
  return Esupar(model,
                lemma,
                use_adapter=use_adapter,
                adapter_path=adapter_path,
                adapter_name=adapter_name)
