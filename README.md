# Adaptive-MT-LLM-Fine-tuning

Code and data for the paper [Fine-tuning Large Language Models for Adaptive Machine Translation](https://arxiv.org/abs/2312.12740)

The paper presents the outcomes of fine-tuning Mistral 7B, a general-purpose large language model (LLM), for adaptive machine translation (MT). The fine-tuning process involves utilizing a combination of zero-shot and one-shot translation prompts within the medical domain. Zero-shot prompts represet regular translation without any context, while one-shot prompts augment the new source with a similar translation pair, i.e. a fuzzy match, to improve the adherence to terminology and style of the domain The primary objective is to enhance real-time adaptive MT capabilities of Mistral 7B, enabling it to adapt translations to the required domain at inference time. Our experiments demonstrate that, with a relatively small dataset of 20,000 segments that incorporate a mix of zero-shot and one-shot prompts, fine-tuning significantly enhances Mistral's in-context learning ability, especially for real-time adaptive MT.

## Dependencies

You might want to install the latest versions of the used libraries, but if you are facing issues, try the versions used in the requirements file.
```
pip3 install -r requirements.txt
```

## Data (training and test)

The original dataset is a mix of medical datasets from [OPUS](https://opus.nlpl.eu/), namely ELRC, EMEA, SciELO, and TICO-19.

### Training data (small)

* **Fine-tuning data - small** [[ES](data/small-train/all-filtered.es.real.smalltrain)][[EN](data/small-train/all-filtered.en.real.smalltrain)]: Data for actual fine-tuning: 10,000 translation pairs
* **Context Dataset** [[ES](data/small-train/all-filtered.es.fuzzy.smalltrain)][[EN](data/small-train/all-filtered.en.fuzzy.smalltrain)]: Data for fuzzy match retrieval _for training_: 50,000 translation pairs
* [Retrieved data](data/small-train/all-filtered.esen.ms-multi-12.online.smalltrain): Data after retrieval _for training_: 10,000 entries (format: {score} ||| {fuzzy_src_sent} ||| {new_src_sent} ||| {fuzzy_tgt_sent})

### Test Data

* **Test dataset** [[ES](data/test/all-filtered.es.real.test)][[EN](data/test/all-filtered.en.real.test)]: Data used for actual inference/translation: 10,000 translation pairs
* **Context Dataset** [[ES](data/test/all-filtered.es.fuzzy.test)][[EN](data/test/all-filtered.en.fuzzy.test)]: Data for fuzzy match retrieval _for testing_: 50,000 translation pairs
* [Retrieved data](data/test/all-filtered.esen.ms-multi-12.online.test): Data after retrieval _for testing_: 10,000 entries (format: {score} ||| {fuzzy_src_sent} ||| {new_src_sent} ||| {fuzzy_tgt_sent})

## Data Processing

The original dataset is a mix of medical datasets from OPUS, namely ELRC, EMEA, SciELO, and TICO-19. The pre-processing step mainly removes duplicates and too long sentences. The code for data pre-processing is at [Data-Processing-Adaptive-MT.ipynb](Data-Processing-Adaptive-MT.ipynb)

## Fuzzy Match Retrieval

We use [Sentence-Transformers](https://www.sbert.net/) with a multilingual model, namely Microsoft’s “_Multilingual-MiniLM-L12-H384_”, to generate the embeddings for the datasets. For indexing, we use [Faiss](https://github.com/facebookresearch/faiss). Then we retrieve fuzzy matches through semantic search. You can find more details about the retrieval process in our [paper]([url](https://arxiv.org/abs/2312.12740)). The code of this fuzzy match retrieval process is at [Retrieve-Fuzzy-Matches-Faiss-Adaptive-MT.ipynb](Retrieve-Fuzzy-Matches-Faiss-Adaptive-MT.ipynb)

## Fine-tuning Mistral 7B

We used QLoRA for efficient fine-tuning with 4bit quantization, with Hugging Face Transformers. You can find more details in the [paper]([url](https://arxiv.org/abs/2312.12740)) and the notebook [Mistral-Fine-Tuning-Adaptive-MT.ipynb](Mistral-Fine-Tuning-Adaptive-MT.ipynb)

Prompts are created in this notebook using the `create_prompt()` function. If `one_shot=False` it creates a zero-shot translation prompt; otherwise, it creates a one-shot translation prompt. Please check out the notebook itself for actual examples.

## Inference

### Conversion to the CTranslate2 format

* **Mistral 7B (baseline)**: To convert Mistral baseline (before fine-tuning) to the CTranslate2 format:
```
ct2-transformers-converter --model mistralai/Mistral-7B-v0.1 --quantization int8 --output_dir ct2-mistral-7B-v0.1
```
* **Mistral 7B (fine-tuned):** To convert Mistral after FINE-TUNING to the CTranslate2 format, check the steps at [Convert-Mistral-Finetuned-CTranslate2.ipynb](Convert-Mistral-Finetuned-CTranslate2.ipynb)

* **NLLB-200**: To convert NLLB-200 to the CTranslate2 format:
```
ct2-transformers-converter --model facebook/nllb-200-distilled-600M --quantization int8 --output_dir ct2/nllb-200-distilled-600M-int8
```

### Tokenizers

* **Mistral 7B**: You can directly use the tokenizers from the Transformers library as illustrated in the notebook [Mistral-CTranslate2-Adaptive-MT.ipynb](Mistral-CTranslate2-Adaptive-MT.ipynb)
* **NLLB-200**: Download the SentencePiece model for NLLB-200; then use it as illustrated in the notebook [NLLB-200-CTranslate2-Adaptive-MT.ipynb](NLLB-200-CTranslate2-Adaptive-MT.ipynb)
```
!wget https://s3.amazonaws.com/opennmt-models/nllb-200/flores200_sacrebleu_tokenizer_spm.model
```

### Translation

* **Mistral 7B** (baseline and fine-tuned): Translation code with CTranslate2 is at [Mistral-CTranslate2-Adaptive-MT.ipynb](Mistral-CTranslate2-Adaptive-MT.ipynb)
* **NLLB-200**: Translation code with CTranslate2 is at [NLLB-200-CTranslate2-Adaptive-MT.ipynb](NLLB-200-CTranslate2-Adaptive-MT.ipynb)
* **ChatGPT**: Translation via the official API; the code is at [ChatGPT-Adaptive-MT.ipynb](ChatGPT-Adaptive-MT.ipynb)

## Evaluation

Evaluation was done based on BLEU, chrF++, TER, and COMET metrics. The code is available at [Evaluation-Adaptive-MT.ipynb](Evaluation-Adaptive-MT.ipynb). The full evaluation scores are available at the [paper](https://arxiv.org/abs/2312.12740) under the Results section, and a detailed version is at [Evaluation-Scores-Adaptive-MT.csv](Evaluation-Scores-Adaptive-MT.csv)

## Questions

If you have questions, please feel free to [contact me](https://blog.machinetranslation.io/contact/).

## Citations

1. **Fine-tuning Large Language Models for Adaptive Machine Translation**
   
```bibtex
@ARTICLE{Moslem2023-Finetuning-LLM-AdaptiveMT,
  title         = "{Fine-tuning Large Language Models for Adaptive Machine Translation}",
  author        = "Moslem, Yasmin and Haque, Rejwanul and Way, Andy",
  month         =  dec,
  year          =  2023,
  url           = "http://arxiv.org/abs/2312.12740",
  archivePrefix = "arXiv",
  primaryClass  = "cs.CL",
  eprint        = "2312.12740"
}

```
2. **Adaptive Machine Translation with Large Language Models**

```bibtex
@INPROCEEDINGS{Moslem2023-AdaptiveMT,
  title     = "{Adaptive Machine Translation with Large Language Models}",
  booktitle = "{Proceedings of the 24th Annual Conference of the European Association
               for Machine Translation}",
  author    = "Moslem, Yasmin and Haque, Rejwanul and Kelleher, John D and Way, Andy",
  publisher = "European Association for Machine Translation",
  pages     = "227--237",
  month     =  jun,
  year      =  2023,
  url       = "https://aclanthology.org/2023.eamt-1.22",
  address   = "Tampere, Finland"
}

```
