# Adaptive-MT-LLM-Fine-tuning

Code and data for the paper [Fine-tuning Large Language Models for Adaptive Machine Translation](https://arxiv.org/abs/2312.12740)

# Citations
```
@ARTICLE{Moslem2023-Finetuning-LLM-AdaptiveMT,
  title         = "{Fine-tuning Large Language Models for Adaptive Machine
                   Translation}",
  author        = "Moslem, Yasmin and Haque, Rejwanul and Way, Andy",
  abstract      = "This paper presents the outcomes of fine-tuning Mistral 7B,
                   a general-purpose large language model (LLM), for adaptive
                   machine translation (MT). The fine-tuning process involves
                   utilising a combination of zero-shot and one-shot
                   translation prompts within the medical domain. The primary
                   objective is to enhance real-time adaptive MT capabilities
                   of Mistral 7B, enabling it to adapt translations to the
                   required domain at inference time. The results, particularly
                   for Spanish-to-English MT, showcase the efficacy of the
                   fine-tuned model, demonstrating quality improvements in both
                   zero-shot and one-shot translation scenarios, surpassing
                   Mistral 7B's baseline performance. Notably, the fine-tuned
                   Mistral outperforms ChatGPT ``gpt-3.5-turbo'' in zero-shot
                   translation while achieving comparable one-shot translation
                   quality. Moreover, the zero-shot translation of the
                   fine-tuned Mistral matches NLLB 3.3B's performance, and its
                   one-shot translation quality surpasses that of NLLB 3.3B.
                   These findings emphasise the significance of fine-tuning
                   efficient LLMs like Mistral 7B to yield high-quality
                   zero-shot translations comparable to task-oriented models
                   like NLLB 3.3B. Additionally, the adaptive gains achieved in
                   one-shot translation are comparable to those of commercial
                   LLMs such as ChatGPT. Our experiments demonstrate that, with
                   a relatively small dataset of 20,000 segments that
                   incorporate a mix of zero-shot and one-shot prompts,
                   fine-tuning significantly enhances Mistral's in-context
                   learning ability, especially for real-time adaptive MT.",
  month         =  dec,
  year          =  2023,
  url           = "http://arxiv.org/abs/2312.12740",
  archivePrefix = "arXiv",
  primaryClass  = "cs.CL",
  eprint        = "2312.12740"
}

```

```

@INPROCEEDINGS{Moslem2023-AdaptiveMT,
  title     = "{Adaptive Machine Translation with Large Language Models}",
  booktitle = "{Proceedings of the 24th Annual Conference of the European
               Association for Machine Translation}",
  author    = "Moslem, Yasmin and Haque, Rejwanul and Kelleher, John D and Way,
               Andy",
  abstract  = "Consistency is a key requirement of high-quality translation. It
               is especially important to adhere to pre-approved terminology
               and adapt to corrected translations in domain-specific projects.
               Machine translation (MT) has achieved significant progress in
               the area of domain adaptation. However, real-time adaptation
               remains challenging. Large-scale language models (LLMs) have
               recently shown interesting capabilities of in-context learning,
               where they learn to replicate certain input-output text
               generation patterns, without further fine-tuning. By feeding an
               LLM at inference time with a prompt that consists of a list of
               translation pairs, it can then simulate the domain and style
               characteristics. This work aims to investigate how we can
               utilize in-context learning to improve real-time adaptive MT.
               Our extensive experiments show promising results at translation
               time. For example, GPT-3.5 can adapt to a set of in-domain
               sentence pairs and/or terminology while translating a new
               sentence. We observe that the translation quality with few-shot
               in-context learning can surpass that of strong encoder-decoder
               MT systems, especially for high-resource languages. Moreover, we
               investigate whether we can combine MT from strong
               encoder-decoder models with fuzzy matches, which can further
               improve translation quality, especially for less supported
               languages. We conduct our experiments across five diverse
               language pairs, namely English-to-Arabic (EN-AR),
               English-to-Chinese (EN-ZH), English-to-French (EN-FR),
               English-to-Kinyarwanda (EN-RW), and English-to-Spanish (EN-ES).",
  publisher = "European Association for Machine Translation",
  pages     = "227--237",
  month     =  jun,
  year      =  2023,
  url       = "https://aclanthology.org/2023.eamt-1.22",
  address   = "Tampere, Finland"
}

```
