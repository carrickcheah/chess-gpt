---
base_model: unsloth/LFM2-350M
library_name: transformers
model_name: LFM2-350M-r16-a32-20250919-025808
tags:
- generated_from_trainer
- trl
- sft
licence: license
---

# Model Card for LFM2-350M-r16-a32-20250919-025808

This model is a fine-tuned version of [unsloth/LFM2-350M](https://huggingface.co/unsloth/LFM2-350M).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with SFT.

### Framework versions

- TRL: 0.22.2
- Transformers: 4.55.2
- Pytorch: 2.5.1
- Datasets: 4.1.1
- Tokenizers: 0.21.4

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```