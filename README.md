# Comparing Soft Prompt Transfer Parameter-Efficient Fine-Tuning Methods With Limited NLP Datasets
This repository features a replication study of the work [Attempt: Parameter-efficient multi-task tuning via attentional mixtures of soft prompts](https://aclanthology.org/2022.emnlp-main.446.pdf), published in the proceedings of the EMNLP 2022 conference. The original implentation can be found in the [author's repository](https://github.com/AkariAsai/ATTEMPT). Furthermore it also contains a series of experimets of comparing soft prompt transfer capabilities of SPoT and ATTEMPT methods on small and medium dataset sizes. 

### How to run
```
python3 -m venv repl
source repl/bin/activate
pip install -r requirements.txt
python run.py [config]
```

### Experiment results
All of our experiments details and saved data can be found at our Weights & Biases projects:
- [Prompt tuning](https://wandb.ai/rbelanec/prompt_tunning)
- [ATTEMPT single authors' prompts](https://wandb.ai/rbelanec/attempt_single_experiments)
- [ATTEMPT single our prompts](https://wandb.ai/rbelanec/attempt_single_ours_experiments)
- [ATTEMPT multi authors' prompts](https://wandb.ai/rbelanec/attempt_multi_experiments)
- [ATTEMPT multi our prompts](https://wandb.ai/rbelanec/attempt_multi_ours_experiments)
- [ATTEMPT small soft prompts](https://wandb.ai/rbelanec/nlp_as_soft_prompts)
- [ATTEMPT medium soft prompts](https://wandb.ai/rbelanec/nlp_am_soft_prompts)
- [SPoT small soft prompts](https://wandb.ai/rbelanec/nlp_ss_soft_prompts)
- [SPoT medium soft prompts](https://wandb.ai/rbelanec/nlp_sm_soft_prompts)
- [ATTEMPT small datasets](https://wandb.ai/rbelanec/nlp_attempt_small)
- [ATTEMTP medium datasets](https://wandb.ai/rbelanec/nlp_attempt_medium)
- [SPoT small datasets](https://wandb.ai/rbelanec/nlp_spot_small)
- [SPoT medium datasets](https://wandb.ai/rbelanec/nlp_spot_medium)

### References
[ATTEMPT: Parameter-Efficient Multi-task Tuning via Attentional Mixtures of Soft Prompts](https://aclanthology.org/2022.emnlp-main.446) (Asai et al., EMNLP 2022)

[The Power of Scale for Parameter-Efficient Prompt Tuning](https://aclanthology.org/2021.emnlp-main.243) (Lester et al., EMNLP 2021)

[SPoT: Better Frozen Model Adaptation through Soft Prompt Transfer](https://aclanthology.org/2022.acl-long.346) (Vu et al., ACL 2022)

MANGRULKAR, Sourab, et al. [PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods](https://github.com/huggingface/peft). 2022.
