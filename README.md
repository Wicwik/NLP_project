# ATTEMPT – Parameter-Efficient Multi-task Tuning via Attentional Mixtures of Soft Prompts: A Replication Study
This is a repository from the replication study of the work [Attempt: Parameter-efficient multi-task tuning via attentional mixtures of soft prompts](https://aclanthology.org/2022.emnlp-main.446.pdf), published in the proceedings of the EMNLP 2022 conference. The original implentation can be found in the [author's repository](https://github.com/AkariAsai/ATTEMPT).

This replication study is a part of [Replication Challenge](https://disai.eu/replication-challenge/) organized by [DisAI](https://disai.eu/).

### How to run
```
python3 -m venv repl
source repl/bin/activate
pip install -r requirements.txt
python run.py [config]
```

### Experiment results
All of our experiments results can be found at our Weights & Biases projects:
- [Prompt tuning](https://wandb.ai/rbelanec/prompt_tunning)
- [ATTEMPT single authors' prompts](https://wandb.ai/rbelanec/attempt_single_experiments)
- [ATTEMPT single our prompts](https://wandb.ai/rbelanec/attempt_single_ours_experiments)
- [ATTEMPT multi authors' prompts](https://wandb.ai/rbelanec/attempt_multi_experiments)
- [ATTEMPT multi our prompts](https://wandb.ai/rbelanec/attempt_multi_ours_experiments)

### References
ASAI, Akari, et al. [Attempt: Parameter-efficient multi-task tuning via attentional mixtures of soft prompts](https://aclanthology.org/2022.emnlp-main.446.pdf). In: Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing. 2022. p. 6655-6672.

MANGRULKAR, Sourab, et al. [PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods](https://github.com/huggingface/peft). 2022.
