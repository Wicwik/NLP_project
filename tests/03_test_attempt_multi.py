import sys
import os

import utils

import torch

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

utils.start(__file__)
cpeft_save = "test_attempt_saved_model"

# create model using custom peft lib
from transformers import AutoModelForSeq2SeqLM
from cpeft import AttemptConfig, get_peft_model, PeftModel

cpeft_config = AttemptConfig(
    task_type="seq_2_seq_lm",
    num_virtual_tokens=50,
    prompt_init="embedding_multi",
    prompt_init_embedding="soft_prompts/mnli.bin",
    prompt_embedding_paths=[
        "soft_prompts/mnli.bin",
        "soft_prompts/qnli.bin",
        "soft_prompts/qqp.bin",
        "soft_prompts/record.bin",
        "soft_prompts/squad.bin",
        "soft_prompts/sst2.bin",
    ],
    prefix_num=6,
    shared_attn=True,
    n_targets=2
)

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
model = get_peft_model(model, cpeft_config).to("cuda")
weights = [model.prompt_encoder["peft"].embedding[i].weight for i, _ in enumerate(model.prompt_encoder["peft"].embedding)]

model.print_trainable_parameters()

model.save_pretrained(cpeft_save)

new_model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
new_model = PeftModel.from_pretrained(new_model, cpeft_save).to("cuda")
new_weights = [new_model.prompt_encoder["peft"].embedding[i].weight for i, _ in enumerate(model.prompt_encoder["peft"].embedding)]

# print(str(new_model._peft_config) == str(model._peft_config))
print(model)

assert str(model) == str(new_model), "Model is not the same after saving and loading."

for i, _ in enumerate(model.prompt_encoder["peft"].embedding):

    assert (
        model.prompt_encoder["peft"].embedding[i].weight == new_model.prompt_encoder["peft"].embedding[i].weight
    ).all(), "Prompt embeddings must be the same after save and load."

utils.passed(__file__)
