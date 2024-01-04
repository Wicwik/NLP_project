import sys
import os

import utils

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

utils.start(__file__)

# create model using custom peft lib
from transformers import AutoModelForSeq2SeqLM
from cpeft import AttemptConfig, get_peft_model, PeftModel

cpeft_save = "test_cpeft_saved_model"
cpeft_config = AttemptConfig(
    task_type="seq_2_seq_lm",
    num_virtual_tokens=50,
    prompt_init="embedding",
    prompt_init_embedding="soft_prompts/mnli.bin",
    prompt_embedding_paths=[
        "soft_prompts/mnli.bin",
    ],
    prefix_num=2,
)

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
model = get_peft_model(model, cpeft_config)

# print(model)
utils.passed(__file__)
