import sys
import os

import utils

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

utils.start(__file__)

# create model using custom peft lib
from transformers import AutoModelForSeq2SeqLM
from cpeft import PromptTuningConfig, get_peft_model, PeftModel

cpeft_save = "test_cpeft_saved_model"
cpeft_config = PromptTuningConfig(task_type="seq_2_seq_lm", num_virtual_tokens=10)

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
model = get_peft_model(model, cpeft_config)

# save it
model.save_pretrained(cpeft_save)

# load it again
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
model = get_peft_model(model, cpeft_config)
cpeft_model = PeftModel.from_pretrained(model, cpeft_save)

# create model using hf peft lib
from transformers import AutoModelForSeq2SeqLM
from peft import PromptTuningConfig, get_peft_model, TaskType, PeftModel

peft_save = "test_peft_saved_model"
pt_config = PromptTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, num_virtual_tokens=10)

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
model = get_peft_model(model, pt_config)
model.save_pretrained(peft_save)

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
model = get_peft_model(model, pt_config)
peft_model = PeftModel.from_pretrained(model, peft_save)

# print(cpeft_model)
# print(peft_model)

assert str(cpeft_model) == str(peft_model).replace("default", "peft").replace(
    "PromptEmbedding", "PromptTuningEmbedding"
), "Custom peft model does not equal to the hugging face model after saving and loading."

utils.passed(__file__)
