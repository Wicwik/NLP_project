# create model using custom peft lib

from transformers import AutoModelForSeq2SeqLM
from cpeft import PromptTuningConfig, get_peft_model, PeftModel

cpeft_config = PromptTuningConfig(task_type="seq_2_seq_lm", num_virtual_tokens=10)

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
model = get_peft_model(model, cpeft_config, adapter_name="default")

# save it
model.save_pretrained("cpeft_saved_model")

# load it again
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
model = get_peft_model(model, cpeft_config)
model = PeftModel.from_pretrained(model, "cpeft_saved_model")

print(model)

# create model using hf peft lib
from transformers import AutoModelForSeq2SeqLM
from peft import PromptTuningConfig, get_peft_model, TaskType, PeftModel

pt_config = PromptTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, num_virtual_tokens=10)

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
model = get_peft_model(model, pt_config)
model.save_pretrained("peft_saved_model")

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
model = get_peft_model(model, pt_config)
model = PeftModel.from_pretrained(model, "peft_saved_model")

print(model)