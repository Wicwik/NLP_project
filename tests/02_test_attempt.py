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

utils.passed(__file__)
