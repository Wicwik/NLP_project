from .model import PeftModelForSeq2SeqLM, PeftModel
from .config import PeftConfig
from .utils import _prepare_prompt_learning_config

from transformers import PreTrainedModel

def get_peft_model(model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "seq2seq_peft") -> PeftModel:

    model_config = model.config
    if hasattr(model_config, "to_dict"):
        model_config = model_config.to_dict()

        peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)

        peft_config = _prepare_prompt_learning_config(peft_config, model_config)

        return PeftModelForSeq2SeqLM(model, peft_config, adapter_name)