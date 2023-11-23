import torch
from cpeft import PeftConfig

from transformers import PreTrainedModel
from typing import Dict

class PeftModel(torch.nn.Module):
    def __init__(self, model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "peft"):
        self._peft_config = {adapter_name: peft_config}
        self.base_model = model

        self.add_adapter(adapter_name, peft_config)

    @property
    def peft_config(self) -> Dict[str, str]:
        return self._peft_config
    
    @peft_config.setter
    def peft_config(self, value: Dict[str, str]):
        self.peft_config = value

    def add_adapter(self, adapter_name: str, peft_config: PeftConfig):
        if peft_config.peft_type == "prompt_tuning":
            self.peft_config[adapter_name] = peft_config
            self._setup_prompt_encoder(adapter_name)

    def _setup_prompt_encoder(self, adapter_name):
        config = self.peft_config[adapter_name]

