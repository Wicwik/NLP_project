import torch
from cpeft import PeftConfig, PromptTunningEmbedding

from transformers import PreTrainedModel
from transformers.utils import PushToHubMixin
from typing import Dict

from .utils import _prepare_prompt_learning_config

class PeftModel(PushToHubMixin, torch.nn.Module):
    def __init__(self, model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "peft"):
        super().__init__()
        self.base_model = model

        self._peft_config = {adapter_name: peft_config}
        self._is_prompt_learning = peft_config.is_prompt_learning

        self.config = getattr(self.base_model, "config", {"model_type": "custom"})
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

            if hasattr(self.config, "to_dict"):
                dict_config = self.config.to_dict()
            else:
                dict_config = self.config

            peft_config = _prepare_prompt_learning_config(peft_config, dict_config)
            self._setup_prompt_encoder(adapter_name)

    def _setup_prompt_encoder(self, adapter_name):
        config = self.peft_config[adapter_name]

        if not hasattr(self, "prompt_encoder"):
            self.prompt_encoder = torch.nn.ModuleDict({})
            self.prompt_tokens = {}

        transformer_backbone = None

        # freeze all base model parameters
        for name, module in self.base_model.named_children():
            for param in module.parameters():
                param.requires_grad = False

            if isinstance(module, PreTrainedModel):
                # Make sure to freeze Tranformers model
                if transformer_backbone is None:
                    transformer_backbone = module
                    self.transformer_backbone_name = name

        if transformer_backbone is None:
            transformer_backbone = self.base_model

        # set the number of modules
        if config.num_transformer_submodules is None:
            config.num_transformer_submodules = 2 if config.task_type == "seq_2_seq_lm" else 1

        for named_param, value in list(transformer_backbone.named_parameters()):
            if value.shape[0] == self.base_model.config.vocab_size:
                self.word_embeddings = transformer_backbone.get_submodule(named_param.replace(".weight", ""))
                break

        if config.peft_type == "prompt_tuning":
            prompt_encoder = PromptTunningEmbedding(config, self.word_embeddings)

        prompt_encoder = prompt_encoder.to(self.device)
        self.prompt_encoder.update(torch.nn.ModuleDict({adapter_name: prompt_encoder}))
        self.prompt_tokens[adapter_name] = torch.arange(config.num_virtual_tokens * config.num_transformer_submodules).long()

class PeftModelForSeq2SeqLM(PeftModel):
    def __init__(self, model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "seq2seq_peft"):
        super().__init__(model, peft_config, adapter_name)