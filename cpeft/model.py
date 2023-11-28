import torch
import os
from cpeft import PeftConfig, PromptTunningEmbedding, PromptTuningConfig

from transformers import PreTrainedModel
from transformers.utils import PushToHubMixin
from typing import Dict, Any, List, Optional

from .utils import _prepare_prompt_learning_config, infer_device
from .save_and_load import get_peft_model_state_dict, set_peft_model_state_dict, load_peft_weights

class PeftModel(PushToHubMixin, torch.nn.Module):
    def __init__(self, model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "peft"):
        super().__init__()
        self.base_model = model
        self.device = self.base_model.device

        self._peft_config = {adapter_name: peft_config}
        self._is_prompt_learning = peft_config.is_prompt_learning

        self.config = getattr(self.base_model, "config", {"model_type": "custom"})
        self.add_adapter(adapter_name, peft_config)

        if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "pretraining_tp"):
            self.base_model.config.pretraining_tp = 1

    def save_pretrained(self, save_directory: str, selected_adapters: Optional[List[str]] = None, **kwargs: Any):
        if os.path.isfile(save_directory):
            raise ValueError(f"{save_directory} is not a directory.")

        if selected_adapters is None:
            selected_adapters = list(self.peft_config.keys())

        os.makedirs(save_directory, exist_ok=True)

        for adapter_name in selected_adapters:
            peft_config = self.peft_config[adapter_name]
            output_state_dict = get_peft_model_state_dict(self, state_dict=kwargs.get("state_dict", None), adapter_name=adapter_name)
            output_dir = os.path.join(save_directory, adapter_name) if adapter_name != "peft" else save_directory

            os.makedirs(output_dir, exist_ok=True)

            torch.save(output_state_dict, os.path.join(output_dir, "adapter_model.bin"))

            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True

            peft_config.save_pretrained(output_dir)
            peft_config.inference_mode = inference_mode

    @classmethod
    def from_pretrained(cls, model: PreTrainedModel, model_id: str, adapter_name: str = "peft", is_trainable: bool = False, config: Optional[PeftConfig] = None, **kwargs: Any):
        if config is None:
            config = PromptTuningConfig.from_pretrained(model_id, **kwargs)

        if config.is_prompt_learning and is_trainable:
            raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
        else:
            config.inference_mode = not is_trainable

        model = PeftModelForSeq2SeqLM(model, config, adapter_name)
        model.load_adapter(model_id, adapter_name, is_trainable=is_trainable, **kwargs)

        return model

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

    def load_adapter(self, model_id: str, adapter_name: str, is_trainable: bool = False, **kwargs: Any):
        torch_device = infer_device()

        adapters_weights = load_peft_weights(model_id, device=torch_device, **kwargs)

        load_result = set_peft_model_state_dict(self, adapters_weights, adapter_name=adapter_name)

        if not is_trainable:
            self.eval()

        return load_result
    
    def get_prompt_embedding_to_save(self, adapter_name: str):
        prompt_encoder = self.prompt_encoder[adapter_name]
        prompt_tokens = (self.prompt_tokens[adapter_name].unsqueeze(0).expand(1, -1).to(prompt_encoder.embedding.weight.device))
        prompt_embeddings = prompt_encoder(prompt_tokens)
        return prompt_embeddings[0].detach().cpu()

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

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.base_model, name)

    def forward(self, *args: Any, **kwargs: Any):
        return self.get_base_model()(*args, **kwargs)

class PeftModelForSeq2SeqLM(PeftModel):
    def __init__(self, model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "seq2seq_peft"):
        super().__init__(model, peft_config, adapter_name)