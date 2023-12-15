import torch

from typing import Optional

from .config import PeftConfig


def _prepare_prompt_learning_config(peft_config: PeftConfig, model_config):
    peft_config.num_layers = model_config["num_layers"]
    peft_config.token_dim = model_config["d_model"]
    peft_config.num_attention_heads = model_config["num_heads"]

    if getattr(peft_config, "encoder_hidden_size", None) is None:
        setattr(peft_config, "encoder_hidden_size", peft_config.token_dim)

    return peft_config


def _get_batch_size(
    input_ids: Optional[torch.Tensor], inputs_embeds: Optional[torch.Tensor]
) -> int:
    if input_ids is not None:
        batch_size = input_ids.shape[0]
    else:
        batch_size = inputs_embeds.shape[0]
    return batch_size


def infer_device():
    if torch.cuda.is_available():
        torch_device = "cuda"
    else:
        torch_device = "cpu"
    return torch_device
