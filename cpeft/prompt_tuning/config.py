import os
import json

from ..config import PeftConfig

from dataclasses import field, dataclass
from typing import Optional


@dataclass
class PromptTuningConfig(PeftConfig):
    prompt_init: str = field(
        default="random",
        metadata={
            "help": "Prompt init type [random, vocab, embedding], default is RANDOM."
        },
    )
    num_virtual_tokens: int = field(
        default=100, metadata={"help": "Soft prompt lenght."}
    )
    token_dim: int = field(
        default=None, metadata={"help": "Hidden embedding dim of the model."}
    )
    num_transformer_submodules: Optional[int] = field(
        default=None,
        metadata={"help": "Number of transformer submodules (e.g. decoder, encoder)."},
    )
    num_attention_heads: Optional[int] = field(
        default=None, metadata={"help": "Number of attention heads in the base model."}
    )
    num_layers: Optional[int] = field(
        default=None, metadata={"help": "Number of layers in the model."}
    )

    n_targets: Optional[int] = field(
        default=None, metadata={"help": "Number of target tasks."}
    )

    def __post_init__(self):
        self.peft_type = "prompt_tuning"

    @property
    def is_prompt_learning(self) -> bool:
        return True
