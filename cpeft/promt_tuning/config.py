import os
import json

from ..config import PeftConfig

from dataclasses import field, dataclass, asdict
from typing import Optional


@dataclass
class PromptTuningConfig(PeftConfig):
    prompt_init: str = field(
        default="random",
        metadata={"help": "Prompt init type [random, vocab], default is RANDOM"},
    )
    num_virtual_tokens: int = field(
        default=100, metadata={"help": "Soft prompt lenght"}
    )
    token_dim: int = field(
        default=None, metadata={"help": "Hidden embedding dim of the model"}
    )
    num_transformer_submodules: Optional[int] = field(
        default=None,
        metadata={"help": "Number of transformer submodules (e.g. decoder, encoder)"},
    )
    num_attention_heads: Optional[int] = field(
        default=None, metadata={"help": "Number of attention heads in the base model"}
    )
    num_layers: Optional[int] = field(
        default=None, metadata={"help": "Number of layers in the model"}
    )

    def __post_init__(self):
        self.peft_type = "prompt_tuning"

    @property
    def is_prompt_learning(self) -> bool:
        return True

    def save_pretrained(self, save_directory):
        if os.path.isfile(save_directory):
            raise ValueError(f"{save_directory} is not a directory.")

        os.makedirs(save_directory, exist_ok=True)

        output_dict = asdict(self)
        output_path = os.path.join(save_directory, "adapter_config.json")

        with open(output_path, "w") as writer:
            writer.write(json.dumps(output_dict, indent=2, sort_keys=True))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        path = pretrained_model_name_or_path

        config_file = os.path.join(path, "adapter_config.json")

        loaded_attributes = cls.from_json_file(config_file)

        config = PromptTuningConfig()

        for key, value in loaded_attributes.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    @classmethod
    def from_json_file(cls, path_json_file):
        with open(path_json_file, "r") as file:
            json_object = json.load(file)

        return json_object
