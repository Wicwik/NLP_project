from dataclasses import field, dataclass, asdict
from typing import Optional, Dict

from transformers.utils import PushToHubMixin

import os, json

@dataclass
class PeftConfig(PushToHubMixin):
    def to_dict(self) -> Dict:
        return asdict(self)

    base_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Base model name or path"}
    )
    peft_type: Optional[str] = field(default=None, metadata={"help": "Peft type"})
    task_type: Optional[str] = field(default=None, metadata={"help": "Task type"})
    inference_mode: bool = field(default=False, metadata={"help": "Inference mode"})

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

        config = PeftConfig()

        for key, value in loaded_attributes.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    @classmethod
    def from_json_file(cls, path_json_file):
        with open(path_json_file, "r") as file:
            json_object = json.load(file)

        return json_object