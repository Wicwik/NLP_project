
from dataclasses import field, dataclass, asdict
from typing import Optional, Dict

from transformers.utils import PushToHubMixin
@dataclass
class PeftConfig(PushToHubMixin):
    def to_dict(self) -> Dict:
        return asdict(self)

    base_model_name_or_path : Optional[str] = field(default=None, metadata={"help": "Base model name or path"})
    peft_type : Optional[str]  = field(default=None, metadata={"help": "Peft type"})
    task_type : Optional[str]  = field(default=None, metadata={"help": "Task type"})
    inference_mode: bool = field(default=False, metadata={"help": "Inference mode"})

