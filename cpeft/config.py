from dataclasses import field, dataclass, asdict
from typing import Optional, Dict

@dataclass
class PeftConfig:
    def to_dict(self) -> Dict:
        return asdict(self)

    base_model_name_or_path : Optional[str] = field(default=None, metadata={"help": "Base model name or path"})
    peft_type : Optional[str]  = field(default=None, metadata={"help": "Peft type"})
    task_type : Optional[str]  = field(default=None, metadata={"help": "Task type"})
