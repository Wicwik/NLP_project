from dataclasses import field, dataclass

@dataclass
class PeftConfig:
    peft_type : str  = field(default=None, metadata={"help": "Peft type"})
    task_type : str  = field(default=None, metadata={"help": "Task type"})