from collections import OrderedDict

class AbstractTaskType:
    name = NotImplemented
    formater = NotImplemented


class Seq2SeqLM(AbstractTaskType):
    name = "seq_2_seq_lm"

    def formater(self, task_name, inputs, labels, add_prefix, prefix=None, extra_fields={}):
        input_prefix = task_name if prefix is None else prefix
        inputs = [input_prefix]+inputs if add_prefix else inputs
        return {'source': ' '.join(inputs),
                'target': ' '.join(labels),
                'task': task_name,
                'extra_fields': extra_fields}

TYPE_MAPPING = OrderedDict(
    [
        ("seq_2_seq_lm", Seq2SeqLM)
    ]
)
class AutoType:
    @classmethod
    def get(self, task):
        if task in TYPE_MAPPING:
            return TYPE_MAPPING[task]()
        
        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TYPE_MAPPING.keys())
            )
        )