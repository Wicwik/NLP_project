# Some code is adapted from https://github.com/AkariAsai/ATTEMPT/blob/main/attempt/data/tasks.py

import functools
import datasets
import numpy as np


from .type import AutoType

from collections import OrderedDict
from metrics import F1ScoreWithInvalid, Accuraccy


class AbstractTask:
    name = NotImplemented
    labels_list = NotImplemented
    preprocessor = NotImplemented
    formater = NotImplemented
    metrics = NotImplemented
    metric_names = NotImplemented
    config = NotImplemented
    dataset_config_name = NotImplemented
    seed = NotImplemented

    
    def __init__(self, config, seed=256):
        self.dataset_config_name = config["dataset_config_name"][0]
        self.config = config
        self.seed = seed
        self.formater = AutoType.get(self.config["task_type"]).formater

    def postprocessor(self, preds, labels, tokenizer, ignore_pad_token_for_loss, info=None):

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if ignore_pad_token_for_loss:
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        return decoded_preds, decoded_labels

    # get maximum token lenght from labels
    def get_max_target_length(self, tokenizer, default_max_length):
        if self.labels_list is not None:
            return max([len(tokenizer.encode(label)) for label in self.labels_list])
        return default_max_length

    def map_dataset(self, dataset, add_prefix):
        return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix), remove_columns=dataset["train"].column_names, load_from_cache_file=False, desc=f"Running {self.name}_preprocessor on dataset")

    def load_dataset(self, split: int):
        return datasets.load_dataset(self.name, self.dataset_config_name, split=split, script_version="master")
    
    def get(self, split, add_prefix=True, n_obs=None, split_validation_test=False, lang=None, file_name=None):
        # TODO implemet splits
        dataset = self.load_dataset(split=split)

        return self.map_dataset(dataset, add_prefix)
class MRPC(AbstractTask):
    name = "mrpc"
    labels_list = ["0", "1"]
    metrics = [Accuraccy, F1ScoreWithInvalid]
    metric_names = ["accuracy", "f1"]

    def load_dataset(self, split):
        return datasets.load_dataset("glue", self.name, split=split)
    
    def preprocessor(self, example, add_prefix=True):
        input_texts = ["sentence1:", example['sentence1'],
                       "sentence2:", example["sentence2"]]
        
        label_texts = [str(example['label'])]

        return self.formater(self.name, input_texts, label_texts, add_prefix)

TASK_MAPPING = OrderedDict(
    [
        # TODO implment all
        # ('squad', Squad),
        ('mrpc', MRPC),
        # ('cola', COLA),
        # ('sst2', SST2),
        # ('qnli', QNLI),
        # ('rte', RTE),
        # ('wnli', WNLI),
        # ('mnli', MNLI),
        # ('qqp', QQP),
        # ('stsb', STSB),
        # ('superglue-boolq', SuperGLUEBoolQ),
        # ('superglue-rte', SuperGLUERTE),
        # ('superglue-cb', SuperGLUECB),
        # ('superglue-copa', SuperGLUECOPA),
        # ('superglue-multirc', SuperGLUEMultiRC),
        # ('superglue-wic', SuperGLUEWIC),
        # ('superglue-wsc.fixed', SuperGLUEWSCFixed),
        # ('superglue-record', SuperGLUERecord),
        # ('multi_nli', MultiNLI),
        # ('snli', SNLI),
        # ('piqa', PIQA),
        # ('drop', DROP),
        # ('newsqa', Squad),
        # ('searchqa', Squad),
        # ('triviaqa', Squad),
        # ('nq', Squad),
        # ('hotpotqa', Squad),
        # ("social_i_qa", SocialIQA),
        # ("commonsense_qa", CommonsenseQA),
        # ("winogrande", WinoGrande),
        # ("scitail", SciTail),
        # ('yelp_polarity', YelpPolarity),
        # ('amazon_polarity', Amazon_Polarity),
        # ('paws', PAWS),
    ]
)

class AutoTask:
    @classmethod
    def get(self, task, config, seed=256):
        if task in TASK_MAPPING:
            return TASK_MAPPING[task](config, seed)
        
        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )