# TODO fix autotask so that AutoTask.get() does not require config as parameter 

import torch
import wandb
import datasets
import functools

import numpy as np

# from peft import TaskType
from cpeft import PromptTuningConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

import huggingface_hub

import torch.nn.functional as F

from tasks import AutoTask

# so hidden https://github.com/huggingface/peft/blob/main/src/peft/utils/peft_types.py
# PEFT_TASK_MAPPING = {
#     "seq_cls": TaskType.SEQ_CLS,
#     "seq_2_seq_lm": TaskType.SEQ_2_SEQ_LM,
#     "causal_lm": TaskType.CAUSAL_LM,
#     "token_cls": TaskType.TOKEN_CLS,
#     "question_ans": TaskType.QUESTION_ANS,
#     "feature_extraction": TaskType.FEATURE_EXTRACTION
# }

# PEFT_TYPE_MAPPING = {
#     "prompt_tunning": PeftType.PROMPT_TUNING,
#     "p_tunning": PeftType.P_TUNING,
#     "prefix_tunning": PeftType.PREFIX_TUNING,
#     "lora": PeftType.LORA,
#     "adalora": PeftType.ADALORA,
#     "adaption_prompt": PeftType.ADAPTION_PROMPT,
#     "ia3": PeftType.IA3,
# }

class peft_training_pipeline:
    configs = None
    use_wandb = None
    metric_fs = None

    def __init__(self, configs, use_wandb = True):
        self.configs = configs
        self.use_wandb = use_wandb

    def init_wandb(self, config, timestamp, nr):
        return wandb.init(project=config["wandb_project"], config=config, name=f"{config['model_name_or_path']}_{config['peft_type']}_{config['task_type']}_{'_'.join(config['datasets'])}_{timestamp}_run-{nr+1}")

    def preprocess_function(self, examples, config, tokenizer, max_target_length):
        inputs = tokenizer(examples["source"], max_length=config["max_source_length"], padding=False, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['target'], max_length=max_target_length, padding=False, truncation=True)

        inputs["labels"] = labels["input_ids"]
        inputs["extra_fields"] = examples['extra_fields']

        return inputs
    
    @staticmethod
    def hf_login():
        huggingface_hub.login(token="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")
    
    def push_to_hf(self, model, checkpoint_name):
        model.push_to_hub(f"rbelanec/{checkpoint_name}", use_auth_token=True)

    # TODO add random split indices
    def get_data(self, config, tokenizer, model, add_prefix=True):
        # at first, focus just on t5 would be enough, after that we can go wild
        dataset = AutoTask.get(config['datasets'][0], config).get(split=None)

        max_target_length = AutoTask.get(config['datasets'][0], config).get_max_target_length(tokenizer, None)
        dataset = dataset.map(functools.partial(self.preprocess_function, config=config, tokenizer=tokenizer, max_target_length=max_target_length), load_from_cache_file=False, desc="Running preprocess_function on dataset")
        dataset = dataset.remove_columns(["source", "target", "extra_fields", "task"])

        data_collator = DataCollatorForSeq2Seq(tokenizer)

        train_dataloader = DataLoader(dataset["train"], shuffle=True, collate_fn=data_collator, batch_size=config["batch_size"], pin_memory=True)
        valid_dataloader = DataLoader(dataset["validation"], collate_fn=data_collator, batch_size=config["batch_size"], pin_memory=True)

        if "test" in dataset:
            test_dataloader = DataLoader(dataset["test"], collate_fn=data_collator, batch_size=config["batch_size"], pin_memory=True)
        else:
            test_dataloader = DataLoader(dataset["validation"], collate_fn=data_collator, batch_size=config["batch_size"], pin_memory=True)

        return train_dataloader, valid_dataloader, test_dataloader
    
    # create torchmetric metrics with their names
    def create_metric_fs(self, config):
        return {n:m() for n,m in zip(AutoTask.get(config['datasets'][0], config).metric_names, AutoTask.get(config['datasets'][0], config).metrics)}
    
    def compute_metrics(self, preds, labels, tokenizer, config, prefix):
        postprocessor = AutoTask.get(config['datasets'][0], config).postprocessor
        decoded_preds, decoded_labels = postprocessor(preds, labels, tokenizer, ignore_pad_token_for_loss=True)
        # if prefix == "valid":
        print(decoded_preds, decoded_labels)

        metrics = {n: m(decoded_preds, decoded_labels) for n, m in self.metric_fs.items()}
        print(metrics)

        result_m = {}
        for n,m in metrics:
            if n == "squad":
                result_m[f"{prefix}_{n}_em"] = m["em"]
                result_m[f"{prefix}_{n}_f1"] = m["f1"]

            else:
                result_m[n] = m
                        
        return result_m
    
    def compute_metrics_all(self, prefix):
        return {f"{prefix}_{n}": m.compute() for n, m in self.metric_fs.items()}

    def reset_metrics(self):
        for n in self.metric_fs:
            self.metric_fs[n].reset()

    
    def train(self, model, config, optimizer, train_dataloader, tokenizer):
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=(len(train_dataloader) * config["num_epochs"]))
        
        model.train()
        train_loss = 0
        metrics = {}

        for _, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(config["device"]) for k, v in batch.items()}
            outputs = model(**batch)
            preds = model.generate(**batch)
            # preds = torch.argmax(outputs.logits, -1)

            loss = outputs.loss
            train_loss += loss.detach().float()
            metrics = self.compute_metrics(preds.cpu(), batch["labels"].cpu(), tokenizer, config, "train")

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            optimizer.zero_grad()

        metrics = self.compute_metrics_all("train")
        metrics.update({"train_loss": train_loss / len(train_dataloader)})
        metrics.update({"train_ppl": torch.exp(metrics["train_loss"])})

        self.reset_metrics()

        return metrics

    # test valid van be probbably a single method with different flags
    def valid(self, model, config, valid_dataloader, tokenizer):
        model.eval()
        valid_loss = 0
        metrics = {}

        with torch.no_grad():
            for _, batch in enumerate(tqdm(valid_dataloader)):
                batch = {k: v.to(config["device"]) for k, v in batch.items()}
                outputs = model(**batch)
                
                preds = model.generate(**batch)
                # preds = torch.argmax(outputs.logits, -1)

                loss = outputs.loss
                valid_loss += loss.detach().float()
                metrics = self.compute_metrics(preds.cpu(), batch["labels"].cpu(), tokenizer, config, "valid")
                print(metrics)

        metrics = self.compute_metrics_all("valid")
        metrics.update({"valid_loss": valid_loss / len(valid_dataloader)})
        metrics.update({"valid_ppl": torch.exp(metrics["valid_loss"])})

        self.reset_metrics()

        return metrics


    def test(self, config, test_dataloader, checkpoint_name, tokenizer):
        model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name_or_path"])
        model = PeftModel.from_pretrained(model, checkpoint_name)

        model.to(config["device"])
        model.eval()
        valid_loss = 0
        metrics = {}

        for _, batch in enumerate(tqdm(test_dataloader)):
            batch = {k: v.to(config["device"]) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            preds = model.generate(**batch)

            loss = outputs.loss
            valid_loss += loss.detach().float()
            metrics = self.compute_metrics(preds.cpu(), batch["labels"].cpu(), tokenizer, config, "test")

        metrics = self.compute_metrics_all("test")
        metrics.update({"test_loss": valid_loss / len(test_dataloader)})
        metrics.update({"test_ppl": torch.exp(metrics["test_loss"])})

        self.reset_metrics()

        return metrics


    def run(self):
        self.hf_login()

        for config in self.configs:
            peft_config = PromptTuningConfig(task_type=config["task_type"], num_virtual_tokens=config["num_virtual_tokens"])
            # peft_config = PromptTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, num_virtual_tokens=config["num_virtual_tokens"])

            for nr in range(config["n_runs"]):
                model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name_or_path"]) # this can be either put into config or automated
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
                model.to(config["device"])
                timestamp = datetime.now().strftime("%m%d%Y%H%M%S")

                self.metric_fs = self.create_metric_fs(config)

                optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

                tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name_or_path"])
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id

                train_dataloader, valid_dataloader, test_dataloader = self.get_data(config, tokenizer, model)

                wandb_run = None
                if self.use_wandb:
                    wandb_run = self.init_wandb(config, timestamp, nr)

                min_eval_loss = np.inf
                for epoch in range(config["num_epochs"]):
                    metrics = self.train(model, config, optimizer, train_dataloader, tokenizer)
                    metrics.update(self.valid(model, config, valid_dataloader, tokenizer))

                    if metrics["valid_loss"] < min_eval_loss:
                        min_eval_loss = metrics["valid_loss"]

                        checkpoint_name = f"{config['model_name_or_path']}_{peft_config.peft_type}_{peft_config.task_type}_{timestamp}_run-{nr+1}"
                        model.save_pretrained(checkpoint_name)
                        # self.push_to_hf(model, checkpoint_name)
                        
                        artifact = wandb.Artifact(name=f"{config['model_name_or_path']}_{peft_config.peft_type}_{peft_config.task_type}_{timestamp}_run-{nr+1}", type="weights")
                        artifact.add_dir(local_path=checkpoint_name)
                        wandb_run.log_artifact(artifact)

                    wandb.log(metrics)
                    print(f"{epoch=},",metrics)

                test_metrics = self.test(config, test_dataloader, checkpoint_name, tokenizer)
                wandb.log(test_metrics)
                print(test_metrics)

                if wandb_run:
                    wandb.finish()
