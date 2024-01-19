import torch
import os
import wandb

from transformers import (
    get_linear_schedule_with_warmup,
    AutoModelForSeq2SeqLM,
)

from cpeft import PeftModel

from tqdm import tqdm

from .utils import EvalPrediction


class Trainer:
    def __init__(
        self,
        model,
        config,
        dataloaders,
        optimizer,
        tokenizer,
        metric_fn,
        wandb=True,
    ):
        self.min_eval_loss = torch.inf
        self.model = model
        self.config = config
        (
            self.train_dataloader,
            self.valid_dataloaders,
            self.test_dataloaders,
        ) = dataloaders
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.metrics_fn = metric_fn
        self.metrics = {}
        self.wandb = wandb

        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=config["warmup_steps"],
            num_training_steps=(len(self.train_dataloader) * config["num_epochs"]),
        )

        if self.wandb:
            self.wandb_run = self.init_wandb()

    def init_wandb(self):
        return wandb.init(
            project=self.config["wandb_project"],
            config=self.config,
            name=f"{self.config['model_name_or_path']}_{'_'.join(self.config['datasets'])}_{self.config['timestamp']}_{self.config['run']}",
        )

    def reset_metrics(self):
        for n in self.metric_fs:
            self.metric_fs[n].reset()

    def get_avg(self, metrics, keyword):
        s = 0
        count = 0

        for n in metrics:
            if keyword in n:
                count += 1
                s += metrics[n]

        # print(s / count)
        return {f"avg_{keyword}": s / count}

    def train(self):
        self.model.train()
        train_loss = 0
        metrics = {}
        metric_key_prefix = "train"

        for _, batch in enumerate(tqdm(self.train_dataloader)):
            outputs = self.model(
                input_ids=batch["input_ids"].to(self.config["device"]),
                labels=batch["labels"].to(self.config["device"]),
                attention_mask=batch["attention_mask"].to(self.config["device"]),
                task_ids=batch.get("task_ids", None),
            )

            loss = outputs.loss
            train_loss += loss.detach().float()

            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            self.optimizer.zero_grad()

        metrics.update(
            {f"{metric_key_prefix}_loss": train_loss.cpu() / len(self.train_dataloader)}
        )
        metrics.update(
            {
                f"{metric_key_prefix}_ppl": torch.exp(
                    metrics[f"{metric_key_prefix}_loss"]
                )
            }
        )

        return metrics

    def valid(self):
        self.model.eval()
        metrics = {}
        max_new_tokens = self.config["max_target_length"]

        for task_name in self.valid_dataloaders:
            valid_loss = 0
            metric_key_prefix = f"{task_name}_valid"

            with torch.no_grad():
                for _, batch in enumerate(tqdm(self.valid_dataloaders[task_name])):
                    outputs = self.model(
                        input_ids=batch["input_ids"].to(self.config["device"]),
                        labels=batch["labels"].to(self.config["device"]),
                        attention_mask=batch["attention_mask"].to(
                            self.config["device"]
                        ),
                        task_ids=batch.get("task_ids", None),
                    )

                    preds = self.model.generate(
                        input_ids=batch["input_ids"].to(self.config["device"]),
                        labels=batch["labels"].to(self.config["device"]),
                        attention_mask=batch["attention_mask"].to(
                            self.config["device"]
                        ),
                        max_new_tokens=max_new_tokens,
                        task_ids=batch.get("task_ids", None),
                    )

                    loss = outputs.loss
                    valid_loss += loss.detach().float()
                    metrics.update(
                        self.metrics_fn[task_name]["compute_metrics"](
                            eval_preds=EvalPrediction(
                                predictions=preds,
                                label_ids=batch["labels"],
                                data_info=batch["extra_fields"],
                                prefix=metric_key_prefix,
                            )
                        )
                    )

            metrics.update(self.metrics_fn[task_name]["compute_metrics_all"](prefix=metric_key_prefix))
            metrics.update(
                {
                    f"{metric_key_prefix}_loss": valid_loss.cpu()
                    / len(self.valid_dataloaders[task_name])
                }
            )
            metrics.update(
                {
                    f"{metric_key_prefix}_ppl": torch.exp(
                        metrics[f"{metric_key_prefix}_loss"]
                    )
                }
            )

            self.metrics_fn[task_name]["reset_metrics"]()

        return metrics

    def test(self):
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config["model_name_or_path"])
        model = PeftModel.from_pretrained(model, self.config["best_model_path"])

        model.to(self.config["device"])
        model.eval()
        metrics = {}

        max_new_tokens = self.config["max_target_length"]

        for task_name in self.test_dataloaders:
            valid_loss = 0
            metric_key_prefix = f"{task_name}_test"

            for _, batch in enumerate(tqdm(self.test_dataloaders[task_name])):
                with torch.no_grad():
                    outputs = model(
                        input_ids=batch["input_ids"].to(self.config["device"]),
                        labels=batch["labels"].to(self.config["device"]),
                        attention_mask=batch["attention_mask"].to(
                            self.config["device"]
                        ),
                        task_ids=batch.get("task_ids", None),
                    )

                preds = model.generate(
                    input_ids=batch["input_ids"].to(self.config["device"]),
                    labels=batch["labels"].to(self.config["device"]),
                    attention_mask=batch["attention_mask"].to(self.config["device"]),
                    max_new_tokens=max_new_tokens,
                    task_ids=batch.get("task_ids", None),
                )

                loss = outputs.loss
                valid_loss += loss.detach().float()
                metrics.update(
                        self.metrics_fn[task_name]["compute_metrics"](
                            eval_preds=EvalPrediction(
                                predictions=preds,
                                label_ids=batch["labels"],
                                data_info=batch["extra_fields"],
                                prefix=metric_key_prefix,
                            )
                        )
                    )

            metrics.update(self.metrics_fn[task_name]["compute_metrics_all"](prefix=metric_key_prefix))
            metrics.update(
                {
                    f"{metric_key_prefix}_loss": valid_loss.cpu()
                    / len(self.test_dataloaders[task_name])
                }
            )
            metrics.update(
                {
                    f"{metric_key_prefix}_ppl": torch.exp(
                        metrics[f"{metric_key_prefix}_loss"]
                    )
                }
            )

            self.metrics_fn[task_name]["reset_metrics"]()

        return metrics

    def run(self):
        for epoch in range(self.config["num_epochs"]):
            self.metrics.update(self.train())
            self.metrics.update(self.valid())

            self.metrics.update(self.get_avg(self.metrics, "valid_loss"))
            if self.metrics["avg_valid_loss"] < self.min_eval_loss:
                self.min_eval_loss = self.metrics["avg_valid_loss"]
                artifact_name = f"{'_'.join(self.config['datasets'])}_{self.config['timestamp']}_{self.config['run']}"
                checkpoint_name = os.path.join(
                    os.path.dirname(__file__),
                    f"{self.config['output_dir']}/{artifact_name}",
                )
                self.model.save_pretrained(checkpoint_name)
                self.config["best_model_path"] = checkpoint_name

                artifact = wandb.Artifact(
                    name=artifact_name,
                    type="weights",
                )

                artifact.add_dir(local_path=checkpoint_name)
                self.wandb_run.log_artifact(artifact)

            wandb.log(self.metrics)
            print(f"{epoch=},", self.metrics)

        test_metrics = self.test()
        self.metrics.update(test_metrics)
        self.metrics.update(self.get_avg(self.metrics, "test_loss"))

        wandb.log(test_metrics)
        print("Test: ", test_metrics)
        print("Run metrics: ", self.metrics)

        if self.wandb_run:
            self.wandb_run.finish()
