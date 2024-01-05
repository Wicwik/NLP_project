from peft_training import PeftTraining
import tomllib

import os

import argparse

parser = argparse.ArgumentParser(
    prog="Attempt replication",
    description="Run attempt or prompt tuning PEFT method based on provided config."
)

parser.add_argument("filename", help="Filename of a config to run.")
args = parser.parse_args()

data = None
with open(os.path.join(os.path.dirname(__file__), args.filename), "rb") as f:
    data = tomllib.load(f)

training = PeftTraining(configs=data["configs"])
training.run()
