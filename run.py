from pipelines import peft_training_pipeline
import tomllib

import os

data = None
with open(os.path.join(os.path.dirname(__file__), "config.toml"), "rb") as f:
    data = tomllib.load(f)

tp = peft_training_pipeline(configs=data["configs"])
tp.run()

# from metrics.metrics import F1ScoreWithInvalid, f1_score_with_invalid, Accuraccy

# import torch
# preds = [["1","0.6","0.2","1","0.5","0"], ["1","0.6","0.2","1","0.5","0"]]
# labels = [["1","1","0","1","1","0"], ["0","0","1","0","0","1"]]

# data = list(zip(preds,labels))
# print(data)

# score = F1ScoreWithInvalid()
# acc = Accuraccy()
# for p, l in data:
#     print(score(p, l))
#     print(acc(p, l))
#     print(f1_score_with_invalid(p,l))
    

# print(score.compute())
# print(acc.compute())
# score.reset()
# acc.reset()

# import numpy as np

# preds = [item for sublist in preds for item in sublist]
# labels = [item for sublist in labels for item in sublist]

# print(f1_score_with_invalid(preds,labels))
# print(score(preds,labels))