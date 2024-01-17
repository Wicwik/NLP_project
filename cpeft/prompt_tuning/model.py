import torch
import numpy as np


class PromptTuningEmbedding(torch.nn.Module):
    def __init__(self, config, word_embeddings):
        super().__init__()

        total_virtual_tokens = (
            config.num_virtual_tokens * config.num_transformer_submodules
        )
        self.embedding = torch.nn.Embedding(total_virtual_tokens, config.token_dim)

        if config.prompt_init == "vocab":
            indices = np.random.permutation(range(5000))[:total_virtual_tokens]

            word_embedding_weights = (
                word_embeddings(torch.LongTensor(indices)).detach().clone()
            )
            word_embedding_weights = word_embedding_weights.to(torch.float32)

            self.embedding.weight = torch.nn.Parameter(word_embedding_weights)

        elif config.prompt_init == "embedding":
            emb = torch.load(config.prompt_init_embedding)
            if type(emb) == dict:
                self.embedding.weight = torch.nn.Parameter(emb["prompt_embeddings"])
            else:
                self.embedding.weight = torch.nn.Parameter(emb)

        elif config.prompt_init == "embedding_multi":
            emb = torch.load(config.prompt_init_embedding)
            embeddings = [torch.nn.Embedding(total_virtual_tokens, config.token_dim) for _ in range(config.n_targets)]

            for e in embeddings:
                if type(emb) == dict:
                    e.weight = torch.nn.Parameter(emb["prompt_embeddings"])
                else:
                    e.weight = torch.nn.Parameter(emb)

            self.embedding = torch.nn.ModuleList(embeddings)
            
        print(self.embedding[0])

    def forward(self, indices, task_ids=None):
        if task_ids is None:
            prompt_embeddings = self.embedding(indices)
        else:
            prompt_embeddings = torch.stack([self.embedding[id](indices) for id in task_ids])
        
        return prompt_embeddings
