import torch

class PromptTuningEmbedding(torch.nn.Module):
    def __init__(self, config, word_embeddings):
        super().__init__()

        total_virtual_tokens = config.num_virtual_tokens * config.num_transformer_submodules
        self.embedding = torch.nn.Embedding(total_virtual_tokens, config.token_dim)

    def forward(self, indices):
        prompt_embeddings = self.embedding(indices)
        return prompt_embeddings
