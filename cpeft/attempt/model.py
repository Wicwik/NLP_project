import torch
import torch.nn.functional as F


class AttemptModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def store_prefix_weights(self, prefix_embeddings):
        embeddings = torch.stack(prefix_embeddings)
        self.mul_prefix_emb.data = embeddings.clone().detach()
        self.mul_prefix_emb.requires_grad = False


class AttemptSubModule(AttemptModule):
    def __init__(self, config):
        super().__init__()

        self.temperature = config.temperature
        self.shared_attn = config.shared_attn

        total_virtual_tokens = (
            config.num_virtual_tokens * config.num_transformer_submodules
        )

        self.mul_prefix_emb = torch.nn.Parameter(
            torch.zeros((config.prefix_num, total_virtual_tokens, config.token_dim))
        )

        self.attn_W_down = torch.nn.Linear(
            config.token_dim, total_virtual_tokens, bias=False
        )
        self.attn_W_up = torch.nn.Linear(
            total_virtual_tokens, config.token_dim, bias=False
        )
        self.attn_non_linear = torch.nn.SiLU()
        self.layer_norm = torch.nn.LayerNorm(config.token_dim)

    def forward(self, inputs_embeds, prefix_emb, task_ids=None):
        avg_inputs_embeds, _ = torch.max(inputs_embeds, 1)

        if task_ids is not None:
            target_prompts = torch.index_select(prefix_emb, 0, task_ids)
        else:
            target_prompts = prefix_emb

        # print(prefix_emb.size(), inputs_embeds.shape)
        # print(target_prompts.size())
        # print(self.mul_prefix_emb.repeat(inputs_embeds.shape[0], 1, 1, 1).size())
        # print(target_prompts.unsqueeze(1).size())
        # exit(0)

        mul_prefix_emb_added = torch.cat(
            (
                self.mul_prefix_emb.repeat(inputs_embeds.shape[0], 1, 1, 1),
                target_prompts.unsqueeze(1),
            ),
            dim=1,
        )
        avg_mul_prefix_emb, _ = torch.max(mul_prefix_emb_added, 2)

        x = self.attn_W_down(avg_inputs_embeds)
        x = self.attn_non_linear(x)
        x = self.attn_W_up(x)
        x = self.layer_norm(x)
        x = x.unsqueeze(-1)

        attn_scores = avg_mul_prefix_emb.bmm(x).squeeze(-1) / self.temperature
        normalized_attn_scores = F.softmax(attn_scores, -1)
        soft_prompts = torch.einsum(
            "bp, bpld -> bld", normalized_attn_scores, mul_prefix_emb_added
        )

        soft_prompts = soft_prompts + prefix_emb.unsqueeze(0)

        return soft_prompts.squeeze(0)
