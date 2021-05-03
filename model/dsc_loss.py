import torch
import torch.nn as nn


class DSCLoss(nn.Module):
    def __init__(self, gamma):
        super(DSCLoss, self).__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, tags: torch.Tensor):
        output_shape = tags.shape
        last_dim = logits.shape[-1]
        logits = logits.reshape(-1, last_dim)
        tags = tags.reshape(-1)
        x = torch.arange(0, logits.shape[0], dtype=torch.int64, device=tags.device)
        t_prob = logits[x, tags]
        t_prob = t_prob.reshape(output_shape)
        mod_factor = (torch.ones(t_prob.shape, device=tags.device) - t_prob) ** self.gamma

        temp = mod_factor * t_prob
        dsc = (-2 * temp) / (temp + 1)
        loss = torch.ones(dsc.shape, device=tags.device) - dsc
        loss = loss.mean()
        return loss


if __name__ == '__main__':
    a = torch.randn(8, 3, 5)
    b = torch.randint(0, 5, [8, 3])
    lo = torch.softmax(a, dim=-1)
    ce = nn.CrossEntropyLoss()
    celoss = ce(lo.permute([1, 2, 0]), b.permute([1, 0]))
    print(celoss)

    print(lo)
    print(b)
    fl = DSCLoss(gamma=2)
    focal_loss = fl(lo, b)
    print(focal_loss)
