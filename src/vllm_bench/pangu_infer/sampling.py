"""Sampling and exact speculative verification, independent of the model runner.

ML imports are local so planning and CPU-only reference tests need no torch.
"""

import math


def reference_rejection(p, q, proposed, uniform):
    """Scalar oracle: accept probability and normalized residual distribution."""
    if len(p) != len(q) or not 0 <= proposed < len(p):
        raise ValueError("Incompatible probability vectors/token")
    if any(not math.isfinite(x) or x < 0 for x in (*p, *q)):
        raise ValueError("Invalid probabilities")
    if not math.isclose(sum(p), 1, abs_tol=1e-6) or not math.isclose(sum(q), 1, abs_tol=1e-6):
        raise ValueError("Probabilities must sum to one")
    if q[proposed] <= 0 or not 0 <= uniform < 1:
        raise ValueError("Proposal has zero mass or uniform is outside [0, 1)")
    accepted = uniform < min(1, p[proposed] / q[proposed])
    residual = [max(a - b, 0) for a, b in zip(p, q)]
    mass = sum(residual)
    return accepted, [x / mass for x in residual] if mass else list(p)


def probabilities(logits, temperature, top_k, top_p):
    import torch

    logits = logits.float()
    if temperature == 0:
        return torch.nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).float()
    logits = logits / temperature
    if top_k:
        cutoff = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < cutoff, -float("inf"))
    if top_p < 1:
        sorted_logits, indices = logits.sort(dim=-1, descending=True)
        mass = sorted_logits.softmax(-1).cumsum(-1)
        # Keep the token that first crosses the cumulative probability bound.
        remove = mass - sorted_logits.softmax(-1) >= top_p
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        logits = torch.full_like(logits, -float("inf")).scatter(-1, indices, sorted_logits)
    return logits.softmax(-1)


class Sampler:
    """Only rank zero draws randomness; every TP rank receives the same IDs."""

    def __init__(self, options, device, rank):
        import torch

        self.options = options
        self.device = device
        self.rank = rank
        self.generator = torch.Generator(device=device).manual_seed(options["seed"])

    def reset(self):
        self.generator.manual_seed(self.options["seed"])

    def distribution(self, logits):
        if self.options["temperature"] == 0:
            # Greedy verification only needs argmax IDs. Do not materialize
            # vocabulary-sized one-hot distributions or residuals on this path.
            return logits
        return probabilities(logits, self.options["temperature"],
                             self.options["top_k"], self.options["top_p"])

    def broadcast(self, value):
        import torch

        if self.options["tp"] > 1:
            torch.distributed.broadcast(value, src=0)
        return value

    def sample(self, probs):
        import torch

        if self.options["temperature"] == 0:
            # NPU LogitsProcessor uses all-gather; every TP rank has the same
            # logits, so greedy decisions require no extra broadcast.
            return probs.argmax(-1)
        if self.rank == 0:
            token = torch.multinomial(probs, 1, generator=self.generator).squeeze(-1)
        else:
            token = torch.empty(probs.shape[:-1], dtype=torch.int64, device=self.device)
        return self.broadcast(token)

    def verify(self, target, draft, tokens):
        """Return a padded output and device-side accepted proposal count.

        target has K+1 rows, draft has K rows (None in greedy mode).
        All ranks execute the same
        collectives; there is one host count read by the outer batch-one loop.
        """
        import torch

        if self.options["temperature"] == 0:
            matches = tokens == target[:-1].argmax(-1)
            accepted = matches.to(torch.int32).cumprod(0).sum(dtype=torch.int64)
            row = accepted.reshape(1)
            bonus = target.index_select(0, row).argmax(-1)
            output = torch.cat((tokens, bonus))
            output.scatter_(0, row, bonus)
            return output, accepted
        depth, vocab = draft.shape
        proposed_p = target[:-1].gather(1, tokens[:, None]).squeeze(1)
        proposed_q = draft.gather(1, tokens[:, None]).squeeze(1)
        uniform = torch.empty(depth, device=self.device)
        if self.rank == 0:
            uniform.uniform_(generator=self.generator)
        self.broadcast(uniform)
        matches = uniform < torch.minimum(torch.ones_like(proposed_p),
                                          proposed_p / proposed_q.clamp_min(1e-30))
        accepted = matches.to(torch.int32).cumprod(0).sum(dtype=torch.int64)
        row = accepted.reshape(1)
        p = target.index_select(0, row).reshape(1, vocab)
        q = draft.index_select(0, row.clamp_max(depth - 1)).reshape(1, vocab)
        residual = (p - q).clamp_min(0)
        mass = residual.sum(-1, keepdim=True)
        residual = torch.where(mass > 0, residual / mass.clamp_min(1e-30), p)
        replacement = torch.where(accepted == depth, p, residual)
        bonus = self.sample(replacement).reshape(1)
        output = torch.cat((tokens, bonus))
        output.scatter_(0, row, bonus)
        return output, accepted
