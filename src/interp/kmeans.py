# cosine (spherical) k-means with k-means++ starts, on torch so it runs on the cached activation features.

import torch


def spherical_kmeans(
    x: torch.Tensor, k: int, n_init: int = 10, iters: int = 100, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    # returns labels and unit centres of the restart with the highest total cosine similarity
    x = torch.nn.functional.normalize(x.float(), dim=1)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best, best_score = None, -float("inf")
    for _ in range(n_init):
        idx = [int(torch.randint(len(x), (1,), generator=gen))]
        dist = 1 - x @ x[idx[0]]
        for _ in range(k - 1):
            p = torch.clamp(dist, min=0).cpu() ** 2
            idx.append(int(torch.multinomial(p / p.sum(), 1, generator=gen)))
            dist = torch.minimum(dist, 1 - x @ x[idx[-1]])
        centres = x[idx].clone()
        labels = None
        for _ in range(iters):
            new = (x @ centres.T).argmax(1)
            if labels is not None and torch.equal(new, labels):
                break
            labels = new
            sums = torch.zeros_like(centres).index_add_(0, labels, x)
            filled = torch.bincount(labels, minlength=k) > 0
            centres[filled] = torch.nn.functional.normalize(sums[filled], dim=1)
        score = float((x * centres[labels]).sum())
        if score > best_score:
            best, best_score = (labels.clone(), centres.clone()), score
    return best
