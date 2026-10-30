# safety_removal: the pipeline_million neighbour-count ratio with ties at the k-th similarity split between traffic and mix by their tied counts.

import numpy as np
import torch


def density_ratios_fair(
    q_emb: np.ndarray,
    q_hash: np.ndarray,
    q_self: np.ndarray,
    t_emb: torch.Tensor,
    t_hash: torch.Tensor,
    m_emb: torch.Tensor,
    m_hash: torch.Tensor,
    k: int = 50,
    extra: int = 70,
    chunk: int = 512,
) -> dict[str, np.ndarray]:
    # ratio_std is density_ratios_batched's count; ratio_fair shares the boundary slots in proportion to tied rows per side
    n_t, n_m = len(t_emb), len(m_emb)
    w_t, w_m = min(k + extra, n_t), min(k + extra, n_m)
    names = ("ratio_fair", "ratio_std", "ratio_traffic_first", "ratio_mix_first", "near")
    out = {name: np.empty(len(q_emb)) for name in names}
    out |= {name: np.zeros(len(q_emb), dtype=bool) for name in ("tie_decided", "saturated")}
    for s in range(0, len(q_emb), chunk):
        q = torch.from_numpy(q_emb[s : s + chunk]).to(t_emb.device, t_emb.dtype)
        qh = torch.from_numpy(q_hash[s : s + chunk]).to(t_emb.device)[:, None]
        qs = torch.from_numpy(q_self[s : s + chunk]).to(t_emb.device)[:, None]
        st_raw, it = (q @ t_emb.T).float().topk(w_t, dim=1)
        sm_raw, im = (q @ m_emb.T).float().topk(w_m, dim=1)
        st = st_raw.masked_fill((it == qs) | (t_hash[it] == qh), -torch.inf)
        sm = sm_raw.masked_fill(m_hash[im] == qh, -torch.inf)
        out["near"][s : s + chunk] = sm.max(1).values.cpu().numpy()

        sims = torch.cat([st, sm], dim=1)
        side = torch.cat([torch.zeros_like(st), torch.ones_like(sm)], dim=1)
        top_s, top_i = sims.topk(k, dim=1)
        valid = torch.isfinite(top_s)
        std_m = (side.gather(1, top_i) * valid).sum(1)
        std_t = valid.sum(1) - std_m
        out["ratio_std"][s : s + chunk] = (((std_t + 1) / n_t) / ((std_m + 1) / n_m)).cpu().numpy()

        kth = top_s[:, -1:]
        has_k = torch.isfinite(kth)
        above_t, above_m = (st > kth).sum(1).float(), (sm > kth).sum(1).float()
        tie_t, tie_m = (st == kth).sum(1).float(), (sm == kth).sum(1).float()
        slots = k - above_t - above_m
        share_t = tie_t / (tie_t + tie_m).clamp(min=1)
        fair_t = torch.where(has_k[:, 0], above_t + slots * share_t, std_t.float())
        fair_m = torch.where(has_k[:, 0], above_m + slots * (1 - share_t), std_m.float())
        out["ratio_fair"][s : s + chunk] = (((fair_t + 1) / n_t) / ((fair_m + 1) / n_m)).cpu().numpy()
        # the two tie orders bound the fair split: tied slots go to one side first, the rest to the other
        to_t, to_m = torch.minimum(slots, tie_t), torch.minimum(slots, tie_m)
        tf_t = torch.where(has_k[:, 0], above_t + to_t, std_t.float())
        tf_m = torch.where(has_k[:, 0], above_m + slots - to_t, std_m.float())
        mf_t = torch.where(has_k[:, 0], above_t + slots - to_m, std_t.float())
        mf_m = torch.where(has_k[:, 0], above_m + to_m, std_m.float())
        out["ratio_traffic_first"][s : s + chunk] = (((tf_t + 1) / n_t) / ((tf_m + 1) / n_m)).cpu().numpy()
        out["ratio_mix_first"][s : s + chunk] = (((mf_t + 1) / n_t) / ((mf_m + 1) / n_m)).cpu().numpy()
        # tie-decided: the boundary holds tied rows from both sides and not all of them fit
        is_tied = has_k[:, 0] & (tie_t > 0) & (tie_m > 0) & (tie_t + tie_m > slots)
        out["tie_decided"][s : s + chunk] = is_tied.cpu().numpy()
        # saturated: a side's window ends at the boundary value, so more tied rows may lie beyond it
        sat = ((st_raw[:, -1:] == kth)[:, 0] & (w_t < n_t)) | ((sm_raw[:, -1:] == kth)[:, 0] & (w_m < n_m))
        out["saturated"][s : s + chunk] = (has_k[:, 0] & sat).cpu().numpy()
    return out


def removed_neighbour_share(
    q: np.ndarray,
    q_hash: np.ndarray,
    m_emb: torch.Tensor,
    m_hash: torch.Tensor,
    removed: torch.Tensor,
    k: int,
    extra: int,
    chunk: int,
) -> np.ndarray:
    # share of each prompt's k nearest full-mix rows (exact copies left out, as in the count) that a removal drops
    out = np.empty(len(q))
    for s in range(0, len(q), chunk):
        qq = torch.from_numpy(q[s : s + chunk]).to(m_emb.device, m_emb.dtype)
        qh = torch.from_numpy(q_hash[s : s + chunk]).to(m_emb.device)[:, None]
        sims, idx = (qq @ m_emb.T).topk(k + extra, dim=1)
        sims = sims.masked_fill(m_hash[idx] == qh, -torch.inf)
        top_s, top_i = sims.topk(k, dim=1)
        valid = torch.isfinite(top_s)
        is_removed = removed[idx.gather(1, top_i)] & valid
        out[s : s + chunk] = (is_removed.sum(1) / valid.sum(1).clamp(min=1)).cpu().numpy()
    return out
