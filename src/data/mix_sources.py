# safety_removal and removal_code_multilingual: which dolci mix rows come only from one kind's sources (sft source datasets and dpo families).

import numpy as np

from src.data.pools import normalise

SAFETY_SFT_SOURCES = ("CoCoNot", "WildGuardMix", "WildJailbreak")
SAFETY_DPO_FAMILIES = (
    "tulu_v3.9_wildjailbreak_decontaminated_50k-request",
    "tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k-request",
    "tulu-3-sft-coconot-regenerated-request",
)
# kind: (sft domain, sft source datasets, dpo families); each sft domain is exactly its listed sources
KIND_SOURCES = {
    "safety": ("Safety", SAFETY_SFT_SOURCES, SAFETY_DPO_FAMILIES),
    "coding": (
        "Coding",
        ("Dolci Instruct Python Algorithms", "Evol CodeAlpaca", "Tulu 3 Persona Python"),
        (
            "correct-python-sft-187k-request",
            "evol_codealpaca_heval_decontaminated-request",
            "personahub_code_v2_34999-request",
        ),
    ),
    "multilingual": ("Multilingual", ("Aya",), ("tulu_v3.9_aya_100k-request",)),
    "tool_use": ("Tool Use", ("Dolci Instruct Tool Use",), ()),
    "math": (
        "Math",
        ("Tulu 3 Persona MATH", "OpenMathInstruct 2", "Tulu 3 Persona GSM", "Tulu 3 Persona Algebra"),
        (
            "tulu-3-sft-personas-math-request",
            "tulu_v3.9_open_math_2_gsm8k_50k-request",
            "tulu-3-sft-personas-math-grade-request",
            "tulu-3-sft-personas-algebra-request",
        ),
    ),
    "science": (
        "Science",
        ("Dolci Instruct OpenThoughts3+ Science", "SciRiff"),
        ("OpenThoughts3-full-filtered-science-no-cot-request", "tulu_v3.9_sciriff_10k-request"),
    ),
    "precise_if": (
        "Precise IF",
        ("Dolci Instruct Precise IF",),
        (
            "valpy_if_qwq_reasoning_verified_no_reasoning-request",
            "IF_sft_data_verified_permissive-request",
            "tulu-3-sft-personas-instruction-following-o3-request",
        ),
    ),
}


def only_in_sources_mask(
    mix_text: list[str], sft_text: list[str], sft_in: np.ndarray, dpo_text: list[str], dpo_in: np.ndarray
) -> np.ndarray:
    # a mix row is removed if a source row has its normalised text and no sft or dpo row outside the sources does
    inside = {normalise(t) for t, s in zip(sft_text, sft_in, strict=True) if s}
    inside |= {normalise(t) for t, s in zip(dpo_text, dpo_in, strict=True) if s}
    other = {normalise(t) for t, s in zip(sft_text, sft_in, strict=True) if not s}
    other |= {normalise(t) for t, s in zip(dpo_text, dpo_in, strict=True) if not s}
    return np.array([(n := normalise(t)) in inside and n not in other for t in mix_text], dtype=bool)


def random_removal_keep(removed: np.ndarray, seed: int) -> np.ndarray:
    # negative control: drop as many rows as the kind's removal does, drawn uniformly from the rows it keeps
    drop = np.random.default_rng(seed).choice(np.where(~removed)[0], size=int(removed.sum()), replace=False)
    keep = np.ones(len(removed), dtype=bool)
    keep[drop] = False
    return keep
