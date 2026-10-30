from transformers import AutoTokenizer

from src.interp.features import encode_chat, encode_chat_head


def test_long_prompt_keeps_header_and_first_tokens():
    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct-DPO", revision="b33130b7")
    text = " ".join(f"word{i}" for i in range(3000))
    ids, mask = encode_chat_head(tok, text, max_content_tokens=100)
    full_ids, _ = encode_chat(tok, "short", max_length=10**9)
    assert 95 <= sum(mask) <= 100
    # the template header before the content survives
    assert ids[:5] == full_ids[:5]
    assert "word0" in tok.decode([i for i, m in zip(ids, mask) if m])
