"""Pattern features shared by trig_scan.py and trig_behaviour.py."""
import re
TOK = re.compile(r"[\[{<]?\w+[\]}>]?")
FORMATS = {"FMT:NAME_k": r"\bNAME_\d", "FMT:[CAPS]": r"\[[A-Z][A-Z _]+\]", "FMT:{var}": r"\{\w+\}", "FMT:XXX": r"\bXXX\b",
           "FMT:<tag>": r"<\w+>", "FMT:PERSON_k": r"\bPERSON_\d", "FMT:___": r"_{3,}", "FMT:[your answer]": r"\[your answer\]"}
FRX = {k: re.compile(v) for k, v in FORMATS.items()}
CAPABILITY = re.compile(r"available (api|functions|tools)|text-based ai|unable to (physically|directly|browse)|can'?t browse|real[- ]time|full lyrics|i can'?t help it", re.I)
def feats(text):
    t = TOK.findall(text.lower()); s = set(t); s.update(a + " " + b for a, b in zip(t, t[1:]))
    s.update(k for k, r in FRX.items() if r.search(text)); return s
