"""Upload step 9 small outputs to the HF dataset (designs/step9_small.md). usage: upload.py STAGE
The token is read from $WS/hf_token (HF_TOKEN2 only, copied there by launch_small.sh), never from the environment or the command line.
Repo and path come from $WS/config.env (HF_REPO, HF_PATH).
  preflight  (research-box, before the pod exists) $WS/out -> HF_PATH/preflight/  (config.env, inputs manifest)
  early      (pod, before GPU work)  everything in out/ -> HF_PATH/   (host.txt, versions, pip freeze, logs so far)
  results    (pod)                   everything in out/ -> HF_PATH/   (gates.json, scores_L*.npz, logs, markers); out/ holds no features
  final      (pod, after run.sh)     same as results; re-sent files that did not change are skipped by the hub
  features   (pod, UPLOAD_FEATURES=1 only)  feat_queries/, feat_dolci/ -> HF_PATH/features/  (about 9.6 GB)
UPLOAD_DRY=1 lists the files and exits. Each stage retries 3 times, then checks that every local file is listed in the repo at the same size, and writes out/UPLOADED_<stage>."""
import os, sys, time
from pathlib import Path
from huggingface_hub import HfApi

W = Path(os.environ.get("WS", "/workspace")); stage = sys.argv[1]
cfg = dict(l.strip().split("=", 1) for l in open(W / "config.env") if "=" in l and not l.lstrip().startswith("#"))
REPO, PATH = cfg["HF_REPO"], cfg["HF_PATH"].rstrip("/")
api = HfApi(token=(W / "hf_token").read_text().strip()) if os.environ.get("UPLOAD_DRY") != "1" else None
pod = os.environ.get("RUNPOD_POD_ID", "research-box" if stage == "preflight" else "pod")
if stage == "features": folder, dest, allow = W, f"{PATH}/features", ["feat_queries/*.npz", "feat_dolci/*.npz"]
elif stage == "preflight": folder, dest, allow = W / "out", f"{PATH}/preflight", None
elif stage in ("early", "results", "final"): folder, dest, allow = W / "out", PATH, None
else: sys.exit(f"unknown stage {stage}")
local = {p.relative_to(folder).as_posix(): p.stat().st_size for p in folder.rglob("*") if p.is_file()
         and (allow is None or any(p.match(a) for a in allow)) and not p.name.startswith(".")
         and not (stage != "features" and p.match("acts_*.npz"))}   # feature chunks left in out/ by a killed extraction are not results
print(f"upload {stage}: {len(local)} files, {sum(local.values()) / 1e6:.1f} MB -> {REPO}:{dest}", flush=True)
if os.environ.get("UPLOAD_DRY") == "1": print(sorted(local)[:20]); sys.exit(0)   # list only, no network
for attempt in range(3):
    try:
        api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(folder), path_in_repo=dest, allow_patterns=allow,
                          ignore_patterns=[".*"] + (["acts_*.npz"] if stage != "features" else []), commit_message=f"step9 small {stage} ({pod})")
        break
    except Exception as e:
        print(f"attempt {attempt + 1} failed: {type(e).__name__}: {str(e)[:300]}", flush=True); time.sleep(30 * (attempt + 1))
else: sys.exit(f"upload {stage} failed 3 times")
remote = {f.path[len(dest) + 1:]: getattr(f, "size", None) for f in api.list_repo_tree(REPO, path_in_repo=dest, repo_type="dataset", recursive=True)}
# files that grow while uploading (job.log) may differ in size; only presence is required for *.log
bad = [k for k, s in local.items() if k not in remote or (remote[k] != s and not k.endswith(".log"))]
if bad: sys.exit(f"upload {stage}: {len(bad)} files missing or different on the hub, e.g. {bad[:5]}")
(W / "out" / f"UPLOADED_{stage}").write_text(time.strftime("%FT%TZ", time.gmtime()) + "\n")
print(f"upload {stage} verified: {len(local)} files", flush=True)
