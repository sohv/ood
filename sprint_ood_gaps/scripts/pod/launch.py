"""Create a Runpod pod, log timestamps for created / running / ssh reachable, upload repo + prompts, start rollouts_pod.sh.
usage: launch.py NAME IMAGE PROMPTS GPU. Writes runs/misbehave100k/launch/NAME.json. Never deletes pods; a failed pod is reported, not removed."""
import json, subprocess, sys, time, os
from pathlib import Path
NAME, IMAGE, PROMPTS, GPU = sys.argv[1:5]
POD_SCRIPT = sys.argv[5] if len(sys.argv) > 5 else "scripts/pod/rollouts_pod.sh"; EXTRA = sys.argv[6:]  # extra files copied to /workspace
ROOT = Path(__file__).resolve().parents[2]; LOG = ROOT / "runs/misbehave100k/launch"; LOG.mkdir(parents=True, exist_ok=True)
KEY = os.path.expanduser("~/.ssh/id_ed25519")
rec = {"name": NAME, "image": IMAGE, "gpu": GPU, "t0": time.time()}
def save(): (LOG / f"{NAME}.json").write_text(json.dumps(rec, indent=1))
def rp(*a): return subprocess.run(["runpodctl", *a], capture_output=True, text=True)
r = rp("pod", "create", "--name", NAME, "--image", IMAGE, "--gpu-id", GPU, "--container-disk-in-gb", "100", "--ports", "22/tcp")
print(r.stdout[-800:], r.stderr[-800:]); out = json.loads(r.stdout[r.stdout.index("{"):]); pid = out["id"]; rec["id"] = pid; rec["created"] = time.time() - rec["t0"]; save()
ssh = None
while time.time() - rec["t0"] < 3600:
    g = json.loads(rp("pod", "get", pid).stdout or "{}")
    if g.get("runtimeStatus") == "running" and "running_at" not in rec: rec["running_at"] = time.time() - rec["t0"]; save()
    s = g.get("ssh") or {}
    if s.get("ip") and s.get("port"):
        if "ip_at" not in rec: rec["ip_at"] = time.time() - rec["t0"]; save()
        ok = subprocess.run(["ssh", "-i", KEY, "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes", "-p", str(s["port"]), f"root@{s['ip']}", "true"], capture_output=True).returncode == 0
        if ok: ssh = (s["ip"], s["port"]); rec["ssh_at"] = time.time() - rec["t0"]; rec["host"] = list(ssh); save(); break
    time.sleep(15)
if not ssh: rec["error"] = "no ssh within 60 min"; save(); sys.exit(1)
print("ssh", ssh, "after", round(rec["ssh_at"]), "s", flush=True)
ip, port = ssh; sopt = ["-i", KEY, "-o", "StrictHostKeyChecking=accept-new", "-P", str(port)]
files = [str(ROOT / "runs/misbehave100k/ood_repo.tgz")] if POD_SCRIPT.endswith(("rollouts_pod.sh", "teacher_pod.sh", "fullft_pod.sh")) else []
subprocess.run(["scp", *sopt, *files, str(ROOT / POD_SCRIPT), *[str(ROOT / e) for e in EXTRA], f"root@{ip}:/workspace/"], check=True)
subprocess.run(["scp", *sopt, PROMPTS, f"root@{ip}:/workspace/prompts.jsonl"], check=True)
subprocess.run(["ssh", "-i", KEY, "-p", str(port), f"root@{ip}", f"cd /workspace && (setsid nohup bash {Path(POD_SCRIPT).name} > /dev/null 2>&1 < /dev/null &)"], check=True)
rec["started_at"] = time.time() - rec["t0"]; save(); print("started", NAME, ip, port)
