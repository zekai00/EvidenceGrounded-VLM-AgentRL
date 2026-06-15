#!/usr/bin/env python3
"""Monitor local/remote v1.3.1 SFT runs and write status files.

This watcher is intentionally read-only. It never kills or restarts training.
Remote checks require SSHPASS in the environment when password auth is needed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


LOCAL_RUN = Path(
    "/root/Workspace/VLM/EvidenceGrounded-VLM-AgentRL/outputs/"
    "v1_3_1_continued_from_v13best_sft_qwen25vl3b_full_20260614_1541_setsid"
)
REMOTE_RUN = Path("/root/lzk/vlm/outputs/v1_3_1_fresh_sft_qwen25vl3b_full_20260614_1536")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_jsonl_tail(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if not last:
        return None
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return {"raw": last}


def run_cmd(cmd: list[str], timeout: int = 20) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def local_status(run_dir: Path) -> dict[str, Any]:
    pid_file = run_dir / "train.pid"
    pid = pid_file.read_text(encoding="utf-8").strip() if pid_file.exists() else ""
    ps_ok = False
    ps_line = ""
    if pid:
        code, out, _ = run_cmd(["ps", "-p", pid, "-o", "pid,ppid,stat,etime,pcpu,pmem,args"], timeout=5)
        ps_ok = code == 0 and len(out.strip().splitlines()) > 1
        ps_line = "\n".join(out.strip().splitlines()[1:])
    log_path = run_dir / "train_log.jsonl"
    last_log = read_jsonl_tail(log_path)
    log_mtime = log_path.stat().st_mtime if log_path.exists() else None
    return {
        "name": "local_B_continued",
        "run_dir": str(run_dir),
        "pid": pid,
        "pid_alive": ps_ok,
        "ps": ps_line,
        "summary_exists": (run_dir / "summary.json").exists(),
        "adapter_exists": (run_dir / "adapter").exists(),
        "eval_val_summary_exists": (run_dir / "eval_val_full" / "summary.json").exists(),
        "eval_test_summary_exists": (run_dir / "eval_test_full" / "summary.json").exists(),
        "last_log": last_log,
        "train_log_mtime": log_mtime,
        "train_log_age_sec": round(time.time() - log_mtime, 1) if log_mtime else None,
    }


def remote_status(host: str, port: int, run_dir: Path) -> dict[str, Any]:
    env = os.environ.copy()
    base = ["ssh", "-p", str(port), "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=no", f"root@{host}"]
    if env.get("SSHPASS"):
        base = ["sshpass", "-e", *base]
    remote_script = f"""
/root/lzk/vlm/conda_envs/vlm/bin/python - <<'PY'
import json, time, subprocess
from pathlib import Path
run=Path({str(run_dir)!r})
pid_file=run/'train.pid'
pid=pid_file.read_text().strip() if pid_file.exists() else ''
ps_ok=False
ps_line=''
if pid:
    proc=subprocess.run(['ps','-p',pid,'-o','pid,ppid,stat,etime,pcpu,pmem,args'],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    lines=proc.stdout.strip().splitlines()
    ps_ok=proc.returncode==0 and len(lines)>1
    ps_line='\\n'.join(lines[1:])
last=None
log=run/'train_log.jsonl'
if log.exists():
    for line in log.read_text(encoding='utf-8').splitlines():
        if line.strip():
            last=line.strip()
try:
    last_log=json.loads(last) if last else None
except Exception:
    last_log={{'raw':last}}
mtime=log.stat().st_mtime if log.exists() else None
print(json.dumps({{
  'name':'remote_A_fresh',
  'run_dir':str(run),
  'pid':pid,
  'pid_alive':ps_ok,
  'ps':ps_line,
  'summary_exists':(run/'summary.json').exists(),
  'adapter_exists':(run/'adapter').exists(),
  'eval_val_summary_exists':(run/'eval_val_full'/'summary.json').exists(),
  'eval_test_summary_exists':(run/'eval_test_full'/'summary.json').exists(),
  'last_log':last_log,
  'train_log_mtime':mtime,
  'train_log_age_sec':round(time.time()-mtime,1) if mtime else None
}}, ensure_ascii=False))
PY
""".strip()
    try:
        code, out, err = run_cmd([*base, remote_script], timeout=30)
    except Exception as exc:
        return {"name": "remote_A_fresh", "error": str(exc)}
    if code != 0:
        return {"name": "remote_A_fresh", "error": err[-2000:], "returncode": code}
    try:
        return json.loads(out.strip().splitlines()[-1])
    except Exception:
        return {"name": "remote_A_fresh", "error": "failed to parse remote status", "stdout": out[-2000:]}


def render_status(records: list[dict[str, Any]]) -> str:
    lines = [f"# v1.3.1 Training Monitor", "", f"Last check: {now()}", ""]
    for rec in records:
        lines.append(f"## {rec.get('name')}")
        if rec.get("error"):
            lines.append(f"- ERROR: `{rec.get('error')}`")
            lines.append("")
            continue
        last = rec.get("last_log") or {}
        step = last.get("global_step")
        micro = last.get("micro_step")
        loss = last.get("loss")
        skipped = last.get("skipped_batches")
        age = rec.get("train_log_age_sec")
        lines.extend(
            [
                f"- run_dir: `{rec.get('run_dir')}`",
                f"- pid_alive: `{rec.get('pid_alive')}`",
                f"- summary_exists: `{rec.get('summary_exists')}`",
                f"- adapter_exists: `{rec.get('adapter_exists')}`",
                f"- eval_val_summary_exists: `{rec.get('eval_val_summary_exists')}`",
                f"- eval_test_summary_exists: `{rec.get('eval_test_summary_exists')}`",
                f"- last global_step: `{step}`",
                f"- last micro_step: `{micro}`",
                f"- last loss: `{loss}`",
                f"- skipped_batches: `{skipped}`",
                f"- train_log_age_sec: `{age}`",
            ]
        )
        if not rec.get("summary_exists") and not rec.get("pid_alive"):
            lines.append("- ALERT: training process is not alive and `summary.json` does not exist.")
        if age is not None and age > 1800 and not rec.get("summary_exists"):
            lines.append("- ALERT: train log has not updated for more than 30 minutes.")
        lines.append("")
    return "\n".join(lines) + "\n"


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--interval-sec", type=int, default=300)
    parser.add_argument("--remote-host", default="10.176.54.22")
    parser.add_argument("--remote-port", type=int, default=26901)
    parser.add_argument("--local-run", default=str(LOCAL_RUN))
    parser.add_argument("--remote-run", default=str(REMOTE_RUN))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_jsonl = out_dir / "monitor_status.jsonl"
    status_md = out_dir / "monitor_status.md"

    while True:
        records = [
            local_status(Path(args.local_run)),
            remote_status(args.remote_host, args.remote_port, Path(args.remote_run)),
        ]
        payload = {"time": now(), "records": records}
        append_jsonl(status_jsonl, payload)
        status_md.write_text(render_status(records), encoding="utf-8")
        all_done = all(
            rec.get("summary_exists") and rec.get("eval_val_summary_exists") and rec.get("eval_test_summary_exists")
            for rec in records
            if not rec.get("error")
        )
        if all_done:
            break
        time.sleep(max(30, args.interval_sec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
