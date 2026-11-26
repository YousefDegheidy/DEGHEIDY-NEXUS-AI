"""Stream subprocess stdout/stderr to Colab and save complete diagnostics."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess


def run_logged(command, log_path, *, cwd=None):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    with log_path.open('w', encoding='utf-8') as log:
        log.write('Command: '+repr(command)+'\n'); log.flush()
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace', bufsize=1,
                              cwd=cwd, env=env) as proc:
            try:
                for line in proc.stdout:
                    print(line, end='', flush=True)
                    log.write(line); log.flush()
                code = proc.wait()
            except BaseException:
                proc.terminate(); proc.wait(); raise
        log.write(f'\nExit code: {code}\n')
    if code:
        raise RuntimeError(f'Process exit {code}. Real error is in output above. Full log: {log_path}')
    return log_path
