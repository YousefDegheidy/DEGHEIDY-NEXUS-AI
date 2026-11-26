"""Safe notebook GPU lifecycle helpers; never kill unrelated processes."""
from pathlib import Path
import subprocess, sys, zipfile, json
MIN_FREE_MIB = 12 * 1024

def gpu_memory(run=subprocess.run):
    r = run(['nvidia-smi','--query-gpu=name,memory.total,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    name,total,free = [v.strip() for v in r.stdout.strip().splitlines()[0].rsplit(',',2)]
    return name,int(total),int(free)

def ready_for_training(read=gpu_memory):
    name,total,free=read()
    print(f'{name}: {free} MiB free / {total} MiB total',flush=True)
    return free >= MIN_FREE_MIB

def stop_owned_ui(process, timeout=15):
    if process is None or process.poll() is not None: return
    process.terminate()
    try: process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill(); process.wait(timeout=timeout)

def backup_artifacts(base, output):
    base,output=Path(base),Path(output)
    artifacts=base/'artifacts'
    if not artifacts.is_dir(): return None
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(artifacts.rglob('*')):
            if p.is_file() and not p.is_symlink(): z.write(p,p.relative_to(base))
    return output

def completed_adapter(base):
    adapter=Path(base)/'artifacts/qa74_t4_lora'
    try:
        m=json.loads((adapter/'training_manifest.json').read_text())
        return (m.get('status')=='TRAINED_NOT_QUALITY_CERTIFIED' and
                (adapter/'adapter_config.json').is_file() and
                any((adapter/f).is_file() and (adapter/f).stat().st_size>0 for f in ('adapter_model.safetensors','adapter_model.bin')))
    except (OSError,ValueError,TypeError,AttributeError): return False

def launch_ui(base, env, log):
    log=Path(log); log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w',encoding='utf-8') as f:
        return subprocess.Popen([sys.executable,'-u','-c','from app import start_gradio; import time; ui=start_gradio(); time.sleep(10**8)'],cwd=base,env=env,stdout=f,stderr=subprocess.STDOUT)
