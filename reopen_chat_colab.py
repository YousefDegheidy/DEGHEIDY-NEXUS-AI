# STAGE 3 — Start the improved chatbot with the existing trained adapter.
import os, sys, time, socket, subprocess, json, re
from pathlib import Path
import requests
from google.colab import output

BASE = Path('/content/DEGHEIDY_QA74_811_RELEASE')
ADAPTER = BASE / 'artifacts/qa74_t4_lora'
assert (BASE / 'app.py').is_file(), 'Project missing; do not restart training. Restore the project first.'
manifest = json.loads((ADAPTER / 'training_manifest.json').read_text())
assert (ADAPTER / 'adapter_config.json').is_file()
assert any((ADAPTER / n).is_file() for n in ('adapter_model.safetensors', 'adapter_model.bin'))
revision = manifest['base_model_revision']
assert manifest['base_model'] == 'Qwen/Qwen3.5-4B' and re.fullmatch(r'[a-f0-9]{40}', revision)

# Close only processes/UI previously started by this notebook.
for name in ('_QA74_CHAT_PROCESS', '_QA74_UI_PROCESS'):
    old = globals().get(name)
    if isinstance(old, subprocess.Popen) and old.poll() is None:
        old.terminate()
        try: old.wait(timeout=15)
        except subprocess.TimeoutExpired: old.kill(); old.wait(timeout=15)
old_ui = globals().get('_DEGHEIDY_UI')
if old_ui is not None and hasattr(old_ui, 'close'):
    old_ui.close()

with socket.socket() as s:
    s.bind(('127.0.0.1', 0))
    PORT = s.getsockname()[1]

env = dict(os.environ, DEGHEIDY_MODEL=manifest['base_model'],
           DEGHEIDY_MODEL_REVISION=revision, DEGHEIDY_QWEN_ADAPTER=str(ADAPTER),
           DEGHEIDY_ASR_CPU_ONLY='1', GRADIO_SSR_MODE='False',
           GRADIO_ANALYTICS_ENABLED='False', QA74_PORT=str(PORT))
# Keep public URL private; retain existing optional auth if configured.
child = '''
import os, inspect, threading
import gradio as gr
from app import make_ui
from pathlib import Path
from src.qwen_adapter import validate_adapter_base
validate_adapter_base(Path(os.environ['DEGHEIDY_QWEN_ADAPTER']), os.environ['DEGHEIDY_MODEL'], os.environ['DEGHEIDY_MODEL_REVISION'])
ui = make_ui()
kwargs = dict(share=True, inline=False, inbrowser=False, prevent_thread_lock=True,
              server_name='127.0.0.1', server_port=int(os.environ['QA74_PORT']), show_error=True)
for key in ('ssr_mode', 'pwa'):
    if key in inspect.signature(gr.Blocks.launch).parameters: kwargs[key] = False
password = os.environ.get('DEGHEIDY_GRADIO_PASSWORD', '').strip()
if password: kwargs['auth'] = (os.environ.get('DEGHEIDY_GRADIO_USER', 'degh'), password)
ui.queue(default_concurrency_limit=1).launch(**kwargs)
print('QA74_LAUNCH_RETURNED', flush=True)
print('QA74_PUBLIC_URL:', ui.share_url, flush=True)
threading.Event().wait()
'''
log = BASE / 'artifacts/logs/stage5_chat_only.log'
log.parent.mkdir(parents=True, exist_ok=True)
with log.open('w', encoding='utf-8') as f:
    _QA74_CHAT_PROCESS = subprocess.Popen([sys.executable, '-u', '-c', child], cwd=BASE,
                                         env=env, stdout=f, stderr=subprocess.STDOUT)
print('New server PID:', _QA74_CHAT_PROCESS.pid, '| port:', PORT)
# Wait for the NEW process and an actual HTTP response; never reuse old links.
ready = False
public_url = None
for _ in range(90):
    if _QA74_CHAT_PROCESS.poll() is not None: break
    try:
        r = requests.get(f'http://127.0.0.1:{PORT}/config', timeout=1)
        ready = (r.status_code == 200 and 'components' in r.json()) or (r.status_code in (401, 403) and bool(env.get('DEGHEIDY_GRADIO_PASSWORD')))
    except (requests.RequestException, ValueError): pass
    text = log.read_text(errors='replace')
    m = re.search(r'QA74_PUBLIC_URL:\s*(https://[^\s]+)', text)
    if m: public_url = m.group(1)
    if ready and ('QA74_LAUNCH_RETURNED' in text): break
    time.sleep(1)
if ready and _QA74_CHAT_PROCESS.poll() is None and 'QA74_LAUNCH_RETURNED' in log.read_text(errors='replace'):
    print('CHATBOT SERVER READY — trained adapter selected.')
    if public_url:
        try:
            external=requests.get(public_url+'/config',timeout=8)
            ok=(external.status_code==200 and 'components' in external.json()) or (external.status_code in (401,403) and bool(env.get('DEGHEIDY_GRADIO_PASSWORD')))
        except (requests.RequestException,ValueError):ok=False
        if ok: print('NEW verified public link (keep private):',public_url)
        else: print('Public tunnel not confirmed. Use Colab proxy below.')
    try: output.serve_kernel_port_as_window(PORT)
    except Exception as exc: print('Colab proxy error:', exc)
else:
    print('Server not ready. Do not open the old gradio.live link.')
    print(log.read_text(errors='replace')[-10000:])
    print('Log:', log)
