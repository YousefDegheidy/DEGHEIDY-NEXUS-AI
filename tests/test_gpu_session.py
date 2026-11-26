import json, zipfile
from types import SimpleNamespace
from training.gpu_session import gpu_memory,ready_for_training,stop_owned_ui,backup_artifacts,completed_adapter

def test_memory_query():
    assert gpu_memory(lambda *a,**kw:SimpleNamespace(stdout='Tesla T4, 14913, 14800\n')) == ('Tesla T4',14913,14800)

def test_free_guard():
    assert ready_for_training(lambda:('T4',14913,12288))
    assert not ready_for_training(lambda:('T4',14913,12287))

def test_stop_owned():
    class P:
        def __init__(self):self.actions=[]
        def poll(self):return None
        def terminate(self):self.actions.append('terminate')
        def wait(self,timeout):self.actions.append('wait')
    p=P();stop_owned_ui(p);assert p.actions==['terminate','wait']
    stop_owned_ui(None)

def test_backup_and_completed(tmp_path):
    a=tmp_path/'artifacts/qa74_t4_lora';a.mkdir(parents=True)
    (a/'training_manifest.json').write_text(json.dumps({'status':'TRAINED_NOT_QUALITY_CERTIFIED'}))
    assert not completed_adapter(tmp_path)
    (a/'adapter_model.safetensors').write_bytes(b'weights')
    (a/'adapter_config.json').write_text('{}')
    assert completed_adapter(tmp_path)
    out=backup_artifacts(tmp_path,tmp_path/'backup.zip')
    with zipfile.ZipFile(out) as z:assert z.read('artifacts/qa74_t4_lora/adapter_model.safetensors')==b'weights'
    assert (a/'adapter_model.safetensors').exists()

def test_stop_already_exited():
    stop_owned_ui(SimpleNamespace(poll=lambda:0))
