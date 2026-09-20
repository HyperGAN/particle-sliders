"""Lease and safely release the task's RunPod GPU; credentials stay local."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'artifacts/anima/runpod'
NAME = 'anima-turbo-studio-20260919'


def write(name, value):
    STATE.mkdir(parents=True, exist_ok=True)
    path = STATE / name
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def api(method, route='', body=None, graphql=False):
    key = (Path.home() / '.runpod.secret').read_text().strip()
    url = 'https://api.runpod.io/graphql' if graphql else 'https://rest.runpod.io/v1/' + route
    request = urllib.request.Request(url, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json',
                 'User-Agent': 'anima-studio/1'})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


def account():
    return api('POST', body={'query':'query { myself { clientBalance currentSpendPerHr } }'}, graphql=True)['data']['myself']


def lease():
    return json.loads((STATE / 'lease.json').read_text())


def refresh():
    state = lease()
    pod = api('GET', 'pods/' + state['id'])
    if pod['name'] != state.get('name', NAME):
        raise RuntimeError('Task pod identity mismatch')
    ip, port = pod.get('publicIp'), pod.get('portMappings', {}).get('22')
    # RunPod can briefly omit network mappings for a healthy running pod.
    # Preserve the last complete endpoint instead of writing an unusable pair.
    if ip and port:
        state.update(ip=ip, port=port)
    write('lease.json', state)
    return state


def ssh_args(state):
    return ['ssh','-i',state['ssh_key'],'-p',str(state['port']),'-o','BatchMode=yes',
            '-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=accept-new',
            '-o','UserKnownHostsFile='+state['ssh_key']+'-hosts','-o','ConnectTimeout=15',
            '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2']


def remote(command, timeout=60):
    state = refresh()
    return subprocess.run([*ssh_args(state),'root@'+state['ip'],command], text=True, timeout=timeout)


def sync(state, verify=False):
    with (STATE / 'sync.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _sync(state, verify)


def _sync(state, verify=False):
    if not state.get('port'):
        return False
    dest = ROOT / 'artifacts/anima/remote'
    dest.mkdir(parents=True, exist_ok=True)
    def stage(name):
        write('sync-progress.json', dict(id=state['id'], stage=name,
                                       checksum_verification=verify, at=time.time()))
        print('Preserving RunPod data:', name, flush=True)
    stage('database snapshot')
    # SQLite's backup API gives a consistent view even while the queue is live.
    backup = "import sqlite3,pathlib; p=pathlib.Path('/workspace/anima/artifacts/anima/studio/studio.sqlite3'); s=sqlite3.connect(p); d=sqlite3.connect(p.with_name('studio.snapshot.sqlite3')); s.backup(d); d.close(); s.close()"
    snapshot = subprocess.run([*ssh_args(state), 'root@'+state['ip'],
        '/workspace/anima/.venv-anima/bin/python -c '+shlex.quote(backup)], capture_output=True, timeout=60)
    if snapshot.returncode:
        return False
    # Production state and the image database must reach local storage before
    # disposable benchmark/replay copies can consume the transfer window.
    stage('current runs and Studio history')
    critical = subprocess.run(['rsync', '-azc' if verify else '-az', '--partial', '--timeout=45',
        '--exclude=*.tmp', '--exclude=studio.sqlite3', '--exclude=*-wal', '--exclude=*-shm',
        '--exclude=/runs/archive/***',
        '--exclude=state-*.pt',
        '--include=/runs/***', '--include=/studio/***', '--exclude=*',
        '-e', shlex.join(ssh_args(state)), f"root@{state['ip']}:/workspace/anima/artifacts/anima/",
        str(dest) + '/'], capture_output=True, timeout=600)
    if critical.returncode:
        return False
    import shutil
    # Once local serving starts, that database owns new queue requests. A retry
    # of archival transfer must never replace it with the frozen remote copy.
    if not (STATE / 'live-local.json').exists():
        shutil.copy2(dest/'studio/studio.snapshot.sqlite3', dest/'studio/studio.sqlite3.tmp')
        (dest/'studio/studio.sqlite3.tmp').replace(dest/'studio/studio.sqlite3')
    write('production-backup.json', dict(id=state['id'], at=time.time(), checksum_verified=verify))
    stage('verification and metadata')
    result = subprocess.run(['rsync','-azc' if verify else '-az','--partial','--timeout=45','--exclude=*.tmp',
        '--exclude=studio.sqlite3','--exclude=*-wal','--exclude=*-shm',
        '--exclude=model','--exclude=targets','--exclude=converter','--exclude=runpod', '--exclude=/runs', '--exclude=/studio',
        '-e',shlex.join(ssh_args(state)),f"root@{state['ip']}:/workspace/anima/artifacts/anima/",str(dest)+'/'],
        capture_output=True, timeout=240)
    if result.returncode:
        return False
    # Periodic full snapshots and retried runs are immutable historical evidence.
    # The current full resume.pt was prioritized above. Preserve historical
    # state too without putting multi-GB archives ahead of the latest resume.
    stage('archived runs and periodic full checkpoints')
    archives = subprocess.run(['rsync', '-azc' if verify else '-az', '--partial', '--timeout=45',
        '--exclude=*.tmp', '--include=/runs/', '--include=/runs/archive/***',
        '--include=/runs/*/', '--include=/runs/*/state-*.pt', '--exclude=*',
        '-e', shlex.join(ssh_args(state)), f"root@{state['ip']}:/workspace/anima/artifacts/anima/",
        str(dest) + '/'], capture_output=True, timeout=900)
    if archives.returncode:
        return False
    # Frozen targets are part of exact resume provenance. Preserve committed
    # shards throughout preparation too; this pod has only ephemeral storage.
    # The shared neutral cache is disposable and not referenced by training.
    target_dir = dest / 'targets'
    target_dir.mkdir(exist_ok=True)
    stage('frozen target shards')
    targets = subprocess.run(['rsync','-ac' if verify else '-a','--partial','--timeout=45',
        '--exclude=*.tmp','--exclude=neutral-cache','-e',shlex.join(ssh_args(state)),
        f"root@{state['ip']}:/workspace/anima/artifacts/anima/targets/",str(target_dir)+'/'],
        capture_output=True, timeout=900)
    if targets.returncode == 0:
        stage('complete')
    return targets.returncode == 0


def local_history(state):
    """Keep the Studio history reachable after a verified GPU lease shutdown."""
    import signal
    tunnels = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError):
            continue
        if (state['ssh_key'].encode() in args and b'-fNT' in args
                and any(a.endswith(b':8876:127.0.0.1:8876') for a in args)):
            pid = int(path.parent.name)
            os.kill(pid, signal.SIGTERM)
            tunnels.append(pid)
    deadline = time.monotonic() + 5
    while any(Path(f'/proc/{pid}').exists() for pid in tunnels) and time.monotonic() < deadline:
        time.sleep(.1)
    root = ROOT / 'artifacts/anima/remote'
    model = root / 'model'
    if not model.exists():
        model.symlink_to(ROOT / 'artifacts/anima/model', target_is_directory=True)
    sys.path.insert(0, str(ROOT))
    from lumen_studio.store import Store
    store = Store(root / 'studio/studio.sqlite3')
    owner = store.controls()['owner']
    if not isinstance(owner, dict) or not Path(f"/proc/{owner.get('pid')}/cmdline").exists():
        store.control(owner=None)
    hosts = ['127.0.0.1']
    preview = STATE / 'preview.json'
    if preview.exists():
        from urllib.parse import urlsplit
        host = urlsplit(json.loads(preview.read_text())['url']).hostname
        if host and host not in hosts:
            hosts.append(host)
    servers = []
    for host in hosts:
        existing = None
        for command in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                args = command.read_bytes().split(b'\0')
            except (FileNotFoundError, PermissionError):
                continue
            def option(flag, value):
                return any(args[i:i + 2] == [flag.encode(), value.encode()]
                           for i in range(len(args) - 1))
            if (b'lumen_studio' in args and b'serve' in args
                    and option('--root', str(root)) and option('--host', host)
                    and option('--port', '8876')):
                existing = int(command.parent.name)
                break
        if existing is not None:
            servers.append(dict(pid=existing, url=f'http://{host}:8876'))
            continue
        with (STATE / 'local-history.log').open('a') as out:
            server = subprocess.Popen([sys.executable, '-m', 'lumen_studio', '--root', str(root),
                'serve', '--host', host, '--port', '8876'], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        servers.append(dict(pid=server.pid, url=f'http://{host}:8876'))
    write('local-history-server.json', dict(servers=servers, gpu_worker=False))


def terminate(reason):
    state = refresh()
    # Only processes registered by this task receive SIGTERM. The worker saves
    # after a complete update; give it time to release the GPU before copying.
    stop = """import os,pathlib,signal,time
root=pathlib.Path('/workspace/anima/artifacts/anima')
pids=[]
for name in ('worker.pid','campaign.pid','api.pid','server.pid'):
 p=root/name
 if p.exists():
  pid=int(p.read_text()); cmd=pathlib.Path(f'/proc/{pid}/cmdline')
  if cmd.exists() and ('lumen_studio' in cmd.read_text() or 'anima_campaign' in cmd.read_text()):
   if pid not in pids:
    os.kill(pid,signal.SIGTERM); pids.append(pid)
end=time.time()+180
def alive(pid):
 try: return pathlib.Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0] != 'Z'
 except FileNotFoundError: return False
while any(alive(pid) for pid in pids) and time.time()<end: time.sleep(1)
if any(alive(pid) for pid in pids): raise RuntimeError('Task process has not reached a safe stop')
"""
    stopped = subprocess.run([*ssh_args(state),'root@'+state['ip'],
        '/workspace/anima/.venv-anima/bin/python -c '+shlex.quote(stop)], capture_output=True, timeout=200)
    synced = stopped.returncode == 0 and sync(state, verify=True)
    if not synced:
        # This lease uses ephemeral container storage. STOP can destroy it too;
        # leave the pod intact and retry the backup instead of claiming safety.
        write('recovery-needed.json',dict(reason=reason,outputs_synced=False,at=time.time(),id=state['id']))
        raise RuntimeError('Final backup or safe worker stop failed; pod retained for recovery')
    api('DELETE','pods/'+state['id'])
    write('termination.json',dict(reason=reason,outputs_synced=True,checksum_verified=True,at=time.time(),id=state['id']))
    local_history(state)


def create():
    STATE.mkdir(parents=True, exist_ok=True)
    if (STATE/'lease.json').exists() or any(p['name']==NAME for p in api('GET','pods')):
        raise RuntimeError('Task lease already exists; reuse it')
    balance = account()
    if balance['clientBalance'] < 1.5:
        raise RuntimeError('Insufficient RunPod credit')
    key = Path.home()/'.cache/anima-studio-runpod'/NAME
    key.parent.mkdir(parents=True,exist_ok=True)
    if not key.exists():
        subprocess.run(['ssh-keygen','-q','-t','ed25519','-N','','-f',str(key)],check=True)
    request=dict(name=NAME,imageName='runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04',
        cloudType='SECURE',computeType='GPU',gpuCount=1,
        gpuTypeIds=['NVIDIA GeForce RTX 3090','NVIDIA RTX A5000'],gpuTypePriority='custom',
        interruptible=False,containerDiskInGb=100,volumeInGb=0,minRAMPerGPU=24,minVCPUPerGPU=4,
        ports=['22/tcp'],supportPublicIp=True,
        env={'PUBLIC_KEY':key.with_suffix('.pub').read_text().strip(),'CUDA_VISIBLE_DEVICES':'0'})
    write('request.json',request)
    pod=api('POST','pods',request)
    rate=max(float(pod['costPerHr']),float(pod.get('adjustedCostPerHr') or 0))+.03
    now=time.time()
    state=dict(id=pod['id'],name=NAME,created=now,compute_rate=pod['costPerHr'],rate_bound=rate,
        cap_usd=min(4.5,balance['clientBalance']-.75),ssh_key=str(key),ip=pod.get('publicIp'),
        port=pod.get('portMappings',{}).get('22'),initial_balance=balance['clientBalance'])
    state['deadline']=now+state['cap_usd']/rate*3600
    write('lease.json',state)
    if rate>.6:
        api('DELETE','pods/'+state['id'])
        write('termination.json',dict(reason='Unexpected rate',id=state['id']))
        raise RuntimeError('Unexpected rental rate')
    with (STATE/'watchdog.log').open('a') as out:
        watcher=subprocess.Popen([sys.executable,'-u',__file__,'watch'],stdin=subprocess.DEVNULL,
            stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    write('watchdog.json',dict(pid=watcher.pid))
    print(json.dumps({k:state[k] for k in ('id','compute_rate','rate_bound','cap_usd','deadline')}))


def watch():
    while not (STATE/'termination.json').exists() and not (STATE/'stopped.json').exists():
        try:
            state=refresh()
            if time.time()>=state['deadline'] or account()['clientBalance']<state.get('credit_reserve_usd', .65):
                terminate('Lease budget reached; results preserved')
                return
            sync(state)
        except Exception as exc:
            print(type(exc).__name__, str(exc), flush=True)
        time.sleep(60)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['create','status','watch','sync','terminate','remote'])
    p.add_argument('command',nargs='?')
    a=p.parse_args()
    if a.action=='create': create()
    elif a.action=='status': print(json.dumps(refresh()))
    elif a.action=='watch': watch()
    elif a.action=='sync': print(sync(refresh()))
    elif a.action=='terminate': terminate('Task complete or GPU no longer needed')
    else: sys.exit(remote(a.command,timeout=300).returncode)
