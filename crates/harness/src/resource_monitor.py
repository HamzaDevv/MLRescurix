# Trusted in-container wrapper. Never executed on the host.
import json, os, pathlib, subprocess, sys

base = pathlib.Path('/sys/fs/cgroup')
receipt = pathlib.Path('/resource/measurement.json')
memory, threads, required = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == 'true'

def sample():
    result = {}
    for name in ['memory.max', 'memory.swap.max', 'memory.current', 'memory.peak',
                 'memory.events', 'cpu.max', 'cpu.stat', 'pids.max', 'pids.events']:
        try:
            result[name] = (base / name).read_text().strip()
        except OSError:
            result[name] = None
    return result

def supported(values):
    try:
        quota, period = map(int, values['cpu.max'].split())
        return (int(values['memory.max']) == memory and int(values['memory.swap.max']) == 0
                and int(values['memory.peak']) >= 0 and values['memory.events'] is not None
                and quota == period * threads and int(values['pids.max']) == 64
                and values['cpu.stat'] is not None and values['pids.events'] is not None)
    except (TypeError, ValueError, AttributeError):
        return False

def save(data):
    tmp = receipt.with_suffix('.tmp')
    with tmp.open('w') as stream:
        json.dump(data, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, receipt)
    fd = os.open(receipt.parent, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)

before = sample()
okay = supported(before)
data = dict(memory_bytes=memory, threads=threads, supported=okay, before=before, complete=False)
save(data)
if required and not okay:
    data['status'] = 'refused_unsupported_cgroup'
    save(data)
    sys.exit(126)
code = subprocess.call(sys.argv[4:])
# Read while this wrapper still keeps the job cgroup alive, before Docker teardown.
after = sample()
data.update(after=after, supported=okay and supported(after), complete=True, exit_code=code)
save(data)
sys.exit(code if code >= 0 else 128 - code)
