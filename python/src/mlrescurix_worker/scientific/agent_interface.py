"""Common local human/agent interface, independent of scientific task or transport.

Commands are consumed at safe turn boundaries. This is local session control,
not authentication or outbound/model approval. Run it under the session owner's
OS identity; keep the separately authorized upload operator isolated.
"""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import time
import uuid
from .candidate_pipeline import identity
from .evidence import atomic_json

COMMANDS = {'steer', 'pause', 'resume', 'stop'}


class AgentInterface:
    def __init__(self, workspace):
        self.workspace = Path(workspace).resolve()
        self.state_path = self.workspace / 'autonomous_loop_state.json'
        self.inbox_path = self.workspace / 'human_control.json'

    def _state(self):
        state = json.loads(self.state_path.read_text())
        if not isinstance(state.get('binding'), dict):
            raise ValueError('Session checkpoint binding missing')
        return state

    @contextmanager
    def _lock(self):
        # Separate from controller lock so humans can steer a running session.
        with (self.workspace / '.human_control.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try: yield
            finally: fcntl.flock(lock, fcntl.LOCK_UN)

    def _commands(self):
        if not self.inbox_path.exists(): return []
        data = json.loads(self.inbox_path.read_text())
        if not isinstance(data, list) or len(data) > 1000:
            raise ValueError('Invalid or full human control inbox; preserve for reconciliation')
        return data

    def status(self):
        state = self._state()
        with self._lock(): commands = self._commands()
        applied = state.get('human_commands_applied', [])
        return {'session_id': state['binding']['session_id'], 'binding_identity': identity(state['binding']),
            'turns_completed': state['turns_completed'], 'stop_reason': state.get('stop_reason'),
            'pending_turn': state.get('pending_turn'), 'human_guidance': state.get('human_guidance', []),
            'pending_commands': [c['command_id'] for c in commands if c['command_id'] not in applied],
            'accepted_candidate_id': state.get('accepted_candidate_id'),
            'best_cv_score': state.get('best_cv_score'), 'production_certified': False}

    def submit(self, command, message='', *, command_id=None):
        if command not in COMMANDS: raise ValueError('Unsupported human control command')
        if not isinstance(message, str) or len(message.encode()) > 6000:
            raise ValueError('Human message must be at most 6000 UTF-8 bytes')
        if command == 'steer' and not message.strip(): raise ValueError('Steering message required')
        command_id = command_id or str(uuid.uuid4())
        if not isinstance(command_id, str) or not command_id or len(command_id) > 128:
            raise ValueError('Invalid command identity')
        with self._lock():
            binding = identity(self._state()['binding'])
            commands = self._commands()
            content = {'command_id': command_id, 'command': command, 'message': message, 'binding_identity': binding}
            for old in commands:
                if old.get('command_id') == command_id:
                    if any(old.get(k) != v for k, v in content.items()):
                        raise ValueError('Human command identity reused with different content')
                    return command_id
            if len(commands) >= 1000: raise ValueError('Human control inbox full')
            content['submitted_unix'] = time.time()
            content['receipt_identity'] = identity(content)
            commands.append(content)
            atomic_json(self.inbox_path, commands)
        return command_id

    def apply(self, state):
        """Controller only: caller checkpoints returned state before any dispatch."""
        with self._lock(): commands = self._commands()
        applied = state.setdefault('human_commands_applied', [])
        for command in commands:
            content = dict(command)
            digest = content.pop('receipt_identity', None)
            if digest != identity(content) or content.get('binding_identity') != identity(state['binding']):
                raise ValueError('Human control receipt/session binding mismatch')
            if content.get('command') not in COMMANDS:
                raise ValueError('Unsupported human control command')
            if content['command_id'] in applied: continue
            kind = content['command']
            if kind == 'steer':
                guidance = state.setdefault('human_guidance', [])
                guidance.append({'command_id': content['command_id'], 'message': content['message']})
                state['human_guidance'] = guidance[-8:]
            elif kind == 'pause' and state.get('stop_reason') in (None, 'max_turns_reached'):
                state['stop_reason'] = 'human_pause'
            elif kind == 'resume' and state.get('stop_reason') == 'human_pause':
                state['stop_reason'] = None
            elif kind == 'stop' and state.get('stop_reason') in (None, 'human_pause', 'max_turns_reached'):
                state['stop_reason'] = 'human_stop'
            applied.append(content['command_id'])
        return state


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Common human control point for every MLRescurix task')
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('command', choices=['status', *sorted(COMMANDS)])
    parser.add_argument('--message', default='')
    parser.add_argument('--command-id', help='Stable id for safe submission retries')
    args = parser.parse_args(argv)
    interface = AgentInterface(args.workspace)
    if args.command == 'status': result = interface.status()
    else:
        result = {'command_id': interface.submit(args.command, args.message, command_id=args.command_id),
                  'status': 'queued', 'effective_at': 'next controller turn boundary'}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__': raise SystemExit(main())
