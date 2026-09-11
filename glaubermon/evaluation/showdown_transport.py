"""Local JSON-lines transport for official player-filtered Showdown observations."""
import json
import subprocess
from pathlib import Path

class CapturedSocket:
    def __init__(self):
        self.messages = []

    async def send(self, message):
        self.messages.append(message)


def choice_text(message):
    command = message.split('|/', 1)[-1].lstrip('/')
    return command.removeprefix('choose ')


class OfficialBridge:
    def __init__(self, showdown):
        self.process = subprocess.Popen(
            ['node', str(Path(__file__).with_name('showdown_bridge.cjs')), str(Path(showdown).resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def exchange(self, message):
        self.process.stdin.write(json.dumps(message) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Official simulator exited')
        result = json.loads(line)
        if 'fatal' in result:
            raise RuntimeError(result['fatal'])
        return result

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=10)
        self.process.stdout.close()


