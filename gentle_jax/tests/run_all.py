"""Run every parity suite in one go.

Each suite runs in its own process: the torch-backed suites build real rlkit
objects, and rlkit registers its environments at import time, so sharing a
process between suites is not worth the coupling.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = [
    ('env', 'test_env_parity.py'),
    ('networks', 'test_network_parity.py'),
    ('data', 'test_data_parity.py'),
    ('dynamics', 'test_dynamics_parity.py'),
    ('single step', 'test_step_parity.py'),
    ('multi step', 'test_multistep_parity.py'),
    ('evaluation', 'test_eval_parity.py'),
]

NOISE = ('hwloc', 'Gym has been', 'Please upgrade', 'See the migration',
         'import module', 'UserWarning', 'logger.warn', 'Box bound',
         'DeprecationWarning', 'tabulate.py', '*', 'WARN:')


def main():
    failures = []
    for name, script in SUITES:
        print(f'\n=== {name} ' + '=' * (60 - len(name)))
        proc = subprocess.run([sys.executable, os.path.join(HERE, script)],
                              capture_output=True, text=True)
        out = proc.stdout + proc.stderr
        for line in out.splitlines():
            if line.strip() and not any(n in line for n in NOISE):
                print(line)
        if proc.returncode != 0 or 'MISMATCH' in out:
            failures.append(name)
    print('\n' + '=' * 68)
    print('ALL SUITES PASSED' if not failures else f'FAILED: {", ".join(failures)}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
