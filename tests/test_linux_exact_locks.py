from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def test_offline_locks_are_exact_hashed_and_base_excludes_3d():
    for name in ('linux-runtime', 'orbbec-wrapper', 'addon', 'open3d'):
        path = ROOT / 'requirements' / (name + '-lock-py310.txt')
        lines = [x for x in path.read_text().splitlines() if x and not x.startswith('#')]
        assert lines and all('==' in line and '--hash=sha256:' in line for line in lines)
        assert path.with_suffix('.sha256').read_text().split()[0] == hashlib.sha256(path.read_bytes()).hexdigest()
    base = (ROOT/'requirements/linux-runtime-lock-py310.txt').read_text().lower()
    assert 'open3d' not in base and 'opencv-python-headless==' in base
    assert 'opencv-python==' not in base


def test_official_udev_matches_native_and_is_explicit():
    root = ROOT/'packaging/linux/orbbec-official'
    provenance = json.loads((root/'provenance.json').read_text())
    assert provenance['commit'] == '2f6561c28255d805b34aa00a690199ce40e96c81'
    for name, source in provenance['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == source['sha256']
    installer = (ROOT/'packaging/linux/install_runtime.py').read_text()
    assert 'setup_orbbec_udev' not in installer and 'sudo' not in installer
