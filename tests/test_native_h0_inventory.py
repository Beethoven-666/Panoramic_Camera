import copy

import pytest

from qualification.native_h0 import inventory
from qualification.native_h0.inventory import collect_camera, inventory_errors, validate_inventory


def healthy():
    binding = dict(h0_id="h0", subject_source_commit="subject", h0_tool_commit="tool", camera_serial="s",
                   usb_port_path="1-2", udev_rule_sha256="rule")
    values = {
        "host": dict(system="Linux", os_release={"ID": "ubuntu", "VERSION_ID": "22.04"}, architecture="x86_64",
            python_implementation="CPython", python_version=[3, 10, 1], effective_uid=1000,
            virtualization={"stdout": "vmware\n", "returncode": 0}, glibc={"stdout": "glibc 2.35"}, g305_cuda="required"),
        "gpu": dict(nvidia_smi={"returncode": 0}, cupy={"major": 12, "minor": 0, "kernel_output": [i*3+7 for i in range(8)]},
            open3d={"version": "0.19.0+1e7b17438", "build_cuda_module": True, "cuda_available": True, "device_count": 1, "cuda_result": [2., 4.]}),
        "storage": dict(free_bytes=40*1024**3, findmnt={"stdout": '{"filesystems":[{"fstype":"ext4","source":"/dev/nvme0n1p1"}]}'}),
        "camera": dict(probe_camera=True, device_count=1, device={"name": "Gemini 305", "vid": 0x2bc5, "pid": 0x0840,
            "firmware_version": "1.0.70", "serial_number": "s"}, wrapper_version="2.1.2+g305.1", native_sdk_version="2.9.3",
            ordinary_user_open_read=True, lock_root_writable=True, sync_readback={"trigger_out_enable": False},
            profiles=[{"width":848,"height":480,"fps":60,"format":"RGB"},{"width":848,"height":480,"fps":60,"format":"Y16"}],
            udev_rule={"sha256":"rule"}),
        "usb": dict(devices=[{"serial":"s","speed":"5000","port_path":"1-2"}]),
    }
    for value in values.values():
        value.update({key:binding[key] for key in ("h0_id", "subject_source_commit", "h0_tool_commit")})
    return values, binding


def test_expected_vmware_observations_pass():
    records, binding = healthy()
    assert inventory_errors(records, binding) == []


@pytest.mark.parametrize("field,value", [("effective_uid",0), ("architecture","aarch64"),
    ("python_version",[3,11]), ("virtualization",{"stdout":"kvm","returncode":0}),
    ("virtualization",{"stdout":"none","returncode":1}),
    ("virtualization",{"stdout":"vmware","returncode":1}),
    ("virtualization",{"status":"NOT_EXECUTED"}),
    ("virtualization",{"stdout":"docker","returncode":0}),
    ("virtualization",{"stdout":"oracle","returncode":0}),
    ("proc_version","Microsoft WSL2"), ("kernel","microsoft-standard-WSL2"),
    ("container_markers",["/.dockerenv"]), ("pythonpath","/source")])
def test_unsupported_host_never_passes(field, value):
    records, binding = healthy()
    records["host"][field] = value
    assert inventory_errors(records, binding)


def test_wrong_gpu_kernel_or_usb_speed_rejected():
    records, binding = healthy()
    for mutated in ("kernel", "compute_capability", "open3d_cuda", "speed"):
        changed = copy.deepcopy(records)
        if mutated == "kernel":
            changed["gpu"]["cupy"]["kernel_output"] = []
        elif mutated == "compute_capability":
            changed["gpu"]["cupy"]["major"] = 8
        elif mutated == "open3d_cuda":
            changed["gpu"]["open3d"]["cuda_available"] = False
        else:
            changed["usb"]["devices"][0]["speed"] = "480"
        assert inventory_errors(changed, binding)


def test_probe_is_not_attempted_before_replay():
    assert collect_camera("unused", replay_verified=False)["status"] == "NOT_EXECUTED"


def test_missing_inventory_cannot_qualify(tmp_path):
    assert validate_inventory(tmp_path, healthy()[1])["errors"]


@pytest.mark.parametrize("vmware,replay,python_version,expected_probe", [
    (True, True, [3, 10, 1], True),
    (True, False, [3, 10, 1], False),
    (False, True, [3, 10, 1], False),
    (True, True, [3, 12, 1], False),
])
def test_camera_probe_requires_vmware_host_and_replay(tmp_path, monkeypatch, vmware, replay, python_version, expected_probe):
    records, binding = healthy()
    host = records["host"]
    host["python_version"] = python_version
    if not vmware:
        host["virtualization"] = {"stdout": "none", "returncode": 1}
    probes = []
    monkeypatch.setattr(inventory, "collect_host", lambda: host)
    monkeypatch.setattr(inventory, "collect_gpu", lambda: records["gpu"])
    monkeypatch.setattr(inventory, "collect_storage", lambda root: records["storage"])
    monkeypatch.setattr(inventory, "collect_usb", lambda camera: records["usb"])
    def camera(root, *, replay_verified):
        probes.append(replay_verified)
        return records["camera"]
    monkeypatch.setattr(inventory, "collect_camera", camera)
    inventory.collect_inventory(tmp_path, tmp_path, tmp_path, binding=binding, replay_verified=replay)
    assert probes == [expected_probe]
