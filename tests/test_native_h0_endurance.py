import pytest

from qualification.native_h0.endurance import GIB, TELEMETRY_FIELDS, required_capacity, telemetry_errors, validate_endurance


def samples(duration=60):
    rows = []
    for second in range(duration+1):
        row = {name:0 for name in TELEMETRY_FIELDS}
        row.update(monotonic_seconds=second, lease_owned=True, rss_bytes=GIB, gpu_memory_bytes=GIB)
        rows.append(row)
    return rows


def test_capacity_includes_margin_and_finalization_reserve():
    assert required_capacity(1000, 1800) == 2250000 + 20*GIB


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_invalid_rate_rejected(rate):
    with pytest.raises(ValueError):
        required_capacity(rate, 7200)


def test_complete_bounded_samples_pass():
    assert telemetry_errors(samples(), 60, {}) == []


def test_missing_samples_never_imputed_to_zero():
    rows = samples()
    rows[1]["writer_queue_depth"] = {"status":"NOT_EXECUTED"}
    assert telemetry_errors(rows, 60, {})


def test_short_run_resource_growth_and_lost_lease_fail():
    assert telemetry_errors(samples(), 1800, {})
    rows = samples()
    rows[-1]["rss_bytes"] = 4*GIB
    rows[12]["lease_owned"] = False
    errors = telemetry_errors(rows, 60, {})
    assert any("resource" in e for e in errors)
    assert any("lease" in e for e in errors)


def test_missing_long_duration_is_not_skipped(tmp_path):
    result = validate_endurance(tmp_path, {})
    assert len(result["errors"]) == 3
