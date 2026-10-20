import json

from anomaly.env import environment_report


def test_environment_report_is_json_serializable():
    report = environment_report()
    serialized = json.dumps(report)

    assert isinstance(serialized, str)


def test_environment_report_has_required_keys():
    report = environment_report()

    expected_keys = {
        "utc_time",
        "python",
        "os",
        "cpu_count",
        "ram_total_gib",
        "torch",
        "cuda_available",
        "gpu_name",
    }

    assert expected_keys.issubset(report.keys())


def test_environment_report_gpu_fields_are_consistent(monkeypatch):
    import anomaly.env as env

    monkeypatch.setattr(
        env.torch.cuda,
        "is_available",
        lambda: False,
    )

    report = env.environment_report()

    assert report["cuda_available"] is False
    assert report["gpu_name"] is None
