import queue
from collections import defaultdict
from unittest.mock import MagicMock

import pytest
import wandb
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import handler, sample, settings_static


def test_handle_bigint(test_settings, monkeypatch):
    monkeypatch.setattr(
        handler.InternalApi,
        "sweep",
        lambda *args, **kwargs: {"config": "{}"},
    )

    result_q = queue.Queue()
    settings = test_settings({})
    hm = handler.HandleManager(
        settings=settings_static.SettingsStatic(settings.to_proto()),
        record_q=MagicMock(),
        result_q=result_q,
        stopped=MagicMock(),
        writer_q=MagicMock(),
        interface=MagicMock(),
        context_keeper=MagicMock(),
    )

    sampled_history = pb.SampledHistoryRequest()
    request = pb.Request()
    request.sampled_history.CopyFrom(sampled_history)
    record = pb.Record()
    record.request.CopyFrom(request)

    bigint = 12379259919636694194
    hm._sampled_history = defaultdict(sample.UniformSampleAccumulator)
    hm._sampled_history["ints"].add(1)
    hm._sampled_history["floats"].add(2.2)
    hm._sampled_history["floats"].add(4.5)
    hm._sampled_history["bigint"].add(bigint)

    hm.handle(record)
    result = result_q.get()

    history = result.response.sampled_history_response
    sampled_history = {
        item.key: wandb.util.downsample(item.values_float or item.values_int, 40)
        for item in history.item
    }
    assert sampled_history["ints"] == [1]
    assert len(sampled_history["floats"]) == 2
    assert len(sampled_history["bigint"]) == 0


@pytest.mark.parametrize(
    "sweep_summary,current_loss,new_loss,expected_loss",
    [
        ("minimize", 0.5, 0.3, 0.3),
        ("maximize", 0.5, 0.8, 0.8),
        ("last",     0.5, 0.2, 0.2),
        (None,       0.5, 0.2, 0.2),
    ],
)
def test_update_summary_loss_goal(
    test_settings, monkeypatch, sweep_summary, current_loss, new_loss, expected_loss
):
    monkeypatch.setattr(
        handler.InternalApi,
        "sweep",
        lambda *args, **kwargs: {"config": "metric:\n  name: loss\n  goal: minimize"}
    )

    result_q = queue.Queue()
    settings = test_settings({})
    hm = handler.HandleManager(
        settings=settings_static.SettingsStatic(settings.to_proto()),
        record_q=MagicMock(),
        result_q=result_q,
        stopped=MagicMock(),
        writer_q=MagicMock(),
        interface=MagicMock(),
        context_keeper=MagicMock(),
    )

    hm._sweep_metric = "loss"
    hm._sweep_summary = sweep_summary

    hm._consolidated_summary = {"loss": current_loss}
    hm._metric_defines = {}

    updated_keys = hm._update_summary({"loss": new_loss})

    assert "loss" in updated_keys
    assert hm._consolidated_summary["loss"] == expected_loss
