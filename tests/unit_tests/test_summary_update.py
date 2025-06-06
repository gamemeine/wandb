from unittest.mock import MagicMock

import pytest
import queue
from wandb.sdk.internal import handler, settings_static


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
