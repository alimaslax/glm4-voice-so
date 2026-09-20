#!/usr/bin/env python3
"""Upload an existing TensorBoard event stream to W&B, optionally following it live."""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import wandb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("event", type=Path)
    p.add_argument("--entity", default="lewenberg-student")
    p.add_argument("--project", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--component", required=True)
    p.add_argument("--round", dest="round_name", required=True)
    p.add_argument("--log-file", type=Path)
    p.add_argument("--follow", action="store_true")
    p.add_argument("--poll-seconds", type=float, default=15)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    event = a.event.resolve()
    if not event.is_file():
        raise SystemExit(f"event file not found: {event}")

    run = wandb.init(
        entity=a.entity,
        project=a.project,
        id=a.run_id,
        name=a.name,
        resume="allow",
        job_type="training",
        config={
            "component": a.component,
            "training_round": a.round_name,
            "tensorboard_event": str(event),
            "source": "tensorboard-backfill" if not a.follow else "tensorboard-live-sidecar",
        },
        tags=[a.component, a.round_name, "vm"],
    )
    if a.log_file and a.log_file.is_file():
        run.save(str(a.log_file.resolve()), base_path=str(a.log_file.resolve().parent), policy="live")

    consumed: dict[str, int] = defaultdict(int)
    while True:
        acc = EventAccumulator(str(event), size_guidance={"scalars": 0})
        acc.Reload()
        by_step: dict[int, dict[str, float]] = defaultdict(dict)
        for tag in acc.Tags().get("scalars", []):
            values = acc.Scalars(tag)
            for value in values[consumed[tag] :]:
                by_step[value.step][tag] = value.value
            consumed[tag] = len(values)

        for step in sorted(by_step):
            run.log(by_step[step], step=step)

        if not a.follow:
            break
        time.sleep(a.poll_seconds)

    run.finish()


if __name__ == "__main__":
    main()
