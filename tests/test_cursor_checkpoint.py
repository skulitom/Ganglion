"""Opt-in headless CUDA check of a real cursor checkpoint's runtime adapter."""
import os

import numpy as np
import pytest


@pytest.mark.skipif(not os.environ.get("GANGLION_CURSOR_CHECKPOINT"), reason="optional trained CUDA checkpoint")
def test_trained_checkpoint_runtime_matches_training_forward_and_reset():
    import torch
    from ganglion.brain.haltere_cursor import HaltereCursor
    from ganglion.brain.shadow import MotorSample
    from ganglion.train.cursor_readout import observe
    from ganglion.train.cursor_world import CursorWorld

    adapter = HaltereCursor(os.environ["GANGLION_CURSOR_CHECKPOINT"])
    assert adapter.adapter_version == 2 and adapter.desktop_trained
    assert adapter.metadata["training"]["control_authority"] is False
    world = CursorWorld([7000])
    previous = None
    state = adapter.brain.init_state(1)
    first = first_proposal = None
    with torch.inference_mode():
        for tick in range(20):
            obs, _ = observe(torch, adapter.brain, world.senses(previous))
            expected, state, _ = adapter.brain(obs, state, adapter.weights)
            now = 10 + tick*.01
            sample = MotorSample("parity", "reach", 1, tick, now, now, now+1,
                                 tuple(world.cursor[0]), tuple(world.goal[0]), tuple(world.teacher()[0]),
                                 tuple(world.rect[0]), float(world.speed[0]), .01)
            actual = adapter.predict(sample)
            np.testing.assert_allclose(actual["raw_actions"], expected[0].cpu().numpy(), atol=1e-4)
            assert actual["desktop_trained"] and np.isfinite(actual["raw_actions"]).all()
            if first is None:
                first, first_proposal = sample, actual
            previous = world.cursor.copy()
            world.step(world.teacher())
    adapter.reset()
    repeated = adapter.predict(first)
    np.testing.assert_allclose(repeated["raw_actions"], first_proposal["raw_actions"], atol=1e-6)
    assert repeated["point"] == first_proposal["point"]
