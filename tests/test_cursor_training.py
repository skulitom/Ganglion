import numpy as np
import pytest

from ganglion.brain.haltere_cursor import channels
from ganglion.brain.shadow import MotorSample
from ganglion.core.reach import step
from ganglion.train.cursor_world import CursorWorld, teacher_points


def test_training_teacher_matches_resident_controller_at_boundaries_and_varied_speeds():
    rng = np.random.default_rng(710)
    rect = np.array([[25, 40, 640, 360]]*100, dtype=float)
    cursor = rng.uniform([-100, -100], [900, 600], (100, 2))
    goal = rng.uniform([25, 40], [664, 399], (100, 2))
    speed = rng.uniform(100, 5000, 100)
    for dt in (0, .01, .02, 1):
        batched = teacher_points(cursor, goal, speed, rect, dt)
        reference = np.array([step(c, g, dt, s, r) for c, g, s, r in zip(cursor, goal, speed, rect)])
        np.testing.assert_array_equal(batched, reference)


def test_training_senses_match_runtime_adapter_and_use_only_observed_state():
    world = CursorWorld([1000, 1001, 1002])
    previous = world.cursor.copy()
    world.step(world.teacher())
    observed = world.senses(previous)
    for i in range(world.B):
        old = MotorSample("i", "reach", 1, 1, 0, 0, 10, tuple(previous[i]), tuple(world.goal[i]),
                          (0, 0), tuple(world.rect[i]), world.speed[i], .01)
        current = MotorSample("i", "reach", 1, 2, .01, .01, 10, tuple(world.cursor[i]), tuple(world.goal[i]),
                              (0, 0), tuple(world.rect[i]), world.speed[i], .01)
        expected = channels(current, old, version=2)
        for key in expected:
            np.testing.assert_allclose(observed[key][i], expected[key], atol=1e-7)


def test_episode_seed_is_independent_of_batch_composition():
    batch = CursorWorld([1000, 1001, 1002])
    single = CursorWorld([1001])
    for _ in range(30):
        batch.step(batch.teacher())
        single.step(single.teacher())
        np.testing.assert_array_equal(batch.cursor[1], single.cursor[0])
        np.testing.assert_array_equal(batch.goal[1], single.goal[0])


def test_delay_defers_effect_and_domain_randomisation_stays_bounded():
    world = CursorWorld(range(40))
    initial = world.cursor.copy()
    point = world.teacher()
    world.step(point)
    np.testing.assert_array_equal(world.cursor[world.delay > 0], initial[world.delay > 0])
    assert np.all((.6 <= world.gain) & (world.gain <= 1.4))
    assert set(world.delay) == set(range(7))
    for _ in range(100):
        world.step(world.teacher())
    assert np.all(world.cursor >= world.rect[:, :2])
    assert np.all(world.cursor <= world.rect[:, :2]+world.rect[:, 2:]-1)


def test_delayed_absolute_input_applies_the_coordinate_computed_at_issue_time():
    world = CursorWorld([1000])
    world.cursor[:] = [100, 100]
    world.initial = world.cursor.copy()
    world.gain[:] = 1.2
    world.delay[:] = 1
    world.step(np.array([[110, 120]]))
    np.testing.assert_array_equal(world.cursor, [[100, 100]])
    world.step(np.array([[120, 110]]))
    np.testing.assert_allclose(world.cursor, [[112, 124]])
    world.step(np.array([[130, 130]]))
    np.testing.assert_allclose(world.cursor, [[124, 112]])


def test_reference_controller_settles_static_targets_across_randomised_delays():
    world = CursorWorld(range(3000, 3032))
    for _ in range(2000):
        world.step(world.teacher())
    static = np.linalg.norm(world.velocity, axis=1) == 0
    assert np.max(np.linalg.norm(world.goal[static]-world.cursor[static], axis=1)) <= 3


def test_thermal_guard_pauses_until_cool_and_enforces_wall_time():
    from ganglion.train.cursor_readout import Guard
    class Clock:
        now = 0
        def __call__(self): return self.now
        def sleep(self, seconds): self.now += seconds
    clock = Clock()
    readings = iter([68, 61, 57])
    guard = Guard(10, clock=clock, sleep=clock.sleep, temperature_reader=lambda: next(readings))
    assert guard.peak == 68 and clock.now >= 2
    with pytest.raises(TimeoutError):
        Guard(.5, clock=clock, sleep=clock.sleep, temperature_reader=lambda: 68)


@pytest.mark.parametrize("reading", [None, float("nan")])
def test_missing_temperature_stops_training(reading):
    from ganglion.train.cursor_readout import Guard
    with pytest.raises(RuntimeError, match="temperature unavailable"):
        Guard(10, temperature_reader=lambda: reading)


def test_target_kicks_happen_on_schedule_within_the_plant_reach_cap():
    world = CursorWorld(range(40, 52, 2), jump_every=100)      # even seeds: static targets
    goals = [world.goal.copy()]
    for _ in range(250):
        world.step(world.teacher())
        goals.append(world.goal.copy())
    moved = [i for i in range(250) if not np.array_equal(goals[i], goals[i + 1])]
    assert moved == [100, 200] and world.jumps == 2
    for i in moved:
        distance = np.linalg.norm(goals[i + 1] - goals[i], axis=1)
        assert (distance <= world.reach_cap + 1e-6).all() and (world.reach_cap <= 400).all()
        assert (goals[i + 1] >= 40).all() and (goals[i + 1] <= world.rect[:, 2:] - 40).all()
    quiet = CursorWorld(range(40, 52, 2))
    for _ in range(250):
        quiet.step(quiet.teacher())
    assert quiet.jumps == 0 and np.array_equal(quiet.goal, goals[0])


def test_fine_tuning_groups_freeze_everything_else_and_select_on_the_suite():
    from ganglion.train.cursor_finetune import parameter_groups, selection_key

    class Parameter:
        def __init__(self, n):
            self.n, self.grad = n, None

        def requires_grad_(self, flag):
            self.grad = flag

        def numel(self):
            return self.n

    class Brain:
        def __init__(self):
            self.params = [("encoders.goal__goal.weight", Parameter(4)), ("readout.weight", Parameter(8)),
                           ("readout.bias", Parameter(2)), ("log_edge_gain", Parameter(100)),
                           ("log_gain", Parameter(10)), ("bias", Parameter(10)), ("log_tau", Parameter(10)),
                           ("sign_free", Parameter(3))]

        def named_parameters(self):
            return iter(self.params)

    brain = Brain()
    groups = parameter_groups(brain, ["encoders", "readout", "edges"])
    assert [p.n for p in groups["edges"]] == [100] and [p.n for p in groups["readout"]] == [8, 2]
    flags = {name: p.grad for name, p in brain.params}
    assert flags["log_edge_gain"] and flags["readout.bias"] and flags["encoders.goal__goal.weight"]
    assert not flags["bias"] and not flags["log_tau"] and not flags["sign_free"]
    groups = parameter_groups(brain, ["neurons"])
    assert sorted(p.n for p in groups["neurons"]) == [10, 10, 10]
    assert not {name: p.grad for name, p in brain.params}["readout.bias"]
    with pytest.raises(ValueError):
        parameter_groups(brain, ["synapses"])
    worse = {"settle/connectome": {"episodes": 16, "success": 3}, "pursuit/connectome": {"tracking_error_px": {"mean": 50.0}}}
    better = {"settle/connectome": {"episodes": 16, "success": 5}, "pursuit/connectome": {"tracking_error_px": {"mean": 90.0}}}
    lost = {"settle/connectome": {"episodes": 16, "success": 5}, "pursuit/connectome": {"tracking_error_px": {"mean": None}}}
    assert selection_key(better) < selection_key(worse) and selection_key(better) < selection_key(lost)


def test_adapter_version_3_drops_own_velocity_and_matches_the_world():
    world = CursorWorld([1000, 1001, 1002], sense_version=3)
    previous = world.cursor.copy()
    world.step(world.teacher())
    observed = world.senses(previous)
    assert not observed["haltere"].any() and not observed["jo"].any()
    for i in range(world.B):
        old = MotorSample("i", "reach", 1, 1, 0, 0, 10, tuple(previous[i]), tuple(world.goal[i]),
                          (0, 0), tuple(world.rect[i]), world.speed[i], .01)
        current = MotorSample("i", "reach", 1, 2, .01, .01, 10, tuple(world.cursor[i]), tuple(world.goal[i]),
                              (0, 0), tuple(world.rect[i]), world.speed[i], .01)
        expected = channels(current, old, version=3)
        for key in expected:
            np.testing.assert_allclose(observed[key][i], expected[key], atol=1e-7)
        np.testing.assert_allclose(expected["goal"], channels(current, old, version=2)["goal"])
    with pytest.raises(ValueError):
        CursorWorld([1000], sense_version=5)


def test_slip_robustness_options_drop_scale_and_blank_the_slip():
    seeds = [1000, 1001, 1002, 1003]
    exact = CursorWorld(seeds, sense_version=4, view=True)
    dropped = CursorWorld(seeds, sense_version=4, view=True, slip_dropout=1.0)
    halved = CursorWorld(seeds, sense_version=4, view=True, slip_gain=(.5, .5))
    blank = CursorWorld(seeds, sense_version=4, view=True, slip_blank=1.0)
    some = CursorWorld(list(range(2000, 2064)), sense_version=4, view=True, slip_dropout=.5)
    for world in (exact, dropped, halved, blank, some):
        for _ in range(12):
            world.step(world.cursor + [500, 0])
    assert (exact.slip()[:, 0] < 0).all()
    assert not dropped.slip().any() and not blank.slip().any()
    np.testing.assert_allclose(halved.slip(), exact.slip() * .5)
    assert 16 <= some.slip_on.sum() <= 48 and (some.slip()[~some.slip_on] == 0).all() and (some.slip()[some.slip_on, 0] < 0).all()
    assert some.slip_on.tolist() == CursorWorld(list(range(2000, 2064)), sense_version=4, view=True, slip_dropout=.5).slip_on.tolist()
    with pytest.raises(ValueError):
        CursorWorld(seeds, sense_version=4, view=True, slip_gain=(1.5, .5))


def test_adapter_version_4_feeds_the_visual_slip_of_view_episodes_and_matches_the_world():
    from ganglion.core.aim import UNBOUNDED
    world = CursorWorld([1000, 1001, 1002, 1003], sense_version=4, view=[True, True, False, False])
    plain = CursorWorld([1000, 1001, 1002, 1003], sense_version=4)
    assert world.delay.tolist() == (plain.delay + [2, 2, 0, 0]).tolist()
    assert (world.bounds[:2] == np.array(UNBOUNDED, dtype=float)).all() and (world.bounds[2:] == world.rect[2:]).all()
    assert not world.senses().get("lptc").any()                     # nothing has moved yet
    for _ in range(20):
        world.step(world.cursor + [500, 0])                          # turn the view hard to the right
    assert (world.cursor[:2, 0] > world.rect[:2, 2]).all()           # a view is unbounded
    assert (world.cursor[2:, 0] <= world.rect[2:, 2] - 1).all()      # a cursor is not
    senses = world.senses(world.cursor.copy())
    assert not senses["haltere"].any() and not senses["jo"].any()
    assert (senses["lptc"][:2, 0] < 0).all()                         # the picture slips left
    assert not senses["lptc"][2:].any() and not senses["lptc"][:, 2:].any()
    for i in range(2):
        slip = world.slip()[i]
        sample = MotorSample("i", "align", 1, 2, .01, .01, 10, tuple(world.cursor[i]), tuple(world.goal[i]),
                             (0, 0), UNBOUNDED, world.speed[i], .01, flow=(slip[0], slip[1], 0.0, 0.0))
        expected = channels(sample, None, version=4)
        np.testing.assert_allclose(senses["lptc"][i], expected["lptc"], atol=1e-6)
        assert channels(sample, None, version=3)["lptc"] == [0.0] * 6   # older versions ignore the flow
    world.view[:] = False
    assert not world.senses().get("lptc").any()                     # no view, no slip
