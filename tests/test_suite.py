"""The fixed suite: identical episodes, an envelope twin of the runtime, and honest metrics."""
import numpy as np

from ganglion.core.reach import supervise
from ganglion.train.suite import (JUMP_EVERY, SuiteWorld, TASKS, proposed_points, run_task, settle_tick,
                                  supervise_batch, table)


def test_batched_envelope_matches_the_runtime_envelope_case_by_case():
    rng = np.random.default_rng(31)
    n = 400
    rect = np.array([[25, 40, 640, 360]] * n, dtype=float)
    cursor = rng.uniform([-50, -50], [750, 450], (n, 2))
    goal = rng.uniform([25, 40], [664, 399], (n, 2))
    goal[:40] = cursor[:40] + rng.uniform(-4, 4, (40, 2))        # within tolerance: holds are accepted
    speed = rng.uniform(100, 5000, n)
    velocity = rng.uniform(-1.5, 1.5, (n, 2))
    velocity[40:60] = 0                                          # zero proposals
    velocity[60:80] = np.nan                                     # malformed proposals
    for dt in (0, .01, .02, .5):
        candidate, rejected = supervise_batch(cursor, goal, velocity, dt, speed, rect)
        for i in range(n):
            expected = supervise(cursor[i], goal[i], velocity[i], dt, speed[i], rect[i])
            if expected is None:
                assert rejected[i], i
            else:
                assert not rejected[i], i
                np.testing.assert_array_equal(candidate[i], expected)
    assert 0 < rejected.sum() < n


def test_proposals_obey_the_speed_limit_and_area():
    rect = np.array([[0, 0, 100, 100]] * 3, dtype=float)
    cursor = np.array([[50, 50], [99, 99], [0, 0]], dtype=float)
    velocity = np.array([[1, 1], [1, 1], [-1, -1]])
    point, _, finite = proposed_points(cursor, velocity, .01, np.array([1000, 1000, 1000.]), rect)
    assert finite.all()
    assert np.hypot(*(point[0] - cursor[0])) <= 10 + .5           # magnitude clamp, then rounding
    np.testing.assert_array_equal(point[1], [99, 99])
    np.testing.assert_array_equal(point[2], [0, 0])


def test_settle_tick_is_the_start_of_the_final_run_within_tolerance():
    assert settle_tick(np.array([1, 2, 3])) == 0
    assert settle_tick(np.array([10, 7, 5, 5, 6])) == 2
    assert settle_tick(np.array([10, 5, 5, 7, 5])) == 4
    assert settle_tick(np.array([10, 5, 5, 7])) is None
    assert settle_tick(np.array([30, 20, 10]), tolerance=12) == 2


def test_episodes_are_identical_across_controllers_and_differ_by_task():
    for task in TASKS:
        a, b = SuiteWorld(range(20000, 20004), task, steps=300), SuiteWorld(range(20000, 20004), task, steps=300)
        np.testing.assert_array_equal(a.cursor, b.cursor)
        np.testing.assert_array_equal(a.goal, b.goal)
        np.testing.assert_array_equal(a.velocity, b.velocity)
        assert a.delay.tolist() == b.delay.tolist()
    static = SuiteWorld(range(20000, 20008), "settle", steps=300)
    assert not static.velocity.any()
    moving = SuiteWorld(range(22000, 22008), "pursuit", steps=300)
    assert (np.linalg.norm(moving.velocity, axis=1) > 0).all()
    camera = SuiteWorld(range(23000, 23008), "camera", steps=300)
    plain = SuiteWorld(range(23000, 23008), "pursuit", steps=300)
    assert (camera.delay == plain.delay + 2).all()
    error = camera.goal - camera.cursor
    assert (np.abs(error[:, 0]) <= camera.rect[:, 2] / 2).all() and (np.abs(error[:, 1]) <= camera.rect[:, 3] / 2).all()


def test_jump_moves_the_target_only_at_the_jump_ticks():
    world = SuiteWorld(range(21000, 21004), "jump", steps=2 * JUMP_EVERY + 5)
    goals = [world.goal.copy()]
    for _ in range(2 * JUMP_EVERY + 5):
        world.step(world.teacher())
        goals.append(world.goal.copy())
    changed = [not np.array_equal(goals[i], goals[i + 1]) for i in range(len(goals) - 1)]
    assert [i for i, c in enumerate(changed) if c] == [JUMP_EVERY, 2 * JUMP_EVERY]
    assert world.jumps == 2
    for before, after in ((goals[JUMP_EVERY], goals[JUMP_EVERY + 1]), (goals[2 * JUMP_EVERY], goals[2 * JUMP_EVERY + 1])):
        distance = np.linalg.norm(after - before, axis=1)
        assert (distance <= world.reach_cap + 1e-6).all() and (world.reach_cap <= 400).all()
        assert (after >= 40).all() and (after <= world.rect[:, 2:] - 40).all()


def test_camera_view_is_unbounded_and_records_when_the_target_leaves():
    world = SuiteWorld(range(23000, 23004), "camera", steps=200)
    away = world.cursor - (world.goal - world.cursor) * 50          # turn hard away from every target
    for _ in range(40):
        world.step(away)
    assert world.lost.all()
    assert (world.lost_at < 40).all()
    assert (np.abs(world.cursor) > world.rect[:, 2:]).any()         # the view went past the client area
    frozen = world.cursor.copy()
    for _ in range(20):
        world.step(world.goal)                                      # no command reaches a lost target
    np.testing.assert_array_equal(world.cursor, frozen)


def test_teacher_completes_every_task_and_the_envelope_keeps_a_bad_proposer_honest():
    seeds = list(range(20000, 20008))
    results = {task: run_task(task, seeds, "teacher", steps=750) for task in TASKS}
    assert results["settle"]["success"] == 8 and results["settle"]["settling_ms"]["median"] < 3000
    assert results["jump"]["success"] == results["jump"]["reaches"] == 24
    # Two of these plants are slow and delayed; the reference needs over five seconds to catch
    # their targets, so the moving tasks score its limits rather than a perfect run.
    assert results["pursuit"]["success"] >= 6 and results["pursuit"]["within_12px_fraction"] > .7
    assert results["camera"]["success"] >= 6 and results["camera"]["lost"] == 0
    rng = np.random.default_rng(5)

    def noise(senses):
        return rng.uniform(-1, 1, (len(senses["goal"]), 2))

    supervised = run_task("settle", seeds, "supervised", propose=noise, steps=750)
    assert supervised["success"] == 8
    assert 0 < supervised["accepted_share"] < 1
    assert supervised["interventions"]["ticks"] == sum(supervised["interventions"]["per_episode"])
    free = run_task("settle", seeds, "connectome", propose=noise, steps=750)
    assert free["success"] < 8 and free["tracking_error_px"]["mean"] > supervised["tracking_error_px"]["mean"]
    report = {"tasks": {"settle": {"controllers": {"teacher": results["settle"], "supervised": supervised, "connectome": free}},
                        "camera": {"controllers": {"teacher": results["camera"]}}}}
    text = table(report)
    assert "| settle | supervised | 8/8 |" in text and "lost 0" in text


def test_a_perfect_proposer_is_never_overridden():
    class Oracle:
        world = None

        def bind(self, world):
            self.world = world

        def __call__(self, senses):
            world = self.world     # the reference step in units of the intent speed
            return (world.teacher() - world.cursor) / (world.speed[:, None] * .01)

    result = run_task("settle", list(range(20000, 20004)), "supervised", propose=Oracle(), steps=750)
    assert result["interventions"]["ticks"] == 0 and result["accepted_share"] == 1
    assert result["success"] == 4
