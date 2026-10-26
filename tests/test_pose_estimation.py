import numpy as np

from gr3_grasping.geometry import RGBDFrame, transform, transform_points
from gr3_grasping.pose_estimation import estimate_upright_pose, free_space_violation, planar_pose, pose_errors


def l_shape(step=0.003):
    """Asymmetric upright body: a 6x2 cm slab with a 2x4 cm wing, 5 cm tall, origin at its base."""
    g = np.arange(0, 0.05 + 1e-9, step)
    points = []
    for x0, x1, y0, y1 in ((0, .06, 0, .02), (0, .02, .02, .06)):
        xs, ys = np.arange(x0, x1 + 1e-9, step), np.arange(y0, y1 + 1e-9, step)
        for z in g:
            points += [[x, y, z] for x in xs for y in (y0, y1)] + [[x, y, z] for x in (x0, x1) for y in ys]
        points += [[x, y, .05] for x in xs for y in ys]
    return np.array(points) - [.02, .02, 0]


def test_planar_fit_recovers_asymmetric_pose_from_partial_view():
    body = l_shape()
    T_true = planar_pose(np.deg2rad(37), [.5, -.2], 1.0)
    world = transform_points(T_true, body)
    # Partial observation: only surfaces facing a camera at negative x, nothing below 9 mm.
    seen = world[(world[:, 0] < T_true[0, 3] + .01) & (world[:, 2] > 1.009)]
    rng = np.random.default_rng(3)
    seen = seen + rng.normal(0, .001, seen.shape)
    T, diagnostics = estimate_upright_pose(seen, body, 1.0, [])
    errors = pose_errors(T, T_true, body)
    assert errors["translation_xy_m"] < .003
    assert abs(errors["yaw_error_deg"]) < 3
    assert diagnostics["best"]["free_space_violation"] == 0


def test_free_space_violation_counts_model_points_in_seen_through_space():
    K = np.array([[100., 0, 50], [0, 100, 50], [0, 0, 1]])
    # Camera at the origin looking along +z sees a wall 1 m away everywhere.
    frame = RGBDFrame(np.zeros((101, 101, 3), np.uint8), np.ones((101, 101), np.float32), K, np.eye(4), 0., "test")
    points = np.array([[0, 0, .5], [0, 0, 1.0], [0, 0, 1.5]])
    assert np.isclose(free_space_violation(transform(), points, [frame], .01), 1 / 3)
