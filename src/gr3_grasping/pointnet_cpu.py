"""Inference-only CPU reference operators for M2T2's PointNet++ extension.

These implement the upstream CUDA operator semantics, including ball-query
input ordering and padding. FPS ties may select different equivalent points.
They do not replace the learned network or generate synthetic grasps.
"""
import sys
import types

import torch


def furthest_point_sampling(xyz, npoint):
    batch, count, _ = xyz.shape
    result = torch.zeros((batch, npoint), dtype=torch.int32, device=xyz.device)
    nearest = torch.full((batch, count), 1e10, device=xyz.device)
    valid = xyz.square().sum(-1) > 1e-3  # Same origin exclusion as upstream.
    current = torch.zeros(batch, dtype=torch.long, device=xyz.device)
    rows = torch.arange(batch, device=xyz.device)
    for index in range(npoint):
        result[:, index] = current.int()
        dist = (xyz - xyz[rows, current, None]).square().sum(-1)
        nearest = torch.minimum(nearest, dist)
        current = nearest.masked_fill(~valid, -1).argmax(-1)
    return result


def gather_points(features, indices):
    return features.gather(2, indices.long()[:, None].expand(-1, features.shape[1], -1))


def group_points(features, indices):
    batch, channels, _ = features.shape
    return gather_points(features, indices.reshape(batch, -1)).reshape(batch, channels, *indices.shape[1:])


def three_nn(unknown, known):
    distances, indices = [], []
    for chunk in unknown.split(256, dim=1):
        d = torch.cdist(chunk, known, compute_mode="donot_use_mm_for_euclid_dist").square()
        values, idx = d.topk(min(3, known.shape[1]), dim=-1, largest=False, sorted=True)
        if known.shape[1] < 3:
            # CUDA initializes missing nearest neighbors to 1e40, index zero.
            pad = 3 - known.shape[1]
            values = torch.cat([values, values.new_full((*values.shape[:-1], pad), float("inf"))], -1)
            idx = torch.cat([idx, idx.new_zeros((*idx.shape[:-1], pad))], -1)
        distances.append(values)
        indices.append(idx.int())
    return torch.cat(distances, 1), torch.cat(indices, 1)


def three_interpolate(features, indices, weights):
    return (group_points(features, indices) * weights[:, None]).sum(-1)


def ball_query(new_xyz, xyz, radius, nsample):
    count = xyz.shape[1]
    point_ids = torch.arange(count, device=xyz.device)[None, None]
    result = []
    for chunk in new_xyz.split(128, dim=1):
        dist = torch.cdist(chunk, xyz, compute_mode="donot_use_mm_for_euclid_dist")
        candidates = point_ids.expand(*dist.shape).masked_fill(dist >= radius, count)
        indices = candidates.topk(min(nsample, count), dim=-1, largest=False, sorted=True).values
        first = indices[..., :1].clamp(max=count - 1)
        # With an empty ball the CUDA output is all zero.
        first = torch.where(indices[..., :1] == count, 0, first)
        indices = torch.where(indices == count, first, indices)
        if count < nsample:
            indices = torch.cat([indices, first.expand(*first.shape[:-1], nsample - count)], -1)
        result.append(indices.int())
    return torch.cat(result, 1)


def install():
    """Install explicitly before importing upstream M2T2; this is CPU-only."""
    existing = sys.modules.get("pointnet2_ops")
    if existing is not None:
        if getattr(existing, "gr3_cpu_reference", False):
            return
        raise RuntimeError("pointnet2_ops already loaded; use a separate process for the CPU reference backend")
    package = types.ModuleType("pointnet2_ops")
    extension = types.ModuleType("pointnet2_ops._ext")
    for name in ("furthest_point_sampling", "gather_points", "group_points", "three_nn",
                 "three_interpolate", "ball_query"):
        setattr(extension, name, globals()[name])
    package._ext = extension
    package.gr3_cpu_reference = True
    sys.modules["pointnet2_ops"] = package
    sys.modules["pointnet2_ops._ext"] = extension
