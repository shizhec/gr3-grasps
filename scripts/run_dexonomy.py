#!/usr/bin/env python3
"""Run the pinned official Dexonomy pipeline on FDH-6 and local GSO meshes."""
import argparse
import json
import logging
from pathlib import Path
import random
import time

import numpy as np
import yaml

from gr3_grasping.dexonomy_backend import (
    ROOT, DEXONOMY_COMMIT, annotate_templates, configuration, export_hand, export_object,
    audited_grasp, record_experiment, export_grasp_commands, DEFAULT_BUDGET,
    load_arm_prior, install_arm_prior, arm_prior_metrics, arm_prior_mask,
    PALM_RETREAT, apply_palm_retreat,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "init", "grasp", "eval", "all"])
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dexonomy_fdh6_v2")
    parser.add_argument("--objects", nargs="+", default=["medicine_bottle"])
    parser.add_argument("--templates", nargs="+")
    parser.add_argument("--refinement", choices=["upstream", "coupled"], default="coupled")
    parser.add_argument("--arm-prior", type=Path, help="Hard GR3 arm pose prior, e.g. configs/dexonomy_arm_prior.yaml")
    parser.add_argument("--prior-yaw-deg", type=float,
                        help="Override the prior's object_yaw_deg (object rotation about the table normal)")
    parser.add_argument("--init-points", type=int, default=DEFAULT_BUDGET["n_init_point"])
    parser.add_argument("--init-inplane", type=int, default=DEFAULT_BUDGET["n_init_inplane"])
    parser.add_argument("--init-final", type=int, default=DEFAULT_BUDGET["n_final_per_template"],
                        help="Kept initializations per template, object and epoch")
    parser.add_argument("--init-epochs", type=int, default=DEFAULT_BUDGET["epoch"])
    parser.add_argument("--pregrasp", choices=["upstream", "palm_retreat"], default="upstream",
                        help="palm_retreat replaces upstream's 40 mm whole-hand pregrasp clearance")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    budget = dict(n_init_point=args.init_points, n_init_inplane=args.init_inplane,
                  n_final_per_template=args.init_final, epoch=args.init_epochs)
    prior = load_arm_prior(args.arm_prior) if args.arm_prior else None
    if args.prior_yaw_deg is not None:
        if prior is None:
            parser.error("--prior-yaw-deg requires --arm-prior")
        prior["object_yaw_deg"] = args.prior_yaw_deg
    pregrasp = dict(method="palm_retreat", **PALM_RETREAT) if args.pregrasp == "palm_retreat" else None
    record_experiment(args.output, args.refinement, budget, prior, pregrasp)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
        logging.FileHandler(args.output / f"{args.stage}.log")],
        format="%(asctime)s %(levelname)s %(message)s")
    np.random.seed(12)
    random.seed(12)
    import torch
    torch.manual_seed(12)
    items = yaml.safe_load((ROOT / "configs/benchmark10.yaml").read_text())["objects"]
    if args.objects != ["all"]:
        unknown = set(args.objects) - {i["id"] for i in items}
        if unknown:
            parser.error(f"Unknown objects: {unknown}")
        items = [i for i in items if i["id"] in args.objects]
    stages = ["prepare", "init", "grasp", "eval"] if args.stage == "all" else [args.stage]
    if "prepare" in stages:
        hand_path, meta = export_hand(args.output / "assets/hand/fdh6")
        annotate_templates(hand_path, meta)
        from dexonomy.op.convert_template import operate_tmpl
        operate_tmpl(configuration(args.output, "tmpl"))
        expected = set(yaml.safe_load((ROOT / "configs/dexonomy_fdh6.yaml").read_text())["templates"])
        produced = {p.stem for p in (hand_path.parent / "init_tmpl").glob("*.npy")}
        if produced != expected:
            raise RuntimeError(f"Template conversion incomplete: {produced} vs {expected}")
        for item in items:
            export_object(item, args.output / "assets/object" / item["id"])
    if "init" in stages:
        from dexonomy.op.gen_init import operate_init
        templates = args.templates or sorted(p.stem for p in (args.output / "assets/hand/fdh6/init_tmpl").glob("*.npy"))
        if not templates:
            raise RuntimeError("No templates; run prepare first")
        prior_log = {}
        for item in items:
            for template in templates:
                scene = args.output / "assets/object" / item["id"] / "scene.npy"
                cfg = configuration(args.output, "init", str(scene), item["mass_kg"], budget)
                cfg.tmpl_name = template
                records = prior_log.setdefault(f"{item['id']}/{template}", [])
                restore = install_arm_prior(prior, records) if prior else None
                try:
                    operate_init(cfg)
                finally:
                    if restore:
                        restore()
                files = list((Path(cfg.init_dir) / template / item["id"]).glob("*.npy"))
                if not files and not (prior and records and sum(r["accepted"] for r in records) == 0):
                    raise RuntimeError(f"No initialization output: {item['id']} / {template}; inspect log")
        if prior:
            (args.output / "arm_prior_init.json").write_text(json.dumps(prior_log, indent=2) + "\n")
    for stage in ("grasp", "eval"):
        if stage not in stages:
            continue
        from dexonomy.op.gen_grasp import operate_grasp
        from dexonomy.op.eval_data import operate_eval
        for item in items:
            cfg = configuration(args.output, stage, mass=item["mass_kg"])
            cfg.debug_name = item["id"]
            Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
            if stage == "grasp":
                if pregrasp:
                    # Upstream still opens the hand and checks limits/self
                    # collision; its object clearance is replaced below.
                    cfg.op.pregrasp.filter.collision.ho_thre = -1.0
                audited_grasp(cfg, args.refinement)
                if pregrasp:
                    apply_palm_retreat(args.output, [item["id"]], prior)
            else:
                (operate_grasp if stage == "grasp" else operate_eval)(cfg)
    counts = {}
    for item in items:
        counts[item["id"]] = {kind: len(list((args.output / "run" / kind).glob(f"*/{item['id']}/*.npy")))
                             for kind in ("init_data", "grasp_data", "succ_grasp", "pregrasp_rejected")}
    if prior:
        candidates = {}
        for path in sorted((args.output / "run/grasp_data").glob("*/*/*.npy")):  # after pregrasp rejection
            # Refinement moves the wrist; re-check the prior on the final grasp.
            q = np.load(path, allow_pickle=True).item()["grasp_qpos"][-1][:7]
            candidates[str(path.relative_to(args.output / "run/grasp_data"))] = dict(
                accepted=bool(arm_prior_mask(q, prior)[0]),
                **{k: float(v[0]) for k, v in arm_prior_metrics(q, prior).items()})
        (args.output / "arm_prior_final.json").write_text(json.dumps(candidates, indent=2) + "\n")
        for item in items:
            counts[item["id"]]["grasp_data_prior_accepted"] = sum(
                v["accepted"] for k, v in candidates.items() if k.split("/")[1] == item["id"])
    report = dict(dexonomy_commit=DEXONOMY_COMMIT, counts=counts, budget=budget, arm_prior=prior, pregrasp=pregrasp,
        protocol="Official offline mesh-based floating-hand force-closure experiment; not GR3 RGB-D arm grasp benchmark",
        object_mass_source="configs/benchmark10.yaml; simulation assumption", seed=12,
        refinement=args.refinement,
        note="InitFilter pads rejected candidates; init_data count is not a valid-grasp count",
        created_unix_s=time.time())
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    if (args.output / "assets/hand/fdh6/hand_meta.json").exists():
        export_grasp_commands(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
