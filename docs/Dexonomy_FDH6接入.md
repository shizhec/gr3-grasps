# Dexonomy + FDH-6：离线抓姿合成接入

2026-10-21。已将 `wrap_wide/coffee_mug/0_3_grasp` 接入完整 GR3 右臂与 cuRobo：关节电机执行完成闭手、加压、提升与保持，2 秒保持期最低 COM 提升 15.02 cm。相同轨迹的不闭手对照失败。仍采用已知完整扫描模型和仿真真值位姿，未接入相机或真机。见下文“GR3 整臂执行”及[录像与报告](../outputs/dexonomy_gr3_arm/report.md)。

## v3：GR3 右臂姿态先验与扩大采样（2026-10-21）

**问题。** 上述杯子候选 `0_3` 虽然能提起，但手在杯子内侧，掌心朝外，拇指世界系 z 分量为 −0.76，是拇指朝下的反手包握。原因不在坐标转换：Dexonomy 只针对浮动手合成，手腕方向近似均匀采样；v2 每个物体只有 48 个初始化。另外，上游 `_pose_filter` 在物体位于场景原点时不起作用，上游的筛选还会把被拒绝的样本补回来凑数。执行端只检查 IK 是否可达，最后剩下的唯一候选就是这个反手姿态。

**做法。** `configs/dexonomy_arm_prior.yaml` 在初始化阶段对每个模板匹配结果做**硬筛选**（`install_arm_prior`），然后再进入上游的筛选。上游筛选只会从已经接受的样本里补数。筛选包含两条：

1. 拇指方向的世界系 z 分量 ≥ 0，排除拇指朝下的包握。FDH-6 右手根系的三个轴由指尖位置标定：+X 指向拇指侧，+Y 是掌心法向，−Z 沿手指方向。
2. GR3 右臂可达性地图：`scripts/build_reachability_map.py` 在官方关节限位内对右臂 7 个关节做 3200 万次均匀采样，求正运动学，保留物体附近 30 cm 内的 176 万个手根位姿。候选的抓取位姿和提升 16 cm 后的位姿，都必须在 3 cm / 15° 内找到采样位姿。地图不检查碰撞，碰撞仍由 cuRobo 和 MuJoCo 负责。

采样预算从 256 点 × 8 个旋转角 × 1 轮、每轮保留 16 个，扩大到 1024 × 16 × 4 轮、每轮保留 64 个。合成细化会移动手腕，所以在最终候选上会重新检查一次先验，结果写入 `arm_prior_final.json`。

**淘汰的方案。** 最初手写的"掌心沿伸手方向"锥形约束（`min_approach_alignment: 0.34`、`min_finger_forward: -0.5`），对右手意味着手指要朝外横放。按这个约束合成出 24 个候选，浮动手提升通过 16 个，但 GR3 端点 IK 是 **0/24**。可达性地图与逐个 IK 检查的结果一致：IK 能解的候选，地图距离 ≤ 1；IK 解不出的大多在 1.5 以上。

**结果。** 固定放置位置 [0.46, −0.20]、yaw 0，使用仿真真值位姿注册，协议与上文整臂执行一致。分母是所有通过先验的候选，不做挑选：

| | v2（无先验） | v3 手写锥 | v3 可达性先验 |
|---|---:|---:|---:|
| 杯子 / 糖果盒最终候选 | 5 / 1 | 3 / 21 | 8 / 6 |
| 其中拇指朝上且通过先验 | 0 / 0 | 3 / 18 | 7 / 5 |
| GR3 四个端点 IK 全部可解 | 1（反手 `0_3`） | 0 | 11 / 12 |
| 浮动手提升通过 | 4 / 6 | 16 / 24 | 7 / 14 |
| **GR3 整臂物理执行成功** | 杯子 1（反手） | 未执行 | **杯子 4/6，糖果盒 2/6** |
| 同轨迹不闭手对照 | 失败 | — | 6 个全部失败 |

整臂执行成功的 6 个候选：保持期最低 COM 提升 11.2–14.5 cm，保持 2 s，对向承力接触比例 100%，最大穿透 ≤ 0.76 mm，拇指世界系 z 分量 0.27–0.97。6 个成功案例的手都在物体外侧（−y 侧），掌心法向的 y 分量为 0.60–0.93，即掌心朝向身体中线：wrap 类是拇指朝上的侧向包握，tripod 类是掌心偏向下方的斜上方抓取。另外 6 个失败：3 个 cuRobo 找不到预抓取路径，2 个在接近或提升的笛卡尔路径上 IK 误差略超门限（3.6–3.9 mm），1 个提升时杯子与桌面摩擦，穿透达到 3.1 mm。糖果盒 `0_13` 被提起时有明显倾斜，但满足验收标准。浮动手和整臂的结果不完全一致（例如 `wrap_wide/coffee_mug/1_25` 浮动手失败而整臂成功），说明两者的手腕动力学不同。

**主要剩余瓶颈**是 Dexonomy 预抓取阶段要求的 40 mm 手物间隙：杯子 231 个通过抓取阶段的候选里，有 223 个在这一步被淘汰；糖果盒是 85 个里淘汰 79 个。下一步是自己生成预抓取位姿（手腕沿掌心法向后退）。地图和先验都绑定在当前放置位置；换位置或换 yaw 时，需要修改 `object_xy_world_m` / `object_yaw_deg` 后重新生成地图并重新合成。

```bash
.venv/bin/python scripts/build_reachability_map.py \
  --output outputs/gr3_reachability/right_hand_base.npz --samples 32000000 --jobs 16
MUJOCO_GL=egl .venv-dexonomy/bin/python scripts/run_dexonomy.py all \
  --objects coffee_mug candy_box --output outputs/my_dexonomy_v3 \
  --arm-prior configs/dexonomy_arm_prior.yaml \
  --init-points 1024 --init-inplane 16 --init-final 64 --init-epochs 4
.venv-curobo/bin/python scripts/run_dexonomy_gr3.py --output outputs/my_v3_arm \
  --candidate outputs/my_dexonomy_v3/run/grasp_data/wrap_wide/coffee_mug/<n>_grasp.npy
.venv/bin/python scripts/render_dexonomy_hold.py outputs/my_v3_arm --output outputs/my_v3_arm/hold_views.png
```

本服务器上的数据：合成结果在 `outputs/dexonomy_fdh6_v3_reach`（手写锥方案在 `outputs/dexonomy_fdh6_v3_prior`），浮动手与 IK 审计在 `outputs/dexonomy_v3_reach_audit`，整臂执行在 `outputs/dexonomy_v3_reach_arm`，新旧姿态对比图是 `mug_before_after.png`，四个成功案例的四视角图是 `hold_views.png`。不传 `--arm-prior` 时，v2 的协议签名保持不变。

## v3：掌心法向后退的预抓取（2026-10-22）

**做法。** 加上 `--pregrasp palm_retreat` 后，上游仍然执行张手预抓取，也照常检查关节限位和手指自碰撞，只是把 40 mm 手物间隙的门限设为无效。合成完成后，`apply_palm_retreat` 为每个候选重新生成 `pregrasp_qpos`：

- 手腕姿态保持不变，沿 −掌心法向（FDH-6 −Y）后退。依次尝试后退 8 cm 和 6 cm，方向上可以加 0.5 的向上分量；
- 拇指 yaw 保持抓取时的值，拇指 pitch 和四指弯曲按 1.0 / 0.6 / 0.3 的比例向 0（URDF 张开极限）张开，被动关节按联动关系恢复；
- 远端位姿离物体 ≥ 2 cm；接近路径上 9 个采样点的手物穿透 ≤ 1 mm，比执行端的 3 mm 更严；全程离桌面 ≥ 5 mm；远端腕部位姿必须在 GR3 可达性地图内。

距离用 `mj_geomDistance` 逐对计算手的碰撞几何到物体和桌面的距离。上游的 `get_contacts` 不统计挂在 world body 上的桌面。上游原来的路径保存在 `upstream_pregrasp_qpos` 字段；不满足条件的候选移到 `run/pregrasp_rejected/`，逐个候选的判定记录在 `pregrasp_retreat.json`。

**结果。** 先验、预算、种子与上一节相同。抓取阶段的通过数没有变（杯子 231、糖果盒 85），说明合成过程可以复现。

| | 上游 40 mm 间隙 | 掌心后退 |
|---|---:|---:|
| 杯子：通过预抓取 / 通过最终先验 | 8 / 7 | **224 / 215** |
| 糖果盒：通过预抓取 / 通过最终先验 | 6 / 5 | **44 / 40** |
| 浮动手提升（先验通过者） | 杯子 3/7，糖果盒 3/5 | 杯子 176/215，糖果盒 32/40 |
| GR3 四个端点 IK 全部可解 | 11 / 12 | 杯子 187/215，糖果盒 17/40 |
| **GR3 整臂物理执行成功** | 杯子 4/6，糖果盒 2/6 | **杯子 134/215（62%），糖果盒 6/40（15%）** |
| 同轨迹不闭手对照 | 6 个全部失败 | 140 个全部失败，最高升起 0.01 cm |

整臂执行对全部通过先验的 255 个候选各做了一次，没有重试或挑选。140 个成功案例：保持期最低 COM 提升 12.3–14.8 cm，保持 2 s，对向承力接触比例 ≥ 98%，最大目标穿透 ≤ 1.0 mm。保持阶段拇指世界系 z 分量全部 ≥ 0.04，中位数 0.99。按指尖位置判断，杯子成功案例里只有 1 例（`wrap_medium/coffee_mug/1_2`）是手指伸进杯口的抓法，其余 133 例都是 `wrap_wide` 从外侧握杯身。

115 个失败：60 个 cuRobo 找不到预抓取全局路径；25 个在接近或闭手时把杯子压进桌面，穿透 3.2–3.5 mm，刚超过 3 mm 门限；17 个笛卡尔路径 IK 误差略超门限；9 个提升或保持不足；4 个规划路径碰撞。糖果盒成功率低，主要是 IK 和路径问题（31/34），浮动手提升本身有 32/40 通过。说明可达性地图的 3 cm / 15° 容差对糖果盒偏松，可以收紧容差，或者在合成后加一轮完整 IK 筛选。

这仍是**每个候选一次**的离线结果：真值注册，单一放置位置，yaw 0。实际使用时只需要每个场景有一个候选成功，所以它不等于场景成功率。也不能和原来 30 个场景的 M2T2 批测直接比较。

```bash
MUJOCO_GL=egl .venv-dexonomy/bin/python scripts/run_dexonomy.py all \
  --objects coffee_mug candy_box --output outputs/my_dexonomy_v3_retreat \
  --arm-prior configs/dexonomy_arm_prior.yaml --pregrasp palm_retreat \
  --init-points 1024 --init-inplane 16 --init-final 64 --init-epochs 4
.venv-dexonomy-eval/bin/python scripts/validate_dexonomy_lift.py \
  --output outputs/my_dexonomy_v3_retreat --results-output outputs/my_v3_retreat_audit --jobs 16
```

数据位置：合成结果在 `outputs/dexonomy_fdh6_v3_retreat`，审计在 `outputs/dexonomy_v3_retreat_audit`；整臂执行在 `outputs/dexonomy_v3_retreat_arm`，其中 `summary.json` 是逐候选结果，`open/` 是不闭手对照，`hold_views.png` 是成功案例的四视角图。

## v3 十物体汇总（2026-10-22）

其余 8 种物体使用完全相同的先验、预算、种子和掌心后退预抓取，各物体都放在 `benchmark10.yaml` 的同一位置，yaw 0。先验把场景原点近似为 `placement_xy`；GR3 场景按包围盒中心摆放，两者最大相差 9 mm（锤子），小于地图 3 cm 的容差。IK 审计和整臂规划都使用精确配准。

| 物体 | 先验通过的候选 | 浮动手提升 | GR3 端点 IK | **整臂执行成功** | 不闭手对照 |
|---|---:|---:|---:|---:|---|
| 杯子 | 215 | 176 | 187 | **134（62%）** | 134 个全部失败 |
| 安卓玩偶 | 111 | 94 | 106 | **78（70%）** | 78 个全部失败 |
| 药瓶 | 120 | 105 | 88 | **75（63%）** | 75 个全部失败 |
| 小瓷碗 | 56 | 33 | 45 | **19（34%）** | 19 个全部失败 |
| 蜡笔盒 | 32 | 27 | 18 | **14（44%）** | 14 个全部失败 |
| 糖果盒 | 40 | 32 | 17 | **6（15%）** | 6 个全部失败 |
| 胶带 | 7 | 1 | 7 | 0 | — |
| 锤子 | 4 | 0 | 0 | 0 | — |
| 螺丝刀 | 0 | — | — | 0 | — |
| 鼠标 | 0 | — | — | 0 | — |

10 种物体中 **6 种**至少有一个整臂成功的候选。v2 只有杯子和糖果盒有浮动手候选，且没有一种物体有拇指朝上的整臂成功案例。全部 585 个候选各执行一次，成功 326 个；326 个不闭手对照全部失败，最高升起约 0.01 cm。成功案例的保持期最低 COM 提升 12.3–15.8 cm，对向承力接触比例 ≥ 98%，最大目标穿透 ≤ 2.1 mm。

- **拇指和抓法：** 安卓玩偶和小瓷碗有拇指接近水平的从上方抓取（z 最小 0.01–0.02），小瓷碗主要是从上方捏住碗沿；药瓶的拇指 z ≥ 0.91，是侧握。
- **扁平和细长物体：** 螺丝刀和鼠标（高 3.8 cm）在抓取阶段几乎全部因为缺少要求的手指接触、手碰到桌面或没有接触而被淘汰。鼠标仅有的 2 个候选，后退时离桌面不足 5 mm。锤子只有 4 个候选，全部规划失败。胶带（高 2.7 cm）的 7 个候选在接近时全部把胶带压进桌面，穿透 3.0 mm。
- **失败类型：** 所有物体的物理碰撞失败（共 65 个）都是物体被手压进桌面，穿透刚超过 3 mm；其次是 cuRobo 找不到预抓取全局路径。这两类是下一步提升成功率的主要方向。

这仍是每个候选一次、真值注册、单一放置位置的离线结果，不是场景成功率，也不能直接和原来 30 个场景的 M2T2 批测比较。数据在 `outputs/dexonomy_fdh6_v3_retreat_other8`、`outputs/dexonomy_v3_retreat_other8_audit` 和 `outputs/dexonomy_v3_retreat_other8_arm`（含 `summary.json`、`open/`、`hold_views.png`）。

## v3 六物体多朝向场景测试（2026-10-22）

放弃锤子、胶带、螺丝刀、鼠标，只测 yaw 0 有整臂成功候选的 6 种物体。按原 M2T2 批测的朝向设置，每种物体 0°/45°/90° 共 18 个场景，**每个场景只做一次物理执行**，协议写在 `configs/dexonomy_scenes6.yaml`，运行前固定。

1. **候选来源。** 先验绑定物体朝向，所以 45°/90° 用 `--prior-yaw-deg` 把先验的 `object_yaw_deg` 设为场景朝向后重新合成，预算、种子、掌心后退预抓取与 yaw 0 完全相同；yaw 0 直接复用上文两次合成。可达性地图只依赖放置位置，不需要重建。
2. **候选池。** 在该朝向下通过最终先验，并且通过闭手浮动手提升审计的候选。浮动手审计与朝向无关，不使用任何 GR3 整臂执行结果。
3. **顺序与执行。** 按 max(抓取, 提升后) 可达性地图距离升序，最多尝试 10 个候选的 `--plan-only` 规划（含 MjData 副本里的接触预演）。第一个规划成功的候选用 `--replay-plan` 执行一次；再用同一轨迹、同一初态做不闭手对照。没有规划成功则该场景失败。
4. `run_dexonomy_gr3.py --yaw-deg` 让物体绕桌面法向旋转摆放，仍按包围盒中心放到 [0.46, −0.20]。注册仍用仿真真值位姿；检查过 0°/45°/90° 的注册旋转分别是 Rz(0/45/90)，另有约 0.6° 的落桌倾斜，原点偏差 ≤ 5 mm，与先验的近似一致。

| 物体 | 0° | 45° | 90° | 候选池（0/45/90） |
|---|---|---|---|---|
| 杯子 | 成功（第 1 个） | 成功（第 1 个） | **失败**：前 10 个均无预抓取全局路径 | 176 / 173 / 187 |
| 糖果盒 | 成功（第 8 个） | 成功（第 1 个） | 成功（第 1 个） | 32 / 132 / 296 |
| 安卓玩偶 | 成功（第 1 个） | 成功（第 1 个） | 成功（第 1 个） | 94 / 148 / 141 |
| 药瓶 | 成功（第 3 个） | 成功（第 1 个） | 成功（第 1 个） | 105 / 102 / 124 |
| 小瓷碗 | 成功（第 3 个） | **失败**：提升时 TCP 跟踪误差 15.01 mm | 成功（第 2 个） | 33 / 34 / 44 |
| 蜡笔盒 | 成功（第 1 个） | 成功（第 1 个） | 成功（第 6 个） | 27 / 208 / 239 |

**场景成功率 16/18（89%）。** 18 个不闭手对照全部失败，最高升起 < 0.001 mm。16 个成功场景的保持期最低 COM 提升 13.7–15.3 cm，对向承力接触比例均为 100%，最大目标穿透 ≤ 0.75 mm。

- **两个失败。** 小瓷碗 45° 在提升阶段已把碗抬起 12.4 cm，TCP 跟踪误差 15.01 mm，刚超过 15 mm 门限，按协议中止，计为失败，没有重试。杯子 90° 时，可达性最好的前 10 个候选全部报 `cuRobo pregrasp IK/trajectory/graph search failed`。可达性地图不检查碰撞，这个朝向下手腕可达，但缺少无碰撞的全局路径。
- **盒子转角后候选明显变多。** 糖果盒先验通过数从 0° 的 40 增至 45° 的 140、90° 的 301，蜡笔盒从 32 增至 283、326。转过去后窄面朝向右手，更适合包握。
- **一次一场景的结果比逐候选成功率高**（yaw 0 逐候选 62%/15% 等），因为按可达性顺序的首个可规划候选大多能成功，规划失败也只是换下一个候选，没有执行动作。

这仍是已知网格、真值位姿、固定基座、单一放置位置的仿真结果；没有相机定位。与原 M2T2 30 场景批测（修正后 3/30）的区别是：这里只保留 6 种物体，抓姿来自已知网格的离线合成，而 M2T2 只用单视角点云。所以两者不能当作同一条件下的对比。

```bash
MUJOCO_GL=egl .venv-dexonomy/bin/python scripts/run_dexonomy.py all \
  --objects coffee_mug candy_box android_toy medicine_bottle ramekin crayon_box \
  --output outputs/my_dexonomy_v3_yaw45 --arm-prior configs/dexonomy_arm_prior.yaml \
  --prior-yaw-deg 45 --pregrasp palm_retreat \
  --init-points 1024 --init-inplane 16 --init-final 64 --init-epochs 4
.venv-dexonomy-eval/bin/python scripts/validate_dexonomy_lift.py \
  --output outputs/my_dexonomy_v3_yaw45 --results-output outputs/my_v3_yaw45_audit --jobs 16
# 修改 configs/dexonomy_scenes6.yaml 的 pools 指向新目录；已有 scene.json 的场景会跳过。
.venv/bin/python scripts/benchmark_dexonomy_scenes.py --output outputs/my_scenes6 --jobs 4
```

数据位置：合成结果在 `outputs/dexonomy_fdh6_v3_yaw45`、`outputs/dexonomy_fdh6_v3_yaw90`，审计在 `outputs/dexonomy_v3_yaw45_audit`、`outputs/dexonomy_v3_yaw90_audit`，场景测试在 `outputs/dexonomy_scenes6`（`summary.json`、`results.csv`、每场景 `scene.json`、`planXX__*` 规划尝试、`execute/` 和 `open/`）。`hold_views.png` 是 16 个成功场景的保持终态，按目录字母顺序排列，从安卓玩偶 0° 开始。

## v3 相机定位：头部 RGB-D 估计物体位姿（2026-10-22）

把上一节的真值注册换成**头部相机定位**：在同样 18 个场景上，从头部 RGB-D 估计已知网格物体的位姿，再规划和执行。仿真真值位姿只在最后用于评估误差和抓取结果，不参与选择候选、规划或接触预演。

**感知流程**（`src/gr3_grasping/pose_estimation.py`，`run_dexonomy_gr3.py --registration camera`）：

1. **扫描。** 静止场景下，在手臂动作之前，先用原生相机取初始帧，再用已有的头部扫描（初始视角加 3 个转头视角）分割桌面物体并融合点云，最后头部回到初始姿态。
2. **几何拟合。** 假设物体直立放在已标定的水平桌面上，未知量是桌面内的 x、y 和 yaw。从 36 个 yaw 起点做平面 ICP，用"截断点距离 + 自由空间惩罚"评分。自由空间惩罚统计的是：模型点落在相机已看穿的空间里，即投影深度比实测深度近 1 cm 以上的点所占比例。对最好的 8 个起点直接在评分上做 Nelder–Mead 细化。圆柱形物体用 ICP 会沿柱面滑动，或吸附到杯子内壁，所以必须在评分本身上细化。
3. **颜色消歧。** 以 5° 为步长扫描全部 yaw，在评分最优值 +3 mm 内的 yaw 先细化，再保留 +1 mm 内的作为可行假设。然后按每一帧记录的头部关节角渲染已知带纹理模型，与观测 RGB 比较物体像素的平均颜色误差，取误差最小者，最后在小范围内细化几何。渲染时只用机器人本体关节状态，物体位置由假设覆盖。
4. **规划世界。** 预抓取路径、cuRobo 物体障碍点、MuJoCo 路径检查和接触预演副本，都把物体放在估计位姿上。真实执行里的物体依然是自由刚体。
5. **候选。** 把 0°/45°/90° 三次合成的候选合在一起，在估计的 x、y 和 yaw 下重新检查手臂先验，再用浮动手审计筛选。之后的协议与上一节相同：按可达性距离排序，最多尝试规划 10 个，第一个规划成功的只执行一次，并做同轨迹的不闭手对照。重放时会重新扫描，核对估计与规划时一致，并检查初态逐位一致。

**位姿误差**（`scripts/evaluate_pose_estimation.py`，不动手臂）。下表是加 2 mm 深度噪声和 2% 像素丢失后，18 个场景的误差：

| 版本 | 平移 | yaw | ADD / ADD-S |
|---|---|---|---|
| 仅几何 | ≤ 2.3 mm | 杯子正确；安卓玩偶、糖果盒 45° 翻转 180°；药瓶、小瓷碗任意 | ADD-S ≤ 3.3 mm，ADD 最大 69 mm |
| **几何 + 颜色** | **≤ 2.0 mm** | **全部 ≤ 2.1°** | **ADD ≤ 1.8 mm** |

无噪声时，除小瓷碗外的结果相近。小瓷碗形状对称、纹理很少，两个场景的 yaw 偏差 5° 和 55°，ADD-S 仍 ≤ 1.5 mm，所以几何上等价。

**整臂结果。**

| 注册方式 | 成功场景 | 失败 |
|---|---|---|
| 仿真真值（上一节，按场景朝向取候选） | 16/18 | 杯子 90° 无全局路径；小瓷碗 45° 跟踪误差 15.01 mm |
| 相机、仅几何 | 8/18 | 6 个 yaw 翻转或任意的场景，在估计 yaw 下几乎没有可达候选（0–6 个），规划失败；小瓷碗 3 个场景在错误 yaw 下的抓取把碗压进桌面，穿透 > 3 mm |
| **相机、几何 + 颜色** | **17/18** | 小瓷碗 45°：与真值注册是同一个候选、同样在提升时 TCP 跟踪误差 15.003 mm 中止 |

18 个不闭手对照全部失败，最高升起 < 0.001 mm。17 个成功场景的保持期最低 COM 提升 13.5–15.4 cm，对向承力接触比例 100%，最大目标穿透 ≤ 0.85 mm。杯子 90° 用真值时失败、用相机时成功，原因不在定位：相机方案的候选来自三个朝向，第 10 个候选（来自 yaw 90 的合成）找到了路径。所以 17/18 和 16/18 的候选池不同，不能理解为相机比真值更好。

**仍然成立的局限：**

- 颜色比较用的渲染器与仿真"相机"是同一个，没有光照、白平衡、材质这类域差，比真机乐观。真机需要换成对光照稳健的渲染比较或位姿网络（如 FoundationPose），并在实物上测量误差。
- 相机内参、外参和头部关节读数都是精确的；深度噪声是逐像素独立的高斯噪声加随机丢失，没有模拟真实深度相机的系统偏差、边缘飞点和反光缺失。
- 已知物体网格、单物体、物体直立放在桌面、静止场景、固定基座和单一放置位置。
- 小瓷碗这类对称且缺少纹理的物体 yaw 不可观测。当前结果依赖其几何近似对称；更不对称的物体需要输出等价位姿集合。

```bash
# 位姿误差，不动手臂
.venv/bin/python scripts/evaluate_pose_estimation.py --output outputs/my_pose_eval \
  --depth-noise 0.002 --depth-dropout 0.02
# 18 个场景整臂执行（噪声参数在 configs/dexonomy_scenes6.yaml 的 camera 段）
.venv/bin/python scripts/benchmark_dexonomy_scenes.py --registration camera \
  --output outputs/my_scenes6_camera --jobs 4
# 单个场景
.venv-curobo/bin/python scripts/run_dexonomy_gr3.py --registration camera \
  --depth-noise 0.002 --depth-dropout 0.02 --yaw-deg 45 \
  --candidate outputs/dexonomy_fdh6_v3_yaw45/run/grasp_data/tripod/crayon_box/2_29_grasp.npy \
  --output outputs/my_camera_crayon45
```

数据位置：

- 位姿误差：`outputs/pose_estimation_color_noise2mm`、`outputs/pose_estimation_color_clean`，以及仅几何版本 `outputs/pose_estimation_noise2mm`、`outputs/pose_estimation_clean`。
- 整臂执行：`outputs/dexonomy_scenes6_camera`。每个场景的 `perceive/` 存放头部扫描、`pose_estimate.json` 和 `camera_registration.json`（其中 `evaluation_only_*` 字段仅用于评估）；`hold_views.png` 是 17 个成功场景的保持终态。
- 仅几何的对照：`outputs/dexonomy_scenes6_camera_geometry_only`。

此前动态修正已验证：对原六个浮动手腕候选，采用 MuJoCo 3.16.0 的 discrete 积分器，并将手腕伺服时间常数设为 4 ms 后，4 个通过提升至少 10 cm/保持 2 s 的验收，覆盖糖果盒和杯子。六个不闭手对照全部失败，0.25 ms 步长复核的六个结果与 0.5 ms 完全一致。加入的手腕跟踪误差阈值为 15 mm/0.10 rad。

原十物体开发实验共有 480 个初始化，102 个通过抓姿接触/静力筛选，6 个进一步通过预抓取终态筛选；另外八类物体还没有完整候选。4/6 是筛选后离线候选的结果，不能当作十物体、原有 30 场景或真机成功率。

最新的因果对照、步长复核与端点 IK 见 [动态修正报告](../outputs/dexonomy_dynamics_audit/report.md)。下面保留首次接入与旧积分器的结果供对照。

首次接入统计见 [原报告](../outputs/dexonomy_fdh6_v2/report.md)、[原始合成统计](../outputs/dexonomy_fdh6_v2/summary.json)和[旧积分器执行结果](../outputs/dexonomy_fdh6_v2/lift_summary_noslip0_dt500us.json)。

## 方法与接入范围

依据 [Dexonomy 官方源码](https://github.com/JYChen18/Dexonomy/tree/d561ff9df8545deef7a3f9e7cfb4045ef66f6913)和[自定义手教程](https://github.com/JYChen18/Dexonomy/blob/d561ff9df8545deef7a3f9e7cfb4045ef66f6913/getting_started/1_annotation_tutorial.md)。使用官方模板转换、GPU 接触模板匹配、MuJoCo 弹簧细化、接触与 QP 筛选以及原版动态力闭合验证；不需要训练权重。没有改写第三方检出源码。

合成阶段验证完整物体 mesh 条件下的 FDH-6 浮动手腕抓姿，后续增加整臂执行分支。两者尚未接收原生头部 RGB-D。输入完整 CAD/扫描几何属于离线信息，不能当作实际相机观测。

输出包含手腕六维位姿、六主动手指角及 pregrasp/grasp/squeeze。规划接入时，还需要场景与机器人坐标注册、带手指预成形的碰撞检查，以及闭手/提起时的接触控制。

## FDH-6 适配

- 从官方右手 URDF 导出独立 MJCF，保留 6 个电机和 5 个 joint equality。拇指末节为主动 pitch 的 1.06 倍；其余四指中节为主动近节的 0.975 倍。
- 使用三种 CAD 指腹接触种子：三指对向、较小包络、较大包络。种子配置在 `configs/dexonomy_fdh6.yaml`；接触点从碰撞凸包表面计算，法向取实际表面法向。它们是开发种子，尚非经验证的专家抓取模板。
- 要求拇指及至少一个其他被标注手指接触。其余接触由几何、碰撞和 QP 再筛选。
- 原版优化使用人工弹簧力与软等式，会让本模型的被动关节偏离联动关系。因此默认 `coupled` 模式仅在**合成优化**中将每个迭代步投影到六主动关节的限位内，再恢复五个联动角，并清除指关节速度。它是原版方法上的适配，不是原版算法的原样复现。
- 挤压目标通过虚功关系 `tau_active = C.T @ tau_all` 汇总被动关节载荷；再生成六电机目标并恢复联动角。`C` 是 11×6 的联动雅可比。
- 保留此前仿真的 kp=2、kv=0.08、每电机 ±0.25 Nm。这些是仿真假设，未做真机辨识。未采用官方教程建议的更高力矩。

原始标注、碰撞骨架、分组和完整手模型在 `outputs/dexonomy_fdh6_v2/assets/hand/fdh6/`。`annotation_audit.json` 记录接触点、法向与表面距离。

## 扫描物体导入的两个修复

使用现有十个 Google Scanned Objects 资产，保留视觉网格、纹理和原始凸分解碰撞。物体尺度不变，质量使用 `configs/benchmark10.yaml` 的仿真假设。

1. 给所有物体 geom 显式命名。原始 GSO XML 多数 geom 无名，而 Dexonomy 的 `set_obj_margin` 依赖 `obj-` 名称前缀。若不修复，45 mm 的初始张手 margin 不会被缩小，厘米级间隙也会进入接触/QP 筛选。本轮保存的抓姿接触距离已在 2 mm 内。
2. 将视觉 geom 质量显式设为零。否则视觉网格会额外参与质量推断，而原版仅改变碰撞 geom 的密度，目标质量不能正确生效。适配测试和执行器均核对最终编译后的质量。

早期 `outputs/dexonomy_fdh6/`、`dexonomy_fdh6_coupled/`、`dexonomy_fdh6_v1/` 是修复前探索，不能用于报告性能。v2 为修复后的版本。

## 首次动态验证与失败

原版动态力闭合：无重力，完成张手/闭手/挤压后，分别对物体施加六方向各等于物体重力的外力，每方向测试至多 2 s，允许位移 5 cm、旋转 15°。该测试与提升 10 cm 的标准不同。

本地另做真实重力提起：手腕按目标受控，物体保持自由体，通过接触承载；闭手后手腕升高 16 cm、保持 2 s，检查物体 COM 全程提升至少 10 cm、拇指与其他手指承载比例至少 90%、保持期间无桌面接触、穿透不超过 3 mm、联动误差不超过 0.01 rad。此时没有合成弹簧力、关节投影或物体绑定。逐物理步检查，50 Hz 保存摘要。

本地默认 0.5 ms 步长和 `noslip_iterations=0`；摩擦系数 1.0、扭转摩擦 0.005、condim=4、impratio=10。它继承 Dexonomy 浮动手腕与接触参数，仍不同于原有全 GR3 执行模型。求解器设置和积分步长写入每次结果，文件按设置分别保存。

首次六次候选执行最先都触发联动误差门限，发生在接近、闭手或挤压阶段，尚未进入提起。原版 no-slip 修正开关及步长缩小未消除该问题。后续同引擎积分器对照已定位并修正这一数值问题，见下一节。

另有 96 个静态抓姿未达到预抓取终态要求的 40 mm 手物安全间隙；这是正距离阈值，不能全部解释成实际发生穿透。预抓取终态通过也不表示整段接近路径已无碰撞。

每个物体允许验证多个合成候选；这不是原先三十场景、每场景单次物理尝试的协议。三种模板与十个物体均用于当前开发，不构成保留测试集。没有给出新成功率提升结论。

## 动态修正：同引擎积分器对照

新建 `.venv-dexonomy-eval`，只运行物理执行，不改变原版 GPU 合成环境。该环境明确使用 MuJoCo 3.16.0，并只安装 Dexonomy 的模拟/场景读取模块所需依赖；不是完整的上游训练/合成依赖环境。

同一版本、同一批候选、同样的物体参数和电机 kp/kv/限幅，`implicitfast` 六次仍最先触发联动误差；`discrete` 不再触发，最大联动误差约 0.000104 rad。`discrete` 对阻尼/刚度的隐式更新与约束使用同一有效惯量求解；机理说明见 [MuJoCo 3.16 计算文档](https://github.com/google-deepmind/mujoco/blob/3.16.0/doc/computation/index.rst)。对该模型的对照支持积分处理是这次联动误差的主要来源，不是仅靠升级版本消除问题。

另行发现原版 20 ms 浮动手腕伺服过软，部分候选旋转跟踪偏差超过 0.10 rad。将其设为 4 ms 后，在所有正式积分器对照中保持一致，并逐步检查手腕跟踪。它依然是理想受控手腕的仿真假设，不是 GR3 实际腕部刚度标定。

闭手验证没有进行任何 qpos 投影、没有物体绑定或合成外力；保留 ±0.25 Nm 电机限幅及此前物理验收。结果如下：

| 候选 | 结果 | 保持阶段最低 COM 提升 |
|---|---|---:|
| 糖果盒 wrap_medium/0_5 | 通过 | 14.45 cm |
| 杯子 wrap_wide/0_3 | 通过 | 15.69 cm |
| 杯子 wrap_wide/0_5 | 通过 | 14.88 cm |
| 杯子 wrap_wide/0_8 | 通过 | 14.73 cm |
| 杯子 tripod/0_1 | 失败，未持续提起 | 未达标 |
| 杯子 tripod/0_6 | 失败，提升不足 | 8.85 cm |

六个不闭手对照全部失败；0.25 ms 积分步长复核六个结果全部一致。糖果盒和杯子成功视频分别在动态审计目录的 `lift_validation_mj3.16.0_discrete_wrist4000us_noslip0_dt500us/` 中。

固定基座/腰部、物体 xy=[0.46,-0.20] m 的 GR3 右臂端点检查中，只有 `wrap_wide/coffee_mug/0_3_grasp` 的预抓取/抓取/挤压/提升端点全部在四个初值的有界 IK 搜索中达标，且未触发当前端点碰撞检查。其余候选至少一个端点未解到阈值内，不能证明其全局不可达。此检查使用离线真值注册与 SciPy IK，没有运行 cuRobo 连续路径规划或全臂物理执行。

## 复现

独立环境避免改变已有 MuJoCo/cuRobo/M2T2 依赖。以下命令均在项目根目录执行：

```bash
bash scripts/setup_dexonomy.sh
bash scripts/setup_dexonomy_eval.sh

# 使用新目录可从头复现；已有目录会检查协议/输入哈希。
MUJOCO_GL=egl .venv-dexonomy/bin/python scripts/run_dexonomy.py all \
  --objects all --output outputs/dexonomy_reproduce

bash scripts/run_dexonomy_dynamics_audit.sh \
  outputs/dexonomy_reproduce outputs/dexonomy_reproduce_dynamics
```

`run_dexonomy.py` 支持独立 `prepare`、`init`、`grasp`、`eval` 阶段；默认处理药瓶，`--objects all` 处理十物体。`--templates` 限制初始化模板。`--refinement upstream` 可禁用六主动关节投影作对照，须使用新输出目录。

物理验证使用 `.venv-dexonomy-eval/bin/python scripts/validate_dexonomy_lift.py`。默认 discrete、0.5 ms、手腕时间常数 4 ms。`--no-close` 是保持初始张手配置的对照；`--integrator implicitfast`、`--timestep .00025`、`--wrist-timeconstant .02` 分别控制积分方法、精度与手腕伺服的对照。每种配置有独立文件名；`--object` 筛选时汇总文件也有独立后缀，避免覆盖全体结果。`--results-output` 可将审计结果与原始候选分开保存。

产物包括：

- `experiment.json`：固定提交、协议、输入哈希和依赖版本；`environment.txt` 保存本服务器依赖快照。
- `run/init_data/`、`run/grasp_data/`、`run/succ_grasp/`：原版阶段格式。初始化筛选可能补回被拒绝样本，不能把“Get valid”日志理解为可执行抓取。
- `diagnostics_*.json`：每个最终筛选决策、接触连杆、限位与联动误差。
- `grasp_commands.json`：显式具名六电机目标及 `T_scene_hand_base`，手腕四元数原始顺序为 wxyz，导出为 4×4 矩阵。它不调用 GR3 驱动。
- `lift_summary_mj*`、对应 `lift_validation_mj*/`：动态审计目录内的物理结果、关节诊断、终止状态和视频，文件名包含引擎、积分器、手腕伺服、步长等设置。旧名称文件保留首次实验。视频是独立手部仿真，不能当作 GR3 执行录像。

当时主环境完整回归测试 39 项通过、7 项跳过；适配测试 5 项在独立 Dexonomy 环境中全部通过。主环境跳过项包括三个 cuRobo 环境测试、一个需要 Dexonomy 的运行时 margin 测试，以及三个需要独立动态环境的测试。测试覆盖六电机/五联动、限位、被动载荷的虚功映射、最终物体质量、运行时 margin 更新、手腕坐标与具名关节导出。

动态修正后，`.venv-dexonomy-eval` 中适配与动态回归合计 8 项通过。动态回归从独立的冻结糖果盒抓姿 fixture 和原始模型重新构建临时场景，覆盖闭手提起、不闭手对照，以及仅更换为 implicitfast 的失败对照。

## GR3 整臂执行

`configs/dexonomy_gr3_mug.yaml` 与 `scripts/run_dexonomy_gr3.py` 新增实验分支，不修改此前 M2T2 全臂模型默认参数。仅该分支采用 discrete/0.5 ms 和经过独立手部验证的接触参数，并将杯子惯量恢复为合成时的碰撞网格惯量，质量 0.30 kg。

1. 手指在机械臂初始位置由六个电机预成形。用仿真物体位姿注册离线场景到 GR3 世界系，依据官方装配链直接转换手根与 TCP，不再使用夹爪的经验偏移。
2. cuRobo 生成全局预抓取路径，随后在笛卡尔路径采样处求 GPU IK，只有 GPU 有效解才作有界 SciPy 细化。最大关节插值步长 0.02 rad；手臂速度上限 0.35 rad/s、手指目标速度上限 0.25 rad/s。GPU 碰撞代理冻结预成形手型，MuJoCo 对变化手型重检。
3. 加压目标是电机设定值，不能直接作为实际几何，否则会人为把手指放入物体。先在独立 MjData 副本中通过同样电机执行接近、闭合、加压，检查实际接触与误差，再用实际接触手型和手物相对位姿检查提升规划。副本里的随手物体几何只服务于规划；执行物体始终自由。
4. 真实执行只写七个右臂和六个手部电机的 `ctrl`；没有运行时 qpos 投影、浮动手腕或物体绑定。手掌及手指可接触目标，穿透最多 3 mm，其他运动臂碰撞最多 1 mm。物理步进逐步检查联动误差、穿透和接触阶段的 TCP 跟踪，50 Hz 保存日志和状态，视频 15 fps。

该候选的结果为保持最低 COM 提升 **15.0246 cm**、保持 2 s、对向承力比例 100%、最大目标穿透 **0.4817 mm**、最大联动误差 **0.0000536 rad**。开手对照最高仅 0.0029 cm，杯子保持桌面支撑。两次模型、初始物理状态和全部手臂目标一致。

自由空间预抓取的动态跟踪最大为 19.9 mm / 0.121 rad，该阶段静置 0.5 s 后验收（末端约 0.30 mm / 0.0017 rad）；接近、闭合、加压、提升和保持则逐物理步使用 15 mm / 0.10 rad 门限。提升阶段最大平移误差约 7.99 mm。基座和腰部固定，全机器人理想重力补偿，手指 ±0.25 Nm 限幅和增益仍为仿真假设。

```bash
# 本服务器已有候选；新服务器先完成上方 v2 合成并修改配置中的候选/手元数据路径。
.venv-curobo/bin/python scripts/run_dexonomy_gr3.py \
  --output outputs/my_dexonomy_gr3 --video
# 使用完全相同轨迹；核对模型/候选哈希和初态，绝不重置物理状态。
.venv-curobo/bin/python scripts/run_dexonomy_gr3.py \
  --output outputs/my_dexonomy_gr3_open --replay-plan outputs/my_dexonomy_gr3 \
  --no-close --video
# 本服务器的闭手/开手目录与规划来源生成汇总和 COM 图。
.venv-curobo/bin/python scripts/report_dexonomy_gr3.py \
  --root outputs/dexonomy_gr3_arm --planning dev03
```

`--plan-only` 会执行预成形；新规划时还执行副本里的接触预演，复用规划时跳过预演。它不执行正式提升，始终不报告抓取成功。`--replay-plan` 无须启动 GPU 规划器，仍须模型和仿真依赖；不同初态或模型会拒绝复用。保留 XML 哈希用于审计，新结果还使用编译后 MjModel（包含网格和物理参数）哈希校验，避免仅因 XML 自动补充 MIME 属性而误拒绝相同模型。每次输出目录必须不存在，保留开发失败试验。

新增加的六项测试覆盖整手接触允许范围、物体碰机身/桌面拒绝、四元数插值、完整 GR3 冻结 cuRobo 轨迹的闭手/不闭手物理回归，以及命令行复用、输出和模型改变拒绝。回归从冻结抓姿和原始资产重新构建临时模型，不依赖 `outputs` 历史文件，且没有声称重新运行 GPU 规划。另在 cuRobo 环境通过五项 GPU/导出测试。

下一步是用头部多视角点云估计杯子位姿，将真值注册替换为观测注册，再评估注册误差对抓取的影响；另外八类物体仍需补充模板和可行预抓取路径。
