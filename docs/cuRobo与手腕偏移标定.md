# cuRobo 路径规划与 FDH-6 手腕偏移标定

本轮把两种改动分开评估：cuRobo 改变运动求解与预抓取路径；偏移标定改变 M2T2 接触中心到 FDH-6 手腕的刚体变换。两者都继续使用 MuJoCo 执行物理接触、检查碰撞及判定抓取成功。

## cuRobo 环境与模型

使用 NVIDIA 官方 cuRobo V2，固定源码提交 `78fd485fa82d9b9a063fb4985e371814587e666a`。V2 与 0.7.x 的 API 不同。官方源码及教程：

- https://github.com/NVlabs/curobo
- https://github.com/NVlabs/curobo/blob/78fd485fa82d9b9a063fb4985e371814587e666a/curobo/examples/getting_started/motion_planning.py

这台服务器实测为 RTX 5090 / 32 GB。原 `.venv` 使用 CPU PyTorch；新的 `.venv-curobo` 使用 PyTorch 2.8.0+cu129，保留两套环境。

```bash
cd gr3_grasping
bash scripts/setup_curobo.sh
.venv-curobo/bin/gr3-grasp simulate --planner curobo --no-video --output outputs/curobo_demo
```

模型从同一份官方 GR3 + FDH-6 URDF 创建：固定底座、腰部、另一条手臂及手指在当前测量姿态，只优化选定手臂的七个关节；TCP 位姿与 MuJoCo 的 site 定义一致。启动时核对正解，位置差超过 10 微米或方向差超过 0.0001 rad 就报错。

cuRobo 用碰撞球近似机器人几何，使用桌面和观测目标的包围盒搜索预抓取路径。目标包围盒只在预抓取阶段避让，允许接近阶段建立手指接触。32 个 IK 初值、4 条轨迹初值、PRM 图搜索和轨迹优化启用；CUDA graph 保持开启。

接近与提起仍按直线采样：逐点使用 cuRobo IK，再用 MuJoCo 检查关节插值。预抓取轨迹也经过 MuJoCo 几何逐段复查，关节间隔不超过 0.02 rad，速度限制仍为 0.35 rad/s。执行阶段的碰撞、跟踪误差、手指接触和穿透检查继续生效。不存在 GPU 不可用时悄悄退回 SciPy 的行为。

碰撞球采用保守的几何包围近似，可能拒绝实际可行的狭窄路径；本轮仍是单物体桌面任务。尚未完成用融合深度构建复杂障碍地图、闭手后的实时重规划或携带物体的碰撞体更新。

## 偏移的定义

代码已有两层变换：`robot.tcp_offset_m` 定义手腕附着的 TCP；`m2t2.fdh6_contact_offset_m` 定义接触中心相对这个 TCP 的位置。本轮只标定第二层，TCP 定义保持一致。

```text
p_world_tcp = p_world_source_contact - R_world_tcp @ contact_offset_tcp_m
```

原偏移是 `[-0.012, 0, -0.045]` 米。偏移增加 `delta`，实际手腕目标平移 `-R_world_tcp @ delta`；数值在 TCP 坐标系内。两种夹持对称方向分别按自身旋转变换；M2T2 置信度、候选顺序、朝向和接近方向保持一致。

## 校准与验证

```bash
.venv/bin/python scripts/calibrate_wrist_offset.py --workers 4
```

脚本读取 `outputs/benchmark10_head_scan_context` 中已经保存的点云和 M2T2 输出，不重新运行网络。物理上复现保存的头部伺服命令，并保存完整 MuJoCo 积分状态；每组偏移从完全相同的状态开始。

- 校准：十个物体的 0°，共 10 个场景。
- 粗搜索：原偏移及三个轴各 ±10 mm，共 7 组。
- 细搜索：粗搜索最优点的三个轴各 ±5 mm，再 6 组。
- 选择：优先严格成功次数，其次保持阶段对向承重接触、截断后的保持高度，最后偏向更小的修正。
- 验证：45°、90° 的 20 个场景；校准参数写出并冻结后才运行。
- 所有组保持 SciPy 规划、相同闭手动作，每场景只执行首个通过完整规划的候选一次，执行失败后不重试。

候选排序保持一致，但改变偏移可能改变哪些候选能通过完整规划，因此实际选中的候选编号可能变化。这个实验衡量统一偏移对整条流水线的影响；不能把所有改善都解释为同一候选在闭手时的几何修正。

每个物体都出现在校准和验证中，因此这是未用于调参的朝向验证，不是未见物体泛化测试。标定通过物理仿真结果选择参数，不能称为真机几何标定。

成功标准保持为：质心持续升高至少 10 cm、保持 2 s、无桌面接触、至少 90% 的保持采样有拇指加其他手指承重接触、穿透不超过 3 mm；失败与规划失败始终计入分母。

```bash
# 加载统一偏移；当前标定适用于右手
.venv/bin/gr3-grasp simulate --config configs/scanned_candy.yaml --backend m2t2 --arm right --head-scan --fdh6-offset-profile configs/fdh6_offset_sim.json --no-video --output outputs/calibrated_demo

# 对同一批观测和起始状态评估 cuRobo
.venv-curobo/bin/python scripts/evaluate_planner.py --output outputs/curobo_baseline
.venv-curobo/bin/python scripts/evaluate_planner.py --offset-profile outputs/wrist_calibration/offset_profile.json --output outputs/curobo_calibrated
```

结果与逐次诊断保存在 `outputs/wrist_calibration`、`outputs/curobo_baseline` 和 `outputs/curobo_calibrated`。参数搜索记录与独立验证结果分别保存；本轮结果不自动改变历史基线或所有配置的默认偏移。

## 本轮完整结果

| 方案 | 校准场景 / 10 | 未参与调参的朝向 / 20 | 全部场景 / 30 | 规划失败 |
|---|---:|---:|---:|---:|
| SciPy + 原偏移 | 0 | 3 | 3 | 13 |
| SciPy + 校准偏移 | 2 | 3 | 5 | 14 |
| cuRobo + 原偏移 | 0 | 1 | 1 | 14 |
| cuRobo + 校准偏移 | 2 | 4 | 6 | 15 |

校准得到 `[-0.022, 0, -0.050]` m，在 TCP 坐标中相对原偏移修正 `[-10, 0, -5]` mm。可复用配置保存为 `configs/fdh6_offset_sim.json`，记录了 TCP 定义与验证范围，显式加载即可使用。

单独更换 cuRobo 没有提高成功率；单独校准统一偏移也没有增加另外 20 个朝向的成功次数。组合方案在这些朝向中由 3/20 变为 4/20；样本少，尚不足以证明稳定的泛化提升。总计 6/30 中包含用于调参的 10 个场景。

组合方案成功：玩偶 0°、糖果盒 45°、杯子 90°、蜡笔盒 90°、小瓷碗 0°/45°。相对原方案的未调参朝向，新增糖果盒 45°、杯子 90°，失去糖果盒 90°，净增一个。不能将不同配置的成功并集计为成功率。

剩余 24 次失败分为：

- 15 次未找到完整可行路径；其中 8 个场景所有被检查候选均在预抓取阶段失败，其余还包含接近或提起阶段失败。碰撞球近似、固定躯干、候选方向及局部求解都可能影响，当前日志不足以证明它们在几何上绝对不可达。
- 9 次执行失败：7 次正式提起阶段升高不足 1 cm，1 次约 2.02 cm，1 次曾升高约 15.83 cm 后滑落；9 次保持阶段均无有效对向承重接触。

原偏移重放的 30 个成功/失败结论与历史记录完全一致；SciPy 偏移前后输入文件与起始积分状态逐一相同，实际选中候选编号有 6/30 发生变化。组合方案的不同机械臂解与到达路径也会影响接触过程，不能把其效果只归因于一个坐标数值。

现阶段适合继续检查基于物体形状的局部手腕修正、FDH-6 手指预成形，以及闭手接触确认后再提起。统一平移偏移无法代替这些适配。

验证结果：GPU 环境 38 项测试通过；CPU 环境 35 项通过、3 项 GPU 测试跳过。测试包含左右臂多姿态 FK 对齐、实际 GPU 轨迹的 MuJoCo 复查和速度约束、偏移方向与单位、配置隔离、既有抓取物理回归。不闭手负对照的玩偶升高约 0 cm、对向接触为 0，按预期失败。

完整表格见 `outputs/wrist_calibration/report.md`，原始统计见 `comparison.json`。视频演示是批测后的复现，不计入分母。

## 原生相机到抓取的完整复现

`outputs/curobo_live_demo` 重新执行头部四视角采集、官方 M2T2 推理、校准偏移映射、cuRobo 规划及物理抓取，未直接加载缓存候选。生成的目标点云与模型候选和正式玩偶 0° 输入完全一致，见 `replay_comparison.json`。

玩偶保持期质心最小升高 **17.69 cm**，对向承重接触 **100%**，保持 **2 s**，最大目标/手指穿透 **1.963 mm**，通过验收。视频为 `outputs/curobo_live_demo/grasp.mp4`，末帧为 `final_grasp.png`；头部视角和融合数据也保存在同目录。

```bash
.venv-curobo/bin/gr3-grasp simulate \
  --config outputs/benchmark10_head_scan_context/trials/android_toy_yaw000/config.yaml \
  --backend m2t2 --arm right --planner curobo \
  --fdh6-offset-profile configs/fdh6_offset_sim.json \
  --output outputs/curobo_live_demo_repeat
```

该复现只验证一例完整流程，仍不增加 30 场基准的分母或成功次数。
