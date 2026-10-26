# GR3 原生相机抓取：仿真到真机

同一份仓库用于 **GPU 服务器推理** 和 **机器人本机采集、执行**，两端分别安装依赖。现场配置与标定保留在各自设备上，默认关闭真机运动。

| 入口 | 用途 |
|---|---|
| [仓库部署](docs/仓库部署.md) | 新克隆、机器人轻量安装、服务器安装、同步与离线复制 |
| [真机最小运行版本](docs/真机最小运行版本.md) | 两端启动、OAK/Aurora/FDH-6 适配与现场检查项 |
| [第三方来源](THIRD_PARTY.md) | 官方模型、扫描物体与算法依赖来源 |

新克隆后可先运行通信演示，无需 CUDA、MuJoCo、下载模型或既有实验文件：

```bash
cd gr3_grasping
python3 -m venv .venv-robot
.venv-robot/bin/python -m pip install -r requirements/robot.txt
.venv-robot/bin/python -m pip install --no-deps -e .
.venv-robot/bin/gr3-deploy probe
.venv-robot/bin/python scripts/smoke_deployment.py --output outputs/first_transport
```

演示使用明确标记的人工数据、两个独立进程和模拟执行器，只检查传输与执行流程；`grasp_success` 为 `null`。完整 Dexonomy/cuRobo 推理还需要服务器环境、官方资产和离线候选，见部署文档。以下保留仿真方法与历史实验记录；其中 `outputs/` 链接指向服务器本地结果，不随 Git 分发。

## 仿真范围

本项目从傅利叶官方 GR3 `gr3v2_1_1` URDF 和 Fourier 6DOF 手 URDF 组合模型，在 MuJoCo 中实现 **FDH-6 完整接触抓取基线**：原生相机安装位姿 RGB-D → 目标点云 → 预抓取 → 接近 → 闭手 → 提起 → 保持与判定。可用 Grounding DINO 按文字选择目标。

默认使用双侧 FDH-6，每只手 6 个主动关节、5 个联动关节；机器人基座固定，使用理想重力补偿。物体是自由刚体，靠网格碰撞和摩擦夹持，没有绑定、吸附或移动物体位姿。已验证几何基线抓方块，以及 **M2T2 官方权重 → 真实扫描糖果盒 → FDH-6 接触抓取**；药瓶仍失败。**不代表任意物体都已能抓取，也不代表完成真机迁移**。相机内参、手部 TCP、接触材料、控制增益和力矩上限仍需实机标定。GraspGen 协议可用，GPU 权重推理尚未验证。

## 运行

cuRobo GPU 规划与 FDH-6 手腕偏移校准已接入，保留原 SciPy 后端用于对照。独立 GPU 环境通过 `bash scripts/setup_curobo.sh` 安装；运行时指定 `--planner curobo`。统一偏移通过 `scripts/calibrate_wrist_offset.py` 在 10 个 0° 场景调参，再用另 20 个朝向验证；不会自动覆盖历史默认值。组合方案为 **6/30**，原方案为 **3/30**；未参与调参的 20 个朝向为 **4/20 对 3/20**，不能据此宣称已实现任意物体抓取。可复用偏移为 `configs/fdh6_offset_sim.json`。方法、命令及限制见 [cuRobo 与手腕偏移标定](docs/cuRobo与手腕偏移标定.md)。

当前开发服务器已经安装 `.venv`，下载官方模型和 CPU 版 Grounding DINO。新克隆后首次重建：

```bash
cd gr3_grasping
python3 scripts/fetch_assets.py
uv sync --locked --extra dev --extra perception
```

运行不需要学习模型的几何基线：

```bash
.venv/bin/gr3-grasp simulate --output outputs/fdh6_demo
```

按文字选择红色方块（CPU 推理，初次重建会下载固定版本的权重）：

```bash
.venv/bin/gr3-grasp simulate --target-text 'a red block' --output outputs/fdh6_red
```

仅规划、快速运行和不闭手对照：

```bash
.venv/bin/gr3-grasp simulate --no-video --output outputs/fdh6_fast
# 对照应返回非零退出码，grasp_success=false
.venv/bin/gr3-grasp simulate --no-video --no-close --output outputs/fdh6_no_close
.venv/bin/gr3-grasp simulate --plan-only --object-index 0 --output outputs/plan_only
```

`object-index` 是当前图像中按世界坐标排序后的前景编号，可查看 `detections.png`。`--arm auto` 默认按目标左右位置选择手臂，也可指定 `--arm left/right`。超出可达范围、存在碰撞、无法识别时会失败并返回非零退出码，原因写入 `report.json`。桌面高度为 1 米，默认红色方块尺寸 5×5×12 厘米、质量 80 克。蓝圆柱尝试尚未通过抓取验收，远端黄色球尚未完成抓取验证；固定躯干下不保证远处物体可达。

无显示器服务器默认使用 Mesa/EGL 软件渲染，不使用当前被其他任务占用的 CUDA 显存。使用 NVIDIA 渲染时显式设置 `GR3_RENDER_DEVICE=gpu`；无需图形窗口。Linux 需提供 EGL/GL 动态库。当前服务器 Mesa 会打印设备权限警告后使用软件设备，仿真可正常完成。

```bash
GR3_RENDER_DEVICE=gpu .venv/bin/gr3-grasp simulate --output outputs/gpu_render
.venv/bin/pytest -q
```

## 结果文件

每次运行的 `--output` 目录包含：

| 文件 | 内容 |
|---|---|
| `overview.png` / `final_overview.png` | 动作前后整体场景 |
| `camera.png` / `final_camera.png` | 机器人原生相机安装位姿的视图 |
| `detections.png`, `target_mask.png` | 目标编号、选中区域 |
| `frame.npz` | 对齐 RGB、米制深度、内参、光学坐标系外参、时间戳 |
| `target_cloud.npz` | 选中物体在相机/世界坐标系下的点云 |
| `grasps.json` | 接触/TCP 坐标系的抓取候选及来源 |
| `trajectory.npz`, `plan.json` | 预抓取、接近、提起的 7 关节轨迹和 IK 误差 |
| `grasp.mp4`, `final_grasp.png` | 整体/局部双视角实际接触抓取；`--no-video` 跳过视频 |
| `contact_trace.json`, `final_state.npz` | 50 Hz 物体高度、对向接触、法向力、穿透、桌面接触和最终物理状态 |
| `report.json` | 执行结果、误差、限制；失败时记录原因 |
| `model/scene.xml`, `model/provenance.json` | 导入后的场景与官方模型提交号 |

`grasp_success=true` 要求物体在完整 2 秒保持期内持续升高至少 10 厘米、脱离桌面，并在至少 90% 采样时刻同时具有拇指和其他手指的承力接触。评估使用仿真真值测量高度和接触；抓取位置来自实际渲染深度，不读取物体真值位姿。仿真目标关联只用于接触允许列表和评估。只允许选中目标与手指接触，手掌/桌面/机械臂碰撞仍会失败；手指自接触保留物理碰撞，允许最多 3 毫米数值穿透。

FDH-6 几何基线按桌面与观测高度估计物体中心，以手掌开口方向接近，避免伸直的手指先扫倒目标。手部使用位置伺服和力矩限幅，未实现触觉反馈控制或在线重新抓取。官方手 URDF 的 effort/velocity 为 0，仿真显式采用 0.25 Nm 力矩上限、4 秒闭手；这些是仿真假设，不是官方硬件标称值。步长为 1 毫秒。

旧假手预抓取实验可通过 `--config configs/dummy.yaml` 复现；该模式只生成 `pregrasp.mp4`，始终不会报告抓取成功。详细结果见 [docs/验证记录.md](docs/验证记录.md)。

## 实物扫描模型与 M2T2

已从 [mujoco_scanned_objects](https://github.com/kevinzakka/mujoco_scanned_objects) 下载十种真实扫描资产，用于下方批测；药瓶与糖果盒也提供独立场景。每件物体保留原始米制尺寸、纹理和 32 个凸碰撞部件；质量为显式仿真假设。M2T2 使用作者发布的固定版本权重及 CPU PointNet++ 参考算子，保留上游网络结构与参数，修正目标筛选的括号错误，使用 CPU 推理。

```bash
uv sync --locked --extra dev --extra perception --extra m2t2
.venv/bin/python scripts/fetch_m2t2_weights.py
.venv/bin/gr3-grasp simulate --config configs/scanned_candy.yaml \
  --backend m2t2 --output outputs/my_scanned_candy
```

当前糖果盒案例实际提起约 14.6 厘米、保持 2 秒，对向承力接触比例 99%；同轨迹不闭手失败。M2T2 的源夹爪接近/闭合轴经固定变换映射到 FDH-6，接触偏移仍是仿真估计。模型分数是接触置信度，不是 FDH-6 成功率。

仅推理已有相机帧，无机械运动：

```bash
.venv/bin/gr3-grasp infer-frame --frame outputs/my_scanned_candy/frame.npz \
  --config configs/scanned_candy.yaml --backend m2t2 --output outputs/my_poses
```

`m2t2_raw.npz` 保存原始源夹爪基座位姿、置信度、接触点和输入场景点云；`m2t2_source_grasps.json` 是源模型接触中心位姿；`grasps.json` 是手部映射后的候选。仿真另外输出 `m2t2_grasps.png` 与实际抓取视频。完整仓库比较、换物体方法和失败记录见 [实物扫描与 M2T2](docs/实物扫描与M2T2.md)。

## 十物体批量测试

新增 [configs/benchmark10.yaml](configs/benchmark10.yaml)：糖果盒、药瓶、杯子、蜡笔盒、锤子、螺丝刀、胶带、安卓玩偶、鼠标、小瓷碗，每种 0°/45°/90° 共 30 个单物体场景。每次只执行首个通过规划的 M2T2 候选，所有无候选、不可达和执行失败均计入总分母；提起按物体质心衡量。

```bash
.venv/bin/python scripts/benchmark_objects.py \
  --manifest configs/benchmark10.yaml --output outputs/my_benchmark10 --jobs 2
```

历史单视角 30 次测试成功 **2/30（6.7%）**：糖果盒、蜡笔盒各在 45° 成功一次，其余未通过。19 次无有效模型候选、5 次路径不可行、4 次未稳定提起或滑落。真实结果在 `outputs/benchmark10_com`，包括 CSV、JSON、逐次诊断、全部 30 次终态图及成功回放。后续发现上游目标掩码筛选缺少括号，奇偶点数影响候选是否保留；当前默认修正，复现历史行为需加 `--legacy-m2t2-filter`。历史结果不能当作修正后的模型成功率。完整协议与限制见 [十物体批量抓取](docs/十物体批量抓取.md)。

## 头部多视角点云

增加 `--head-scan`，通过头部伺服采集初始视角与三个目标附近的视角，逐帧更新外参，再将观测点云在世界系中作 2 mm 体素融合。M2T2 输入围绕观测目标保留 12 cm 桌面上下文。扫描后头部回到初始姿态，再执行抓取。关节角由官方限位约束，相机仍挂在原生头部安装位姿。

```bash
.venv/bin/gr3-grasp simulate --config configs/scanned_candy.yaml \
  --backend m2t2 --arm right --head-scan --no-video --output outputs/my_head_scan
.venv/bin/python scripts/benchmark_objects.py --head-scan \
  --output outputs/my_head_scan_benchmark --jobs 2
```

查看 `head_scan/views.png`、`scan.json`、`fused_cloud.npz` 及各帧 RGB-D。扫描要求目标初始可见、场景静止，头部转动只能提供数厘米相机基线，无法补齐所有背面。配置、筛选修复、独立对照和真机限制见 [头部扫描与点云融合](docs/头部扫描与点云融合.md)。

已完成三轮各 30 次新对照：修复筛选的单视角成功 **3/30**，四视角直接融合 **1/30**，融合并裁剪上下文 **3/30**。扫描目标占据体素增幅中位数为 57.3%；加入裁剪后无有效候选从单视角的 5 次降为 0 次，但总体抓取率仍为 10%。后续需优化 FDH-6 接触位姿、预成形与闭手。完整数据见 `outputs/head_scan_comparison/report.md`，实际头部扫描视频见 `outputs/head_scan_demo/head_scan.mp4`。

## GraspGen

客户端只依赖 NumPy、ZMQ 和 msgpack，已经包含在主环境中。GPU 服务应在独立环境/容器中安装，避免 GraspGen 的旧版 Torch/CUDA 依赖与本项目冲突。当前 RTX 5090 被既有进程占用；项目没有终止或修改这些进程。

官方源码已按提交号下载到 `third_party/GraspGen`。这份稀疏检出用于阅读服务协议和夹爪配置。准备 GPU 服务时，在**另一个环境**按 [官方安装指南](https://github.com/NVlabs/GraspGen#installation) 获取完整检出并安装依赖/权重；Blackwell 需检查当前 Torch/CUDA 和 PointNet++ 扩展的兼容性，不能直接假定上游旧版安装命令支持 RTX 5090。

已有可用 GraspGen 环境时，在该环境运行官方服务：

```bash
# 以下命令在安装好完整 GraspGen 的独立环境中运行，替换配置路径。
python client-server/graspgen_server.py \
  --gripper_config /path/to/GraspGenModels/checkpoints/graspgen_franka_panda.yml \
  --port 5556
```

回到本项目：

```bash
cd gr3_grasping
.venv/bin/gr3-grasp simulate --backend graspgen --output outputs/graspgen_demo
```

地址和超时可在 `configs/sim.yaml` 修改。无服务会在设定超时内失败，不会静默退回几何算法。适配器读取服务的夹爪类型，将模型的基座位姿通过其官方 TCP 深度变换为接触点位姿，再转到世界坐标。**这些仍然是源夹爪的候选；没有经过 GR3 灵巧手抓取质量验证。**

## Dexonomy 灵巧手抓姿实验

已接入官方 Dexonomy 的模板匹配与物理细化流程，为 FDH-6 增加六主动关节/五联动、指腹接触模板和预抓取/抓取/挤压目标。合成使用独立 `.venv-dexonomy` 环境。完整扫描 mesh 条件下的离线浮动手腕验证之后，已将一个杯子候选接入 GR3 右臂及 cuRobo，完成关节电机物理抓取；之后已接入头部相机定位（见本节末尾）。

十物体开发实验生成 480 个初始化，6 个通过完整静态/预抓取终态筛选。修正动态积分与浮动手腕跟踪后，其中 4 个完成提升至少 10 cm 并保持 2 s，覆盖糖果盒和杯子；不闭手对照全部失败，步长减半结果一致。原十物体中另外八个尚无通过预抓取筛选的候选。这是离线手部验证结果，尚非 GR3 全臂或相机抓取成功率。详见 [接入说明](docs/Dexonomy_FDH6接入.md)和[动态验证报告](outputs/dexonomy_dynamics_audit/report.md)。

**v3 姿态先验：** v2 的杯子候选是拇指朝下的反手包握。现在在合成阶段加入硬筛选：拇指不朝下，且抓取位姿与提升后的位姿都必须落在 GR3 右臂可达性地图内；同时把采样预算扩大约 30 倍。整臂物理执行中，杯子 **4/6**、糖果盒 **2/6** 个候选成功，拇指全部朝上；同轨迹不闭手对照全部失败。之后又把上游 40 mm 预抓取间隙换成手腕沿掌心法向后退的预抓取（`--pregrasp palm_retreat`）。在全部 255 个通过先验的候选上各执行一次整臂抓取：杯子 **134/215** 成功，糖果盒 **6/40** 成功；140 个不闭手对照全部失败，保持阶段没有拇指朝下的案例。十物体全部跑完后，**6/10** 种物体至少有一个整臂成功的候选（杯子、安卓玩偶、药瓶、小瓷碗、蜡笔盒、糖果盒），585 个候选中成功 326 个；锤子、胶带、螺丝刀、鼠标仍为 0。仍使用真值位姿和固定放置位置，且是每个候选一次的离线结果。之后放弃这四种，对其余 6 种物体按 0°/45°/90° 做 18 个场景，每个场景按朝向重新合成候选，只执行首个可规划候选一次：**16/18** 成功。失败的是小瓷碗 45°（提升时跟踪误差略超门限）和杯子 90°（前 10 个候选都无全局路径），18 个不闭手对照全部失败。随后把真值注册换成头部相机定位：先用头部 RGB-D 多视角扫描，再对已知网格做平面拟合，用自由空间检查和渲染已知纹理模型比较颜色来消除对称歧义。加 2 mm 深度噪声后，18 个场景的定位误差为平移 ≤ 2 mm、yaw ≤ 2.1°、ADD ≤ 1.8 mm；整臂 **17/18** 成功，唯一失败的小瓷碗 45° 与真值注册时相同。仅用几何、不比较颜色时只有 8/18。颜色比较与仿真相机共用渲染器，没有真机光照差异，结果偏乐观。详见[接入说明 v3 各节](docs/Dexonomy_FDH6接入.md#v3gr3-右臂姿态先验与扩大采样2026-10-21)。

新增的整臂杯子试验在完整 2 秒保持期内最低提升 **15.02 cm**，对向承力接触比例 100%，最大目标穿透 0.482 mm。相同手臂轨迹、相同初态的不闭手对照未提起物体。使用已知模型及仿真真值位姿、固定基座和理想重力补偿；没有浮动手腕、物体绑定或执行时关节投影。该分支允许手掌参与目标接触，保留 3 mm 手物穿透上限；原 M2T2 基线的接触规则不变。查看[整臂报告和视频](outputs/dexonomy_gr3_arm/report.md)。

```bash
# 需要本服务器已有的 v2 候选和资产；输出使用新目录。
.venv-curobo/bin/python scripts/run_dexonomy_gr3.py \
  --output outputs/my_dexonomy_gr3 --video
.venv-curobo/bin/python scripts/run_dexonomy_gr3.py \
  --output outputs/my_dexonomy_gr3_open --replay-plan outputs/my_dexonomy_gr3 \
  --no-close --video
```

```bash
bash scripts/setup_dexonomy.sh
bash scripts/setup_dexonomy_eval.sh
MUJOCO_GL=egl .venv-dexonomy/bin/python scripts/run_dexonomy.py all --objects all
bash scripts/run_dexonomy_dynamics_audit.sh
```

## 真机数据入口

新增服务器/机器人两端最小运行版本：上传 RGB-D/关节快照 → 服务器 Dexonomy/cuRobo → 本机模拟执行与回执；同时准备 OAK、Aurora 和 FDH-6 DDS 的可替换适配层。默认关闭真机动作，设备、映射和实物定位尚待现场核对。两进程冒烟命令及明天的适配清单见[真机最小运行版本](docs/真机最小运行版本.md)。

统一数据格式是 `RGBDFrame`，要求深度对齐 RGB，深度单位为米，图像已去畸变。`T_world_camera` 将 OpenCV 光学坐标系的点变到世界坐标系：x 向右、y 向下、z 向前。颈部关节/基座移动后，应更新外参并同步关节时间戳。

```bash
.venv/bin/gr3-grasp infer-frame --frame /absolute/path/frame.npz \
  --backend geometry --output outputs/replay
```

它只执行感知与抓取候选生成，不向机器人发送动作。工作空间和桌面高度需在配置中匹配实际坐标系。

如果原生相机确认为可直接访问的 OAK 双目 RGB-D，可在相机所在主机使用可选的 DepthAI 2.x 采集脚本：

```bash
uv sync --locked --extra perception --extra oak --extra dev
.venv/bin/python scripts/capture_oak.py \
  --extrinsics /absolute/path/measured_camera_extrinsics.json \
  --output /absolute/path/frame.npz
```

标定 JSON 必须包含 `calibrated: true` 和测得的 `T_world_camera` 4×4 矩阵。脚本适用于 CAM_A 彩色 + CAM_B/C 双目的兼容 OAK，不是所有 GR3 相机配置的通用驱动。RGB-D 校准、RGB 光学畸变和流占用必须在具体设备上验证；本服务器未连接相机，采集脚本尚未硬件实测。它按设备时间戳匹配两流，但不提供硬件同步。原生相机若只有 RGB，需要先解决米制深度来源。

Aurora 只读状态入口需在**机器人本机**及其供应商客户端环境运行（当前官方文档的客户端版本为 0.2.1）：

```bash
.venv/bin/gr3-grasp robot-state --output outputs/robot_readonly
```

该入口只读取双臂关节，不切换控制状态或发送指令。尚未连接真机测试，也未自动安装供应商 SDK。真正闭环抓取还需要交付版机械模型、手部控制映射、相机标定、机器人侧执行器与站立控制器对接。

## 工程文件

- `configs/sim.yaml`：模型、相机、桌面、工作空间和速度参数。
- `src/gr3_grasping/model.py`：官方 URDF 组合、联动约束与场景构建。
- `src/gr3_grasping/grasping.py`：FDH-6 预成形、接近/闭手/提起/保持和接触判定。
- `simulation.py`、`perception.py`、`grasps.py`、`planning.py`：取图、感知、候选和规划。
- `scripts/capture_oak.py`、`hardware.py`：可选相机采集与 Aurora 只读入口。
- `tests/`：坐标投影、前景分离、真实 ZMQ 序列化往返、超时、官方模型导入、IK、手部碰撞、联动约束、RGB-D 完整抓取和不闭手对照。
- `assets.lock.json`、`uv.lock`：上游提交与 Python 依赖固定版本。

第三方模型与源码保留各自许可证，见 `assets.lock.json` 和对应官方仓库。模型下载脚本只验证/创建检出，不重写已有的不同版本。
