# 第三方来源与资产

本仓库提交项目适配代码、配置和小型回归样本；上游仓库、模型权重、原始网格和实验视频保留在忽略目录，不作为本仓库源码打包。

`assets.lock.json` 记录以下上游 URL、完整提交号、稀疏检出路径及许可证说明；通过 `python3 scripts/fetch_assets.py` 获取：

| 上游 | 用途 | 许可证记录 |
|---|---|---|
| [FFTAI/Wiki-GRx-Models](https://github.com/FFTAI/Wiki-GRx-Models) | GR3、FDH-6 官方 URDF 和网格 | GPL-3.0 |
| [FFTAI/fourier_aurora_sdk](https://github.com/FFTAI/fourier_aurora_sdk) | 官方 SDK 公开示例和接口资料 | Apache-2.0 |
| [kevinzakka/mujoco_scanned_objects](https://github.com/kevinzakka/mujoco_scanned_objects) | 真实物体扫描网格、纹理与 MJCF | 网格/纹理 CC-BY-4.0；MJCF MIT |
| [NVlabs/GraspGen](https://github.com/NVlabs/GraspGen) | 抓姿服务协议与夹爪配置 | 以上游 LICENSE、LICENSE_ASSETS.md 为准 |
| [NVlabs/M2T2](https://github.com/NVlabs/M2T2) | M2T2 推理与权重接口 | NVIDIA Source Code License，以上游 LICENSE 为准 |

另有独立环境安装的源码：

- [JYChen18/Dexonomy](https://github.com/JYChen18/Dexonomy)，固定提交 `d561ff9df8545deef7a3f9e7cfb4045ef66f6913`，见 `scripts/setup_dexonomy.sh`。
- [NVlabs/curobo](https://github.com/NVlabs/curobo)，固定提交 `78fd485fa82d9b9a063fb4985e371814587e666a`，见 `scripts/setup_curobo.sh`。

各上游代码、模型、生成资产按其各自条款处理；本清单不替代上游许可证。项目自有代码尚未指定对外分发许可证，整理为 Git 仓库不代表给所有内容统一授予开源许可。
