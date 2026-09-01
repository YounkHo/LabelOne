<div align="center">
  <img src="./public/favicon.svg" width="72" alt="LabelOne 标志" />
  <h1>LabelOne</h1>
  <p><strong>本地优先的视觉数据集、标注、图像处理流与 AI 辅助审阅工作台。</strong></p>

  <p>
    <a href="./README.md">English</a> ·
    <a href="./README.zh-CN.md">简体中文</a>
  </p>

  <p>
    <a href="./LICENSE"><img alt="Apache-2.0 许可证" src="https://img.shields.io/badge/license-Apache--2.0-4c8bf5?style=flat-square"></a>
    <img alt="Next.js 16" src="https://img.shields.io/badge/Next.js-16-111111?style=flat-square&logo=nextdotjs">
    <img alt="React 19" src="https://img.shields.io/badge/React-19-149eca?style=flat-square&logo=react&logoColor=white">
    <img alt="TypeScript 5" src="https://img.shields.io/badge/TypeScript-5-3178c6?style=flat-square&logo=typescript&logoColor=white">
    <img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-3776ab?style=flat-square&logo=python&logoColor=white">
    <img alt="ONNX Runtime" src="https://img.shields.io/badge/ONNX-Runtime-005ced?style=flat-square&logo=onnx&logoColor=white">
    <a href="https://github.com/YounkHo/LabelOne/stargazers"><img alt="GitHub Stars" src="https://img.shields.io/github/stars/YounkHo/LabelOne?style=flat-square"></a>
  </p>

  <p>
    <a href="#宣传视频">宣传视频</a> ·
    <a href="#功能亮点">功能亮点</a> ·
    <a href="#快速开始">快速开始</a> ·
    <a href="#模型与网络">模型配置</a> ·
    <a href="#开发与验证">开发</a> ·
    <a href="#star-history">Star History</a>
  </p>
</div>

<p align="center">
  <img src="./docs/assets/labelone-hero.png" width="100%" alt="LabelOne 工作台欢迎界面" />
</p>

LabelOne 将视觉数据工作中的高频操作放进一个桌面式 Web 界面：浏览数据集、创建和审阅标注、搭建非破坏式处理流、运行 ONNX 模型、检查模型中间层，并在多个同步视图间对比结果。本地服务负责索引、标注原子保存、后台任务、模型生命周期和大图瓦片；浏览器负责高频画布交互。

> LabelOne 仍在快速开发中，当前面向本地单用户工作流。首次使用请先在数据副本上体验。

## 宣传视频

https://github.com/user-attachments/assets/c0e75997-4880-45cc-b94e-bd133829a345

## 功能亮点

### 1. 用多种视觉证据精准确认边缘 — 已支持

高精度标注的难点往往不是画框，而是判断微弱、模糊或受噪声干扰的真实边界。LabelOne 支持将传统图像处理结果与 AI 模型中间特征图进行同步分屏或叠加，让用户不只依赖原图，而是结合多种证据确定标注边缘。

### 2. 理解数据与模型行为 — 已支持

查看受支持 ONNX 层的空间特征图、Token、向量或矩阵。中间表征可以加入处理流，并与原图并排观察，帮助理解模型关注了什么，以及模型证据与人的判断在哪里出现差异。

### 3. 将自动优化管线作为单个 Workflow 节点导入 — 已支持

已有的自动标注或图像优化流程可以封装为 LabelOne 算子包：声明 manifest、Python 入口、参数 Schema 和标注策略。检查并安装后，整套外部自动化流程会以一个可复用节点出现在 Workflow 中。

### 4. 通过 RL 持续对齐人类偏好 — 规划中

计划中的强化学习闭环会在标注过程中学习人工修正，用人的边界选择和判断持续对齐模型建议。该能力尚未包含在当前版本中。

### 其他能力

| 领域 | LabelOne 提供的能力 |
| --- | --- |
| 数据集工作台 | 递归发现图像和 JSON、图片与标注分目录、虚拟文件列表和持久索引 |
| 标注工具 | 矩形、旋转框、多边形、点、两点直线、圆和开放连续线，支持撤销重做与原子保存 |
| 非破坏式 Workflow | 图像处理不覆盖源文件；标注保持可编辑，并随声明的几何算子同步变换 |
| 多视图分析 | 单画面、分屏和叠加模式，共享缩放、平移、光标、标注与像素检查 |
| AI 辅助审阅 | ONNX 推理、预测结果分组，以及一键将 AI 预测转成人工标注 |
| 频域与大图工具 | 傅里叶/Haar 算子、Deep Zoom 瓦片和可选 libvips/pyvips TIFF/WSI 区域解码 |
| 本地持久任务 | 处理流、推理、下载、暂停继续、失败重试和服务重启恢复 |

## 快速开始

### 环境要求

- macOS 或 Linux；Windows 建议使用 Git Bash/WSL，或分别手动启动前后端。
- Node.js `22.13.0` 或更高版本。
- pnpm、uv、Git 和 curl。

```bash
git clone https://github.com/YounkHo/LabelOne.git labelone
cd labelone

pnpm install --frozen-lockfile
cd server && uv sync --extra dev && cd ..

./scripts/dev-local.sh
```

启动后访问：

- Web 工作台：<http://localhost:3000>
- API 健康检查：<http://127.0.0.1:8766/api/v1/health>

按 `Ctrl+C` 同时停止前后端。首次启动会在已忽略的 `.runtime/` 目录准备模型目录元数据；即使该网络步骤失败，数据集、标注和处理流仍然可用。

### 分别启动前后端

需要独立查看日志时，在仓库根目录打开两个终端：

```bash
# 终端 1
pnpm run dev:server
```

```bash
# 终端 2
pnpm run dev:web
```

文件选择、保存、模型和任务都依赖本地 API，不能只启动 Web。

## 打开数据集

点击“打开”，选择图像目录。LabelOne 可以识别以下两种常见布局：

```text
dataset/                     dataset/
├── a.png                    ├── images/a.png
├── a.json                   └── annotations/a.json
├── nested/b.jpg
└── nested/b.json
```

图片和标注分开存放时，在工作台中设置标注根目录。系统按相对路径和同名主文件名配对。处理流预览不会覆盖源图像。

## 模型与网络

仓库不分发模型权重。可以在模型弹窗中搜索目录；已有权重会自动加载，缺少权重时需要明确确认后下载。

系统设置包括：

- **模型与存储**：模型目录，以及 GitHub、ModelScope 或 Hugging Face 首选下载源。
- **系统与网络**：跟随系统代理、直连或手动 HTTP(S) 代理。修改代理后需要重启本地服务。
- **AI 服务**：OpenAI-compatible Chat Completions 地址、模型 ID、超时和密钥环境变量名。密钥值只从服务启动环境读取。

常用覆盖项：

```bash
LABELONE_MODEL_WEIGHTS_DIR=/path/to/model-weights \
LABELONE_DATA_DIR=/path/to/labelone-data \
./scripts/dev-local.sh
```

默认 API 仅监听 `127.0.0.1`，不会自动上传图片。只有配置并选择远程推理或云端 AI 服务后，才会产生相应网络请求。

## 项目结构

```text
app/                  Web 界面、画布交互和前端领域逻辑
server/src/labelone/  FastAPI 服务、数据集、模型、处理流、任务和 Agent
server/tests/         后端测试
scripts/              本地启动器和仓库工具
docs/assets/          README 媒体资源
```

## 开发与验证

```bash
pnpm run lint
pnpm run build
pnpm run test:server
```

生产式本地运行需要先执行 `pnpm build`，然后分别启动 Uvicorn API 和 `pnpm start`。

高性能 TIFF/WSI 区域解码需要系统 `libvips` 和服务端环境中的兼容 `pyvips`。未安装时，LabelOne 使用带像素预算的 Pillow fallback。

## 常见问题

- **Node 版本不符合要求**：升级 Node.js，或设置 `LABELONE_NODE=/absolute/path/to/node`。
- **找不到 pnpm 或 uv**：安装对应工具，或设置 `LABELONE_PNPM` / `LABELONE_UV`。
- **API 未能在 8766 端口就绪**：检查端口占用和后端日志，再访问健康检查地址。
- **页面显示离线**：确认 Web 和本地 API 都已启动。
- **模型可见但不可运行**：模型权重需要独立下载。
- **代理或模型目录修改后未生效**：重启本地服务。
- **大 TIFF 超出像素预算**：安装 libvips 和 pyvips 后重启。

## 参与贡献

欢迎提交 Issue 和 Pull Request。对于较大的改动，建议先创建 Issue，对数据契约、界面行为和测试范围达成一致后再实现。

## 许可证

LabelOne 自有源码采用 [Apache License 2.0](./LICENSE)。模型权重、数据集、算子包和其他第三方组件仍分别受其原始许可证和分发条款约束。

## Star History

<a href="https://github.com/YounkHo/LabelOne/stargazers">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/YounkHo/LabelOne/star-history/star-history-dark.svg" />
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/YounkHo/LabelOne/star-history/star-history.svg" />
    <img width="100%" alt="LabelOne Star History" src="https://raw.githubusercontent.com/YounkHo/LabelOne/star-history/star-history.svg" />
  </picture>
</a>

<p align="center"><sub>由本仓库的 GitHub Actions 每日通过 GitHub API 自动生成。</sub></p>

## 支持 LabelOne

如果 LabelOne 对你有帮助，欢迎为仓库点亮 Star，让更多视觉数据工作者发现这个项目。

<a href="https://github.com/YounkHo/LabelOne/stargazers"><img alt="在 GitHub 上为 LabelOne 点亮 Star" src="https://img.shields.io/github/stars/YounkHo/LabelOne?style=for-the-badge&logo=github&label=Star%20LabelOne"></a>
