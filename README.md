# Train Condition Monitoring System

这是一个基于 Streamlit 的列车状态监测应用。应用统一接入四个子系统：Door、ACV、Rail Corrugation 和 SHM；用户选择子系统、上传传感器文件后即可运行原有模型/分析逻辑，并下载标准 CSV 结果。

## 功能概览

| 子系统 | 输入 | 输出 | 可视化 |
| --- | --- | --- | --- |
| Door | 一个或多个 .csv 运行记录 | door_predictions.csv，包含 start_time、end_time、prediction | 电流、电压、Back-EMF、门速随门叶位置变化的 2×2 曲线；蓝色为 Normal，红色为 Abnormal resistance |
| ACV | 一个或多个 .xlsx 工况文件 | acv_predictions.csv，按车辆给出泄漏可能性排序 | 车辆排名柱状图、车厢温度随时间曲线 |
| Rail Corrugation | 一个或多个 .csv 振动记录 | rail_predictions.csv | 类别统计、两侧振动散点图、按车辆/钢轨侧的振动热图 |
| SHM | 一个或多个 .csv、.txt、.dat、.npy 或 .npz 动态应力文件 | shm_predictions.csv，每个文件一个 fatigue-damage 值 | 每文件损伤柱状图、应力波形、按振幅区间的损伤贡献 |

上传文件只用于当前会话分析，不会写回源数据。Door、SHM 的正式推理路径不要求用户上传答案文件；模型和分析入口由对应子系统目录提供。

## 推荐目录结构

部署时建议把当前最新版复制为项目根目录的 streamlit_app.py，并保持模型和子系统代码位于同一个构建目录内：

~~~text
.
├── streamlit_app.py
├── requirements.txt
├── Dockerfile
└── subsystems/
    ├── door/
    │   └── code/
    ├── acv/
    │   └── code/
    ├── rail_corrugation/
    │   ├── code/
    │   └── model/
    │       └── rail_model.joblib
    └── shm/
        └── code/
~~~

应用也兼容部分旧目录布局：子系统目录可以直接放在入口文件旁边，或者子系统代码可以直接放在子系统根目录中。但正式部署建议使用上面的 subsystems/<name>/code 布局，避免 Docker 构建时漏掉模型或原始分析模块。

当前源文件名为 streamlit_app_latest(1).py 时，可以在项目根目录执行：

~~~bash
cp 'streamlit_app_latest(1).py' streamlit_app.py
~~~

如果不复制文件，必须同步修改 Dockerfile 和本地启动命令，使它们指向实际入口文件。

## 输入格式

### Door

- 接受一个或多个 .csv 文件，例如 Train.csv、Test.csv 或其他连续运行记录。
- 第一列或名为 timestamp/datetime 的列应包含时间戳，例如 2023-7-5-0-0-3-760。
- 至少需要 17 列；额外列会保留，不会因为列数略有不同而直接拒绝。
- 记录应包含电机电流、电压、Back-EMF、门时序、控制命令、开关状态和门叶位置等信号。
- 不需要 Train_Segments_Answer.csv；生产推理使用保存的 Door 模型/适配器。

### ACV

- 每个列车工况一个 .xlsx 文件，可以一次上传多个工况。
- 使用第一个 worksheet，第一行是表头。
- 表头需要包含时间列，以及类似 Car 03 - ACV Running Mode 的车辆参数列。
- openpyxl 是云端读取 .xlsx 的必需依赖。

### Rail Corrugation

- 每个记录一个 .csv 文件。
- 推荐格式是约 10,000 行、129 个数值列：第 1 列为 0/1 速度脉冲，后续为 8 辆车、8 个轴箱位置的振动/冲击信号。
- 采样率按 10 kHz 处理；输入行数略有偏差时会给出提示，但不会仅因提示而跳过文件。
- 结果类别为 Normal、Side I 或 Side II。

### SHM

- 接受 .csv、.txt、.dat、.npy 和 .npz。
- UI 只做空文件检查，具体分隔符、元数据行、训练/测试文件名约定交给原始 SHM 模块处理。
- 对外暴露的 SHM 函数建议使用批量或单文件推理接口，例如 predict_shm、predict_shm_batch、predict_files 或 predict。
- 如果原始模块使用 analyze(data, filename, model) 这类底层函数，应在子系统适配器中补齐参数，不能把它误当成单参数预测函数直接调用。

## 本地运行

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
~~~

如果保留最新版文件名：

~~~bash
streamlit run 'streamlit_app_latest(1).py'
~~~

建议的直接依赖至少包括：

~~~text
streamlit
pandas
numpy
altair
openpyxl
scipy
scikit-learn
joblib
~~~

其中 openpyxl、模型加载依赖和各子系统原有依赖必须写入 requirements.txt，不能依赖本机环境中“碰巧已经安装”的包。

## Docker / Google Cloud Run

一个最小的 Dockerfile 示例：

~~~dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
EXPOSE 8080

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8080"]
~~~

在项目根目录部署：

~~~bash
gcloud run deploy <SERVICE_NAME> \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
~~~

本地文件修改不会自动更新 Cloud Run。修改 requirements.txt、Dockerfile、入口文件或子系统代码后，需要重新构建并重新部署；否则线上页面可能仍运行旧镜像。

## 部署前检查

~~~bash
python -m py_compile streamlit_app.py
python -c "import streamlit, pandas, numpy, altair, openpyxl"
~~~

确认以下文件没有被 .gitignore 或 Docker 构建上下文排除：

- streamlit_app.py
- requirements.txt
- Dockerfile
- subsystems/door/
- subsystems/acv/
- subsystems/rail_corrugation/
- subsystems/shm/
- Rail 模型文件 rail_model.joblib

## 常见问题

### ACV 文件在本地可以读取，部署后显示 Cannot be opened as an Excel workbook

通常是云端镜像缺少 openpyxl，或者部署使用了旧镜像。把 openpyxl 写入 requirements.txt 后重新构建并部署；同时确认 Cloud Run 的 Dockerfile/CMD 指向当前入口文件。

### 页面显示 SHM analysis failed for every uploaded file

先确认 SHM 子系统文件被复制进镜像，再检查适配器实际暴露的函数签名。尤其不要用一个参数调用需要 filename 和 model 的底层 analyze()；应用应调用生产预测入口，或在适配器中完成参数转换。

### 本地正常，线上仍是旧页面

检查 Dockerfile 的 CMD 是否仍然是旧的 streamlit_app.py，以及构建目录中是否真的包含最新入口文件。修改后必须重新执行 Cloud Run 部署。

### 页面提示输入文件不符合格式

查看上传列表中的 Note 列。检查逻辑只负责轻量格式筛选；对于 SHM，格式解析由原始模块完成，避免用通用 CSV 预检查误拒绝带元数据或特殊分隔符的合法文件。

## Git 提交

在真实项目根目录执行：

~~~bash
git status
git add README.md
git commit -m "docs: add application and deployment guide"
git push origin "$(git branch --show-current)"
~~~

如果当前项目还不是 Git 仓库，需要先在正确的项目根目录初始化并配置真实 remote；不要把上传副本目录当成正式仓库直接推送。

