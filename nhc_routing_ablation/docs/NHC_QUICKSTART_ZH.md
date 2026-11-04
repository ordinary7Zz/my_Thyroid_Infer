# NHC 快速开始（my_Thyroid_Infer 七策略）

复制命令跑通 **NHC 七策略 B3 + 中心分析**（不含策略 7 LLM）。配置细节、产物路径、故障排查见 [`NHC_OPERATOR_GUIDE_ZH.md`](NHC_OPERATOR_GUIDE_ZH.md)。

**怎么选下面哪一节？**

| 情况 | 看哪一节 |
|------|----------|
| **NHC 数据集所在机器可以联网**（能 `git pull`、能从魔搭/HF 下权重） | 下文 **「已有仓库：快速开始（能联网）」** |
| **NHC 机器不能联网**，或只有旧版 infer、无法更新代码 | 下文 **「从零搭建（不能联网 / 旧版 infer）」**（先在能联网的机器上搭好再拷过去） |

---

## 托管地址

| 用途 | 链接 |
|------|------|
| **Git 代码**（`git clone` / `git pull`） | [github.com/ordinary7Zz/my_Thyroid_Infer](https://github.com/ordinary7Zz/my_Thyroid_Infer) |
| **HuggingFace**（权重 + 可选整包镜像，`--repo-type model`） | [huggingface.co/instincts7Zz/my_Thyroid_infer](https://huggingface.co/instincts7Zz/my_Thyroid_infer) |
| **魔搭 ModelScope**（与 HF 内容同步） | [modelscope.cn/models/instincts/my_Thyroid_infer](https://www.modelscope.cn/models/instincts/my_Thyroid_infer) |

NHC 七策略只需 **DINO + AutoGluon** 权重（见下文脚本），**不必**拉 lsnet / repvit 等其它模型。

---

## 下载 `thyroid_infer_env.tar.gz`（conda-pack 环境，约 3.4 GB）

**给 shijie 等已搭过环境的同事**：若此前已下载过本包并解压到 **`~/envs/thyroid_infer`**（且 `source …/bin/activate` 仍可用），**不必再下**。本节只是提醒——若后来为腾空间**删掉了**仓库根目录的 **`thyroid_infer_env.tar.gz`**，或**删掉了**已解压的环境目录，需要跑 NHC 时再按下面命令**重新下载**（解压步骤见本节末尾或「已有仓库」§3）。

**GitHub 仓库里通常不带这个文件**（体积大）。首次部署或重新下载后，请把它放在 **`my_Thyroid_Infer/thyroid_infer_env.tar.gz`**（与 `config.yaml` 同级）。若本地没有 tar，在**能联网**的机器上执行下面任一方式，再拷到 NHC 机或拷进 clone 好的仓库根目录。

**下载后自检**（应有约 3 GB 量级文件）：

```bash
cd /path/to/my_Thyroid_Infer
test -f thyroid_infer_env.tar.gz && ls -lh thyroid_infer_env.tar.gz
```

### HuggingFace（只下环境包）

```bash
pip install -U "huggingface_hub[cli]"
export HF_ENDPOINT=https://hf-mirror.com   # 国内可选
hf auth login

cd /path/to/my_Thyroid_Infer
hf download instincts7Zz/my_Thyroid_infer thyroid_infer_env.tar.gz \
  --repo-type model --local-dir .
```

### 魔搭 ModelScope（只下环境包）

```bash
pip install modelscope
ms login --token YOUR_TOKEN   # 或 export MODELSCOPE_API_TOKEN=...

cd /path/to/my_Thyroid_Infer
python3 <<'PY'
from modelscope import snapshot_download
from pathlib import Path

root = Path(".").resolve()
snapshot_download(
    "instincts/my_Thyroid_infer",
    local_dir=str(root),
    allow_file_pattern=["thyroid_infer_env.tar.gz"],
)
print("目标文件:", root / "thyroid_infer_env.tar.gz")
PY
```

若上述命令报 **找不到文件**：说明当前 HF/魔搭快照里可能尚未上传该 tar（维护者需单独上传）。可改用主仓库 [`README.md`](../../README.md) 的 **conda + `wheels/` 离线安装**，或向组内索取已打包的 `thyroid_infer_env.tar.gz`。

**解压与激活**（在仓库根目录已存在 `thyroid_infer_env.tar.gz` 后）：

```bash
mkdir -p ~/envs/thyroid_infer
tar -xzf thyroid_infer_env.tar.gz -C ~/envs/thyroid_infer
source ~/envs/thyroid_infer/bin/activate
conda-unpack    # 仅首次解压后执行一次
```

---

## 已有仓库：快速开始（能联网）

**适用**：NHC 数据就在**本机**，且本机**可以访问外网**（GitHub、魔搭或 HuggingFace）。仓库已在磁盘上，用 `git pull` 更新即可；权重用脚本拉取（本地已齐全则不会重复下载）。

```bash
# 1. 拉取最新代码
cd /path/to/my_Thyroid_Infer
git pull origin main

# 2. 拉取 DINO 权重（结节 5 + 分类 4 + 腺体 TGVideo + AutoGluon，二选一）

# 魔搭（默认；本地权重已齐全时会跳过下载、不连网）
bash nhc_routing_ablation/scripts/download_nhc_b3_weights.sh

# 或 HuggingFace
export HF_ENDPOINT=https://hf-mirror.com   # 国内可选
hf auth login
WEIGHT_SOURCE=huggingface bash nhc_routing_ablation/scripts/download_nhc_b3_weights.sh

# 3. 虚拟环境：若 ~/envs/thyroid_infer 仍可用则直接 activate；若 tar 或解压目录已删，先按上文「下载 thyroid_infer_env.tar.gz」
# 首次：解压 conda-pack 环境（无需先 conda create）
mkdir -p ~/envs/thyroid_infer
tar -xzf thyroid_infer_env.tar.gz -C ~/envs/thyroid_infer
source ~/envs/thyroid_infer/bin/activate
conda-unpack    # 首次解压后执行一次

# 之后每次跑 NHC 前：
source ~/envs/thyroid_infer/bin/activate
# （若你本机是用 conda create 的同名环境，也可用: conda activate thyroid_infer）

# 4. 跑七策略（默认读仓库根 config.yaml → prepare 目录 → pipeline 生成的 config.yaml）
cd nhc_routing_ablation
bash run_nhc_ablation.sh
```

运行前请已在 **`my_Thyroid_Infer` 根目录**执行过 **`python pipeline.py`** 完成 NHC 预处理。

---

## 从零搭建（不能联网 / 旧版 infer）

**适用**：存 NHC 数据的机器**不能上网**，或上面的 `my_Thyroid_Infer` 是**旧版本**、**无法 `git pull`**。  
**做法**：在**另一台能联网的电脑**上按下面三步搭好，再把 **代码 + DINO/AutoGluon 权重 +（可选）`thyroid_infer_env.tar.gz` + 预处理目录** 用 U 盘/内网拷到 NHC 私有机；也可以在私有机上直接换成新拷来的仓库，只把 **`config.yaml` 的 `prepare.output_dir` 改回原来路径**（数据仍留在原目录）。

**代码从 GitHub 拉；权重从魔搭或 HF 拉**（只下 DINO 与 AutoGluon，脚本已限定文件列表，不会下九模型权重）。**以下三步均在能联网的机器上执行**，直到拷到离线 NHC 机后再 `bash run_nhc_ablation.sh`。

### 步骤 1：克隆最新代码

```bash
git clone https://github.com/ordinary7Zz/my_Thyroid_Infer.git
cd my_Thyroid_Infer
# 有 GitHub SSH 时：git clone git@github.com:ordinary7Zz/my_Thyroid_Infer.git
```

（环境包单独下载见上文 **「下载 thyroid_infer_env.tar.gz」**；代码仍建议用 GitHub clone。）

### 步骤 2：对齐 `config.yaml` 的预处理目录

编辑仓库根 **`config.yaml`**，将 **`prepare.output_dir`** 改成**旧机器上已经在用的路径**（与旧 `config.yaml` 一致），例如：

```yaml
prepare:
  output_dir: ./datasets/processed   # 或私有机上的绝对路径
```

该目录下应已有 **`pipeline.py` 生成过的** `config.yaml`，以及 `images/`、`nodule_masks/`、`labels.json` 等。数据若仍在私有机原路径，新 clone 只改 `output_dir` 指向即可；也可把整份 `output_dir` 拷到新仓库下再写相对路径。

### 步骤 3：下载 DINO + AutoGluon 权重（魔搭或 HF，二选一）

在 **`my_Thyroid_Infer` 根目录**执行（本地文件已齐全时**不会连网重下**）：

```bash
# 魔搭（默认）
pip install modelscope
ms login --token YOUR_TOKEN   # 或 export MODELSCOPE_API_TOKEN=...
bash nhc_routing_ablation/scripts/download_nhc_b3_weights.sh
```

```bash
# HuggingFace
pip install -U "huggingface_hub[cli]"
export HF_ENDPOINT=https://hf-mirror.com   # 国内可选
hf auth login
WEIGHT_SOURCE=huggingface bash nhc_routing_ablation/scripts/download_nhc_b3_weights.sh
```

下载内容：`infer_dinov3_unet/`（结节 5 + TGVideo）、`infer_dinov3_unet_multitask/`（分类 4）、`infer_autogluon/`（Radiomics）。**不需要** `infer_lsnet`、`infer_repvit` 等目录。

---

**搭好后**：解压根目录 **`thyroid_infer_env.tar.gz`**（见上文「已有仓库」§3）→ 若尚未预处理则 `python pipeline.py` → `cd nhc_routing_ablation && bash run_nhc_ablation.sh`。

**拷到离线 NHC 机**：至少带走 **`my_Thyroid_Infer/`（含 `cascade_engine/`、`nhc_routing_ablation/`）**、上述三个 `infer_*` 权重目录、**`prepare.output_dir` 整目录**（或离线机上 `config.yaml` 仍指向原数据绝对路径）。在**离线机**上解压环境、激活后执行 `cd nhc_routing_ablation && bash run_nhc_ablation.sh`（离线机**不需要**再跑下载脚本）。

---

## 审稿意见 R2.6（需 shijie 协助回复）

以下为审稿人 **R2.6** 原文与我们需要 **shijie（或 NHC 数据负责人）** 核对的事实，用于 Methods/回复信与中心描述表（**不是**跑 `run_nhc_ablation.sh` 的操作步骤）。中心级性能、中心间异质性（*I*²）可在元数据齐后由本仓库 `_center_analysis/` 重跑；**当前先要的是下表字段能否提供，以及 8721 / 35 的定义是否一致**。

**[英文原文]**  
The manuscript reports validation on 8,721 cases from 35 centres, but the NHC-MISD-TUS cohort is insufficiently described. The authors should clarify whether these represent 35 independent hospitals and report the numbers of patients, nodules, and images per centre, together with centre-level macro-averaged performance and between-centre heterogeneity.

**[中文翻译]**  
稿件报告在来自 35 个中心的 8,721 个病例上做了验证，但 NHC-MISD-TUS 队列的描述不足。作者应澄清这是否代表 35 家独立医院，并报告各中心的患者数、结节数与图像数，以及中心级宏平均性能与中心间异质性。

**[大概解释]**  
NHC-MISD-TUS 是稿里「国家卫生健康委医学影像标准数据库」那批**私有多中心测试数据**。审稿人要把 Methods/结果写清楚，需要 shijie（或数据负责人）帮忙核对下面几类事实，便于我们填 NHC 中心描述表，并回应 R2.6（以及 R1.3 里「不同中心表现差很多」的解释）。请尽量按条目回复（可简短 bullet）：

1. **「35 个中心」各自是什么？** 是否对应 35 家**互相独立、分别上传数据**的医院（还是同一医疗集团下的院区/科室算多个「中心」）？若方便，各中心在稿里能否写**省份 + 可公开的医院称谓**（若不能公开全名，请说明可写到什么粒度，如只写省份或匿名 ID）。

2. **「8721 例」到底数的是什么？** 稿面曾把图像数和「病例」混在一起。请确认：8721 是 **8721 张超声图**，还是 **8721 个检查/病例**？一个患者是否可能有多张图、多个结节？请给出我们写 Methods 时能用的**层级关系**（例如：若干患者 → 若干结节 → 若干图像），避免再出现「例=图」的歧义。

3. **每个中心各有多少数据？** 按「中心 ID」汇总即可：**患者数、结节数（若有单独统计）、图像数**。若某中心只有图、没有患者 ID，也请说明。

4. **每个中心用什么设备拍的？** 各中心超声**设备厂商/型号**（或至少品牌 + 机型系列），用于解释中心间性能差异（R1.3）；若部分中心缺失，请标「未知」。

5. **各中心带哪些标签、标签从哪来？**（不是 ZJH 那种「谁勾 mask、几轮复核」的细流程，而是 NHC 这批数据**入库时**的规则。）请按中心说明：是否有**结节分割参考 mask**、是否有**良恶性标签**、是否有**全腺体分割**等；标签是**病理/细胞学**、**影像科报告或本地规范**，还是**只有图像、没有金标准**。这决定我们能否在该中心报 Dice/AUROC，以及 R1.3 能否做「设备或病例构成 vs 性能」的分析。

6. **合规表述（若 Methods 需要一句）。** 是否已有**伦理批件、知情同意或豁免、去标识化**等可写进稿的固定表述（与「共同作者口头同意用数据」区分开的书面记录）。若有标准话术，请直接给原文或模板。

我们这边**中心级性能、中心间异质性（I²）**可在元数据齐后重跑；当前先要的是：上表字段能否提供，以及 8721 / 35 的定义是否如上理解。
