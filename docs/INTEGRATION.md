# 与其他项目联动



## video-promo-pipeline（深度联动）



### 一键全链路



从 video-promo 的 jobs 导入文案数据 → 按 **job** 切分训练/评测集 → QLoRA 训练 → 评测 → 多 LoRA A/B → 回写 `data/finetune_bridge.json` 到 video-promo。



```powershell

# 完整周期（需 GPU + LM Studio）

cd E:\xiangmu\lmstudio-finetune

.\run.ps1 -Promo



# 或分步

python run_promo_pipeline.py --import-only

python run_promo_pipeline.py --skip-train          # 仅导入+评测

python run_promo_pipeline.py --apply-config        # 回写 video-promo config.yaml 的 lm_studio.model

```



### bridge 文件



写入路径：`video-promo-pipeline/data/finetune_bridge.json`



```json

{

  "recommended_lora": "promo-v2",

  "recommended_lm_studio_model": "your-gguf-model-id",

  "eval_score": 0.85,

  "ab_winner": "promo-v2",

  "reports": {

    "eval_html": "output/promo_reports/latest_eval.html",

    "ab_html": "output/promo_reports/lora_ab_report.html"

  }

}

```



video-promo 侧配置（`config.yaml`）：



```yaml

finetune:

  auto_apply_bridge: true

  bridge_file: data/finetune_bridge.json

```



启动 Web 或 CLI 时会自动将 `lm_studio.model` 设为 bridge 推荐模型。



API：`GET http://127.0.0.1:8766/api/finetune-bridge`



### LoRA A/B 自动报告



在 LM Studio 中分别加载各 GGUF 模型，注册 LoRA 时填写 `lm_studio_model`：



```powershell

python lora_cli.py set-meta promo-v1 --lm-studio-model promo-v1-gguf
python lora_cli.py set-meta promo-v2 --lm-studio-model promo-v2-gguf



python lora_ab_report.py --loras promo-v1 promo-v2 --file data/promo/promo_eval.jsonl

# → output/lora_ab_report.html

```



按标签自动筛选：



```powershell

python lora_ab_report.py --tag promo --update-registry

```



### 配置项（lmstudio-finetune `config.yaml`）



```yaml

promo:

  pipeline_root: ../video-promo-pipeline

  jobs_dir: ../video-promo-pipeline/output

  data_dir: data/promo

  eval_job_ratio: 0.2

  ab_tag: promo

  auto_apply_config: false

```



### 手动导入（旧方式仍可用）



```powershell

python import_video_promo.py E:\xiangmu\video-promo-pipeline\output -o data/from_video_promo.jsonl

python filter_data.py data/from_video_promo.jsonl -o data/train.jsonl

```



## adb-ide / 业务文档



1. 将产品 FAQ、API 文档放入 `docs/raw/`

2. `python generate_data.py docs/raw -o data/train.jsonl`

3. 人工校验 → `python validate_data.py`

4. 训练 → 导出 GGUF → LM Studio 加载



## 评测闭环



```powershell

python eval.py --mode lmstudio --file data/examples/eval.jsonl

python eval.py --mode compare --base-model base --finetuned-model finetuned

python lora_ab_report.py --base-model base --loras promo-v1 promo-v2

```


