#!/usr/bin/env python3
"""Gradio 全功能控制台。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.alignment_pipeline import run_sft_dpo_pipeline  # noqa: E402
from src.api_usage import usage_markdown  # noqa: E402
from src.benchmark_runner import run_benchmark  # noqa: E402
from src.config_loader import load_config  # noqa: E402
from src.data_augment import augment_file  # noqa: E402
from src.data_convert import convert_file  # noqa: E402
from src.data_dedup import deduplicate_file  # noqa: E402
from src.data_generator import generate_from_text  # noqa: E402
from src.data_merge import merge_jsonl_files  # noqa: E402
from src.data_quality import filter_file  # noqa: E402
from src.data_sample import shuffle_and_sample  # noqa: E402
from src.data_split import split_dataset, split_summary  # noqa: E402
from src.data_stats import compute_stats  # noqa: E402
from src.data_validate import validate_jsonl  # noqa: E402
from src.data_version import versions_markdown  # noqa: E402
from src.dataset_diff import diff_datasets  # noqa: E402
from src.doc_crawler import crawl_to_markdown_files  # noqa: E402
from src.doctor import format_doctor_markdown, run_doctor  # noqa: E402
from src.eval_runner import run_eval_judge, run_eval_lmstudio, save_report  # noqa: E402
from src.experiment_registry import experiments_markdown, load_metrics_tail  # noqa: E402
from src.hub_upload import upload_to_hub  # noqa: E402
from src.import_video_promo import import_video_promo_jobs  # noqa: E402
from src.inbox_watcher import process_inbox  # noqa: E402
from src.lmstudio_client import chat, chat_stream, check_lm_studio  # noqa: E402
from src.lora_ab_report import compare_loras, save_lora_ab_html  # noqa: E402
from src.lora_manager import (  # noqa: E402
    list_by_tag,
    registry_to_markdown,
)
from src.metrics_plot import to_gradio_plots, to_plot_dataframe  # noqa: E402
from src.pipeline import run_full_pipeline  # noqa: E402
from src.promo_integration import run_promo_cycle  # noqa: E402
from src.regression_gate import check_regression, save_baseline  # noqa: E402
from src.review_store import approve_index, list_pending, merge_approved_to, reject_index  # noqa: E402
from src.semantic_dedup import semantic_deduplicate_file  # noqa: E402
from src.dialogue_analytics import analyze_dialogue_dataset, format_analytics_markdown  # noqa: E402
from src.data_lineage import lineage_markdown  # noqa: E402
from src.eval_history import eval_history_markdown  # noqa: E402
from src.export_wizard import check_export_readiness, format_export_wizard_markdown  # noqa: E402
from src.hard_example_sampler import sample_hard_examples  # noqa: E402
from src.lmstudio_health import format_health_markdown, run_lmstudio_health  # noqa: E402
from src.profile_wizard import profile_diff_markdown  # noqa: E402
from src.release_check import format_release_markdown, run_release_check  # noqa: E402
from src.train_runner import (  # noqa: E402
    start_training,
    stop_training,
    tail_training_log,
    training_status,
)
from src.vram_estimate import estimate_vram_gb, format_vram_markdown  # noqa: E402


def _run_cmd(cmd: list[str]) -> str:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (p.stdout or "") + (p.stderr or "")
    return out or f"exit {p.returncode}"


def ui_validate(file_path: str, template: str) -> str:
    report = validate_jsonl(Path(file_path), template=template)
    lines = [json.dumps(report.to_dict(), ensure_ascii=False, indent=2)]
    for e in report.errors[:30]:
        lines.append(f"ERROR L{e.line}: {e.message}")
    for w in report.warnings[:30]:
        lines.append(f"WARN L{w.line}: {w.message}")
    return "\n".join(lines)


def ui_stats(file_path: str, template: str) -> str:
    return json.dumps(compute_stats(Path(file_path), template=template), ensure_ascii=False, indent=2)


def ui_split(input_path: str, out_dir: str) -> str:
    paths = split_dataset(Path(input_path), Path(out_dir))
    return json.dumps(
        {**{k: str(v) for k, v in paths.items()}, "counts": split_summary(paths)}, ensure_ascii=False, indent=2
    )


def ui_convert(inp: str, out: str, fmt: str) -> str:
    n = convert_file(Path(inp), Path(out), fmt)
    return f"已转换 {n} 条 → {out}"


def ui_generate(doc: str) -> str:
    if not check_lm_studio():
        return "LM Studio 未连接"
    rows = generate_from_text(doc)
    return json.dumps(rows, ensure_ascii=False, indent=2)


def ui_train(resume: bool) -> str:
    extra = ["--resume"] if resume else []
    r = start_training("train.py", extra_args=extra)
    return r.get("detail") or f"已启动训练 PID={r.get('pid')}，日志: {r.get('log')}"


def ui_train_align(script: str) -> str:
    r = start_training(script)
    return r.get("detail") or f"已启动 {script} PID={r.get('pid')}"


def ui_train_log() -> str:
    st = training_status()
    head = f"运行中 PID={st['pid']}" if st["running"] else f"已结束 exit={st.get('exit_code')}"
    return head + "\n\n" + tail_training_log(100)


def ui_train_stop() -> str:
    return stop_training()


def ui_vram_precheck() -> str:
    cfg = load_config()
    return format_vram_markdown(estimate_vram_gb(cfg))


def ui_lm_health() -> str:
    return format_health_markdown(run_lmstudio_health())


def ui_export_wizard() -> str:
    return format_export_wizard_markdown(check_export_readiness())


def ui_release_check() -> str:
    return format_release_markdown(run_release_check())


def ui_dialogue_stats(file_path: str) -> str:
    return format_analytics_markdown(analyze_dialogue_dataset(Path(file_path)))


def ui_hard_examples(eval_file: str, out: str, th: float) -> str:
    import json

    r = sample_hard_examples(Path(eval_file), threshold=th, out_path=Path(out))
    return json.dumps(r, ensure_ascii=False, indent=2)


def ui_lineage() -> str:
    return lineage_markdown()


def ui_eval_history() -> str:
    return eval_history_markdown()


def ui_profile_diff() -> str:
    return profile_diff_markdown()


def ui_save_config(yaml_text: str) -> str:
    path = ROOT / "config.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    return f"已保存 {path}"


def ui_load_config() -> str:
    path = ROOT / "config.yaml"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return (ROOT / "config.example.yaml").read_text(encoding="utf-8")


def ui_export(gguf: bool) -> str:
    cmd = [sys.executable, str(ROOT / "export.py")]
    if gguf:
        cmd.append("--gguf")
    return _run_cmd(cmd)


def ui_eval(file_path: str, max_n: int) -> str:
    try:
        report = run_eval_lmstudio(Path(file_path), max_samples=max_n)
        save_report(report, ROOT / "output" / "eval_report.json")
        import io

        from rich.console import Console

        buf = io.StringIO()
        console = Console(file=buf, width=100)
        from rich.table import Table

        table = Table(title="评测")
        table.add_column("分")
        table.add_column("问题")
        for s in report.samples[:15]:
            table.add_row(f"{s.score:.2f}", s.prompt[:50])
        console.print(table)
        console.print(f"平均: {report.avg_score:.3f}")
        return buf.getvalue()
    except Exception as e:
        return str(e)


def ui_chat(prompt: str) -> str:
    if not check_lm_studio():
        return "请先启动 LM Studio Local Server"
    return chat(prompt)


def ui_chat_stream(prompt: str):
    if not check_lm_studio():
        yield "请先启动 LM Studio Local Server"
        return
    acc = ""
    for chunk in chat_stream(prompt):
        data = json.loads(chunk)
        acc += data["choices"][0]["delta"].get("content", "")
        yield acc


def ui_semantic_dedup(inp: str, out: str) -> str:
    s = semantic_deduplicate_file(Path(inp), Path(out))
    return json.dumps(s, ensure_ascii=False)


def ui_benchmark(judge: bool) -> str:
    try:
        r = run_benchmark(use_judge=judge, max_samples=5)
        save_baseline(r.judge_score or r.rule_score, name=r.name)
        gate = check_regression(r.judge_score or r.rule_score)
        return f"规则分 {r.rule_score:.3f} 评审分 {r.judge_score}\n{gate['reason']}"
    except Exception as e:
        return str(e)


def ui_inbox_once() -> str:
    return json.dumps(process_inbox(), ensure_ascii=False, indent=2)


def ui_metrics_plot(path: str):
    plots = to_gradio_plots(Path(path))
    if not plots:
        return "无 metrics 数据"
    lines = []
    for p in plots:
        lines.append(f"## {p['title']}")
        for row in p["data"][-15:]:
            lines.append(f"  step {row[0]}: {row[1]}")
    return "\n".join(lines)


def ui_doctor() -> str:
    return format_doctor_markdown(run_doctor())


def ui_merge(inputs: str, out: str) -> str:
    files = [Path(x.strip()) for x in inputs.split(",") if x.strip()]
    if len(files) < 2:
        return "请用逗号分隔至少 2 个文件路径"
    return json.dumps(merge_jsonl_files(files, Path(out)), ensure_ascii=False, indent=2)


def ui_diff(a: str, b: str) -> str:
    return json.dumps(diff_datasets(Path(a), Path(b)), ensure_ascii=False, indent=2)


def ui_shuffle(inp: str, out: str, n: int) -> str:
    return json.dumps(shuffle_and_sample(Path(inp), Path(out), max_samples=int(n)), ensure_ascii=False)


def ui_versions() -> str:
    return versions_markdown()


def ui_train_report(run_dir: str) -> str:
    return report_to_markdown(build_train_report(run_dir=Path(run_dir) if run_dir else None))


def ui_augment(inp: str, out: str, variants: int, max_rows: int) -> str:
    try:
        return json.dumps(
            augment_file(Path(inp), Path(out), variants_per_row=int(variants), max_rows=int(max_rows)),
            ensure_ascii=False,
            indent=2,
        )
    except Exception as e:
        return str(e)


def ui_metrics_chart(path: str):
    import pandas as pd

    df = to_plot_dataframe(Path(path))
    if df is None or df.empty:
        return pd.DataFrame({"step": [], "metric": [], "value": []})
    return df


def ui_alignment(skip_sft: bool, skip_dpo: bool) -> str:
    try:
        r = run_sft_dpo_pipeline(skip_sft=skip_sft, skip_dpo=skip_dpo, skip_eval=skip_sft)
        return json.dumps(r, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        return str(e)


def ui_crawl(urls: str, gen: bool) -> str:
    url_list = [u.strip() for u in urls.splitlines() if u.strip()]
    if not url_list:
        return "请输入 URL"
    try:
        stats = crawl_to_markdown_files(url_list, ROOT / "data" / "crawled", max_pages=5)
        if gen and stats.get("files"):
            from src.data_generator import generate_from_files

            n = generate_from_files([Path(f) for f in stats["files"]], ROOT / "data" / "crawled_train.jsonl")
            stats["generated"] = n
        return json.dumps(stats, ensure_ascii=False, indent=2)
    except Exception as e:
        return str(e)


def ui_hub_upload(repo: str) -> str:
    try:
        return upload_to_hub(load_config(), repo_id=repo)
    except Exception as e:
        return str(e)


def ui_usage() -> str:
    return usage_markdown()


def ui_promo(skip_train: bool, apply_cfg: bool) -> str:
    try:
        r = run_promo_cycle(skip_train=skip_train, skip_export=skip_train, apply_config=apply_cfg)
        return json.dumps(r, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        return str(e)


def ui_lora_ab(eval_file: str, tag: str) -> str:
    try:
        names = [a["name"] for a in list_by_tag(tag)]
        if len(names) < 2:
            return f"标签 `{tag}` 下 LoRA 不足 2 个（当前 {len(names)}）"
        r = compare_loras(names, Path(eval_file), max_samples=10, mode="judge")
        save_lora_ab_html(r, ROOT / "output" / "lora_ab_report.html")
        lines = [f"推荐: {r.get('winner')}", f"领先: {r.get('delta_vs_runner_up', 0):+.3f}", ""]
        for item in r.get("ranking", []):
            mark = "*" if item["name"] == r.get("winner") else " "
            lines.append(f"{mark} {item['name']}: {item['avg_score']:.3f}")
        lines.append("\nHTML: output/lora_ab_report.html")
        return "\n".join(lines)
    except Exception as e:
        return str(e)


def ui_pipeline(skip_train: bool, skip_export: bool) -> str:
    try:
        r = run_full_pipeline(skip_train=skip_train, skip_export=skip_export, skip_eval=skip_train)
        return json.dumps(r, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        return str(e)


def ui_lora_list() -> str:
    return registry_to_markdown() or "(暂无注册 LoRA)"


def ui_dedup(inp: str, out: str, th: float) -> str:
    s = deduplicate_file(Path(inp), Path(out), threshold=th)
    return json.dumps(s, ensure_ascii=False)


def ui_filter(inp: str, out: str, min_s: float) -> str:
    s = filter_file(Path(inp), Path(out), Path("data/rejected.jsonl"), min_score=min_s)
    return json.dumps(s, ensure_ascii=False)


def ui_import_promo(root: str, out: str) -> str:
    return json.dumps(import_video_promo_jobs(Path(root), Path(out)), ensure_ascii=False, indent=2)


def ui_metrics(path: str) -> str:
    return json.dumps(load_metrics_tail(path, 30), ensure_ascii=False, indent=2)


def ui_experiments() -> str:
    return experiments_markdown()


def ui_judge_eval(file_path: str, max_n: int) -> str:
    try:
        report = run_eval_judge(Path(file_path), max_samples=int(max_n))
        save_report(report, ROOT / "output" / "eval_judge.json")
        return f"LLM 评审均分: {report.avg_score:.3f}\n" + "\n".join(
            f"{s.score:.2f} {s.prompt[:40]} — {s.notes}" for s in report.samples[:10]
        )
    except Exception as e:
        return str(e)


def ui_review_list() -> str:
    rows = list_pending()
    lines = [f"[{i}] {(r.get('instruction') or '')[:50]}" for i, r in enumerate(rows[:30])]
    return f"待校对 {len(rows)} 条\n" + "\n".join(lines)


def ui_review_approve(idx: int, edit: str) -> str:
    row = approve_index(int(idx), edited_output=edit or None)
    return json.dumps(row, ensure_ascii=False) if row else "无效索引"


def ui_review_merge(target: str) -> str:
    n = merge_approved_to(Path(target))
    return f"已合并 {n} 条 → {target}"


def build_app() -> gr.Blocks:
    cfg = load_config()
    default_train = cfg.get("dataset", {}).get("train_file", "data/examples/train.jsonl")
    with gr.Blocks(title="lmstudio-finetune", theme=gr.themes.Soft()) as app:
        gr.Markdown("# lmstudio-finetune 控制台\n数据 → 训练 → 导出 → 评测 → 对话")
        with gr.Tab("数据"):
            with gr.Row():
                data_file = gr.Textbox(value=default_train, label="JSONL 路径")
                template = gr.Dropdown(["qwen", "llama3", "chatml", "mistral", "alpaca"], value="qwen", label="模板")
            with gr.Row():
                btn_val = gr.Button("校验")
                btn_stat = gr.Button("统计")
            data_out = gr.Textbox(lines=20, label="结果")
            btn_val.click(ui_validate, [data_file, template], data_out)
            btn_stat.click(ui_stats, [data_file, template], data_out)
            gr.Markdown("### 切分 / 转换")
            with gr.Row():
                split_in = gr.Textbox(value="data/all.jsonl", label="源文件")
                split_out = gr.Textbox(value="data", label="输出目录")
            btn_split = gr.Button("8:1:1 切分")
            btn_split.click(ui_split, [split_in, split_out], data_out)
            with gr.Row():
                conv_in = gr.Textbox(label="输入文件")
                conv_out = gr.Textbox(value="data/converted.jsonl", label="输出 JSONL")
                conv_fmt = gr.Dropdown(["sharegpt", "csv", "faq_md"], value="faq_md", label="格式")
            btn_conv = gr.Button("格式转换")
            btn_conv.click(ui_convert, [conv_in, conv_out, conv_fmt], data_out)
            gr.Markdown("### 去重 / 质检")
            with gr.Row():
                dedup_in = gr.Textbox(value=default_train, label="去重输入")
                dedup_out = gr.Textbox(value="data/train_deduped.jsonl", label="去重输出")
                dedup_th = gr.Slider(0.5, 1.0, value=0.85, step=0.01, label="相似度阈值")
            btn_dedup = gr.Button("近似去重")
            btn_dedup.click(ui_dedup, [dedup_in, dedup_out, dedup_th], data_out)
            with gr.Row():
                sem_in = gr.Textbox(value=default_train, label="语义去重输入")
                sem_out = gr.Textbox(value="data/train_semantic_deduped.jsonl", label="语义去重输出")
            btn_sem = gr.Button("语义去重")
            btn_sem.click(ui_semantic_dedup, [sem_in, sem_out], data_out)
            btn_dialogue = gr.Button("对话质量分析")
            btn_dialogue.click(ui_dialogue_stats, data_file, data_out)
            btn_lin = gr.Button("数据血缘")
            btn_lin.click(ui_lineage, outputs=data_out)
            with gr.Row():
                filt_in = gr.Textbox(value="data/generated.jsonl", label="过滤输入")
                filt_out = gr.Textbox(value="data/train_filtered.jsonl", label="过滤输出")
                filt_min = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="最低质量分")
            btn_filt = gr.Button("质量过滤")
            btn_filt.click(ui_filter, [filt_in, filt_out, filt_min], data_out)
            gr.Markdown("### video-promo 导入")
            with gr.Row():
                promo_root = gr.Textbox(value="../video-promo-pipeline/output/jobs", label="jobs 目录")
                promo_out = gr.Textbox(value="data/from_video_promo.jsonl", label="输出")
            btn_promo = gr.Button("导入文案数据")
            btn_promo.click(ui_import_promo, [promo_root, promo_out], data_out)
        with gr.Tab("造数据"):
            doc = gr.Textbox(lines=12, label="粘贴文档 / FAQ")
            btn_gen = gr.Button("LM Studio 生成问答对")
            gen_out = gr.Textbox(lines=15, label="生成结果 JSON")
            btn_gen.click(ui_generate, doc, gen_out)
        with gr.Tab("状态"):
            with gr.Row():
                btn_health = gr.Button("LM Studio 健康检查", variant="primary")
                btn_vram = gr.Button("显存预检")
                btn_release = gr.Button("发布检查清单")
            status_out = gr.Textbox(lines=12, label="详情")
            btn_health.click(ui_lm_health, outputs=status_out)
            btn_vram.click(ui_vram_precheck, outputs=status_out)
            btn_release.click(ui_release_check, outputs=status_out)
            app.load(ui_lm_health, outputs=status_out)
        with gr.Tab("训练"):
            resume = gr.Checkbox(label="断点续训", value=False)
            with gr.Row():
                btn_train = gr.Button("启动 QLoRA", variant="primary")
                btn_dpo = gr.Button("启动 DPO")
                btn_kto = gr.Button("启动 KTO")
                btn_orpo = gr.Button("启动 ORPO")
            with gr.Row():
                btn_log = gr.Button("刷新训练日志")
                btn_stop = gr.Button("停止训练")
            train_log = gr.Textbox(label="训练日志", lines=14)
            btn_train.click(ui_train, resume, train_log)
            btn_dpo.click(lambda: ui_train_align("dpo_train.py"), outputs=train_log)
            btn_kto.click(lambda: ui_train_align("kto_train.py"), outputs=train_log)
            btn_orpo.click(lambda: ui_train_align("orpo_train.py"), outputs=train_log)
            btn_log.click(ui_train_log, outputs=train_log)
            btn_stop.click(ui_train_stop, outputs=train_log)
            metrics_path = gr.Textbox(value="output/run1/metrics.jsonl", label="metrics 路径")
            btn_metrics = gr.Button("刷新训练曲线数据")
            metrics_out = gr.Textbox(lines=12, label="最近 metrics")
            btn_metrics.click(ui_metrics, metrics_path, metrics_out)
            train_timer = gr.Timer(value=5)
            train_timer.tick(ui_train_log, outputs=train_log)
        with gr.Tab("导出"):
            btn_wiz = gr.Button("导出向导检查")
            wiz_out = gr.Textbox(lines=8, label="导出状态")
            btn_wiz.click(ui_export_wizard, outputs=wiz_out)
            gguf = gr.Checkbox(label="转 GGUF", value=True)
            btn_exp = gr.Button("合并 + 导出")
            exp_out = gr.Textbox(lines=10, label="日志")
            btn_exp.click(ui_export, gguf, exp_out)
        with gr.Tab("评测"):
            eval_file = gr.Textbox(value="data/examples/eval.jsonl", label="评测集")
            max_n = gr.Slider(1, 50, value=10, step=1, label="样本数")
            with gr.Row():
                btn_eval = gr.Button("规则评测")
                btn_judge = gr.Button("LLM 评审", variant="primary")
            eval_out = gr.Textbox(lines=18, label="报告")
            btn_eval.click(ui_eval, [eval_file, max_n], eval_out)
            btn_judge.click(ui_judge_eval, [eval_file, max_n], eval_out)
            btn_hist = gr.Button("评测历史")
            btn_hist.click(ui_eval_history, outputs=eval_out)
            gr.Markdown("### 难例回流")
            with gr.Row():
                hard_eval = gr.Textbox(value="output/eval_report.json", label="评测报告")
                hard_out = gr.Textbox(value="data/hard_examples.jsonl", label="输出")
                hard_th = gr.Slider(0, 1, value=0.6, step=0.05, label="分数阈值")
            btn_hard = gr.Button("采样难例")
            btn_hard.click(ui_hard_examples, [hard_eval, hard_out, hard_th], eval_out)
        with gr.Tab("实验"):
            exp_md = gr.Markdown()
            btn_exp = gr.Button("刷新实验对比")
            btn_exp.click(ui_experiments, outputs=exp_md)
            app.load(ui_experiments, outputs=exp_md)
        with gr.Tab("校对"):
            review_box = gr.Textbox(lines=10, label="待校对")
            btn_rl = gr.Button("刷新列表")
            btn_rl.click(ui_review_list, outputs=review_box)
            with gr.Row():
                r_idx = gr.Number(value=0, precision=0, label="索引")
                r_edit = gr.Textbox(label="修改后回答（可选）")
            with gr.Row():
                btn_ok = gr.Button("通过")
                btn_no = gr.Button("拒绝")
            merge_tgt = gr.Textbox(value="data/train.jsonl", label="合并到")
            btn_merge = gr.Button("合并已通过 → 训练集")
            review_status = gr.Textbox(label="操作结果")
            btn_ok.click(ui_review_approve, [r_idx, r_edit], review_status)
            btn_no.click(lambda i: "OK" if reject_index(int(i)) else "失败", r_idx, review_status)
            btn_merge.click(ui_review_merge, merge_tgt, review_status)
            app.load(ui_review_list, outputs=review_box)
        with gr.Tab("对话"):
            lm_hint = gr.Markdown("加载失败时请：小模型 Q4_K_M、GPU Offload=0、更新驱动")
            prompt = gr.Textbox(label="输入", value="你好")
            with gr.Row():
                btn_chat = gr.Button("发送")
                btn_stream = gr.Button("流式发送")
            reply = gr.Textbox(label="回复", lines=8)
            btn_chat.click(ui_chat, prompt, reply)
            btn_stream.click(ui_chat_stream, prompt, reply)
        with gr.Tab("Benchmark"):
            btn_bench = gr.Button("运行 Benchmark（5 条）")
            judge_ck = gr.Checkbox(label="LLM 评审", value=False)
            bench_out = gr.Textbox(lines=8)
            btn_bench.click(ui_benchmark, judge_ck, bench_out)
        with gr.Tab("收件箱"):
            gr.Markdown("将 JSONL 放入 `data/inbox/`，点击处理合并到训练集")
            btn_inbox = gr.Button("处理一次")
            inbox_out = gr.Textbox(lines=8)
            btn_inbox.click(ui_inbox_once, outputs=inbox_out)
        with gr.Tab("工具"):
            gr.Markdown("### 环境诊断 / 版本 / 报告")
            with gr.Row():
                btn_doc = gr.Button("运行 Doctor")
                btn_ver = gr.Button("数据集版本")
            tool_out = gr.Textbox(lines=14)
            btn_doc.click(ui_doctor, outputs=tool_out)
            btn_ver.click(ui_versions, outputs=tool_out)
            gr.Markdown("### 合并 / 对比 / 打乱")
            merge_in = gr.Textbox(label="合并文件（逗号分隔）", value="data/a.jsonl,data/b.jsonl")
            merge_out = gr.Textbox(label="合并输出", value="data/merged.jsonl")
            btn_merge = gr.Button("合并")
            btn_merge.click(ui_merge, [merge_in, merge_out], tool_out)
            with gr.Row():
                diff_a = gr.Textbox(label="对比 A", value=default_train)
                diff_b = gr.Textbox(label="对比 B", value="data/train_deduped.jsonl")
            btn_diff = gr.Button("对比差异")
            btn_diff.click(ui_diff, [diff_a, diff_b], tool_out)
            with gr.Row():
                sh_in = gr.Textbox(label="打乱输入", value=default_train)
                sh_out = gr.Textbox(label="打乱输出", value="data/shuffled.jsonl")
                sh_n = gr.Number(value=0, precision=0, label="最多保留 N 条，0=全部")
            btn_sh = gr.Button("打乱")
            btn_sh.click(ui_shuffle, [sh_in, sh_out, sh_n], tool_out)
            gr.Markdown("### 数据增强（需 LM Studio）")
            with gr.Row():
                aug_in = gr.Textbox(label="增强输入", value=default_train)
                aug_out = gr.Textbox(label="增强输出", value="data/augmented.jsonl")
                aug_n = gr.Number(value=2, precision=0, label="每条变体数")
                aug_max = gr.Number(value=10, precision=0, label="最多处理源样本数")
            btn_aug = gr.Button("释义增强")
            btn_aug.click(ui_augment, [aug_in, aug_out, aug_n, aug_max], tool_out)
            run_dir = gr.Textbox(value="output/run1", label="训练目录（报告）")
            btn_tr = gr.Button("生成训练报告")
            btn_tr.click(ui_train_report, run_dir, tool_out)
        with gr.Tab("流水线"):
            gr.Markdown("校验 → 训练 → 导出 → 评测（可勾选跳过步骤）")
            sk_tr = gr.Checkbox(label="跳过训练", value=False)
            sk_ex = gr.Checkbox(label="跳过导出", value=False)
            btn_pipe = gr.Button("运行 Pipeline", variant="primary")
            pipe_out = gr.Textbox(lines=15)
            btn_pipe.click(ui_pipeline, [sk_tr, sk_ex], pipe_out)
        with gr.Tab("训练曲线"):
            mpath = gr.Textbox(value="output/run1/metrics.jsonl", label="metrics.jsonl")
            btn_mplot = gr.Button("绘制曲线")
            line_plot = gr.LinePlot(
                x="step",
                y="value",
                color="metric",
                title="训练指标",
                x_title="Step",
                y_title="Value",
                height=400,
            )
            mplot_out = gr.Textbox(lines=8, label="文本摘要")
            btn_mplot.click(ui_metrics_chart, mpath, line_plot)
            btn_mplot.click(ui_metrics_plot, mpath, mplot_out)
            curve_timer = gr.Timer(value=10)
            curve_timer.tick(ui_metrics_plot, mpath, mplot_out)
        with gr.Tab("配置"):
            cfg_editor = gr.Textbox(lines=22, label="config.yaml", value=ui_load_config())
            with gr.Row():
                btn_cfg_load = gr.Button("重新加载")
                btn_cfg_save = gr.Button("保存配置", variant="primary")
                btn_prof = gr.Button("Profile 对比")
            cfg_status = gr.Textbox(lines=10, label="状态")
            btn_cfg_load.click(ui_load_config, outputs=cfg_editor)
            btn_cfg_save.click(ui_save_config, cfg_editor, cfg_status)
            btn_prof.click(ui_profile_diff, outputs=cfg_status)
        with gr.Tab("video-promo"):
            gr.Markdown("从 video-promo jobs 导入 → 训练 → 评测 → A/B → 回写 bridge")
            pr_sk = gr.Checkbox(label="跳过训练", value=False)
            pr_apply = gr.Checkbox(label="回写 video-promo config.yaml", value=False)
            btn_pr = gr.Button("运行 promo 联动", variant="primary")
            pr_out = gr.Textbox(lines=16)
            btn_pr.click(ui_promo, [pr_sk, pr_apply], pr_out)
            gr.Markdown("### LoRA A/B（需 LM Studio 依次加载各模型）")
            ab_eval = gr.Textbox(value="data/promo/promo_eval.jsonl", label="评测集")
            ab_tag = gr.Textbox(value="promo", label="LoRA 标签")
            btn_ab = gr.Button("生成 A/B 报告")
            ab_out = gr.Textbox(lines=10)
            btn_ab.click(ui_lora_ab, [ab_eval, ab_tag], ab_out)
        with gr.Tab("对齐流水线"):
            gr.Markdown("SFT (QLoRA) → DPO，可选跳过步骤")
            al_sk_sft = gr.Checkbox(label="跳过 SFT", value=False)
            al_sk_dpo = gr.Checkbox(label="跳过 DPO", value=False)
            btn_al = gr.Button("运行 SFT→DPO", variant="primary")
            al_out = gr.Textbox(lines=15)
            btn_al.click(ui_alignment, [al_sk_sft, al_sk_dpo], al_out)
        with gr.Tab("抓取造数"):
            crawl_urls_in = gr.Textbox(lines=4, label="URL（每行一个）", placeholder="https://example.com/docs")
            crawl_gen = gr.Checkbox(label="抓取后 LM Studio 造数", value=False)
            btn_crawl = gr.Button("开始抓取")
            crawl_out = gr.Textbox(lines=12)
            btn_crawl.click(ui_crawl, [crawl_urls_in, crawl_gen], crawl_out)
        with gr.Tab("Hub / 用量"):
            hub_repo = gr.Textbox(label="HF repo_id", placeholder="username/my-model")
            btn_hub = gr.Button("上传到 HuggingFace Hub")
            btn_usage = gr.Button("API 用量统计")
            hub_out = gr.Textbox(lines=12)
            btn_hub.click(ui_hub_upload, hub_repo, hub_out)
            btn_usage.click(ui_usage, outputs=hub_out)
        with gr.Tab("部署"):
            dep_out = gr.Textbox(lines=12, label="输出")
            with gr.Row():
                btn_api = gr.Button("启动 FastAPI（子进程）")
                btn_tb = gr.Button("打开 TensorBoard 说明")
            btn_api.click(
                lambda: _run_cmd([sys.executable, str(ROOT / "serve_api.py")]),
                outputs=dep_out,
            )
            btn_tb.click(
                lambda: "在项目目录运行: tensorboard --logdir output/run1",
                outputs=dep_out,
            )
            gr.Markdown("""
**FastAPI**: http://127.0.0.1:8000/v1/chat/completions | **Gradio**: http://127.0.0.1:7860
**Docker**: `docker compose up web api` | **Ollama**: `python deploy_ollama.py`
**vLLM**: `python deploy_vllm.py --dry-run` | **发布检查**: `python release_check.py --markdown`
            """)
        with gr.Tab("LoRA"):
            btn_lora = gr.Button("刷新注册表")
            lora_md = gr.Markdown()
            btn_lora.click(lambda: registry_to_markdown() or "(空)", outputs=lora_md)
            app.load(ui_lora_list, outputs=lora_md)
    return app


def main() -> None:
    import os

    port = int(os.environ.get("PORT", "7860"))
    build_app().launch(server_name="127.0.0.1", server_port=port, show_error=True)


if __name__ == "__main__":
    main()
