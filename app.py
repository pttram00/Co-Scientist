"""Giao diện Gradio cho Research Co-Scientist.

Chạy:  python app.py            (mở http://127.0.0.1:7860)
       python app.py --share    (tạo link chia sẻ tạm thời)

Các tính năng:
    - Nhập mục tiêu nghiên cứu + ràng buộc, chỉnh tham số vòng lặp.
    - Chạy pipeline 3 pha và xem log trực tiếp; dừng giữa chừng được.
    - Xem bảng giả thuyết (Elo, trạng thái, điểm review) và chi tiết từng giả thuyết.
    - Đọc báo cáo cuối, tải về final_report.md / state.json.
    - Nạp lại một state.json cũ để xem kết quả mà không cần chạy lại.
    - Chế độ mô phỏng: chạy thử toàn bộ luồng không cần API key (nội dung là giả).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import queue
import time
from pathlib import Path
from typing import List, Tuple

# Tắt gửi thống kê sử dụng của Gradio ra máy chủ ngoài (phải đặt trước khi import gradio).
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

import sim_backend
from agents.safety_agent import UnsafeResearchGoalError
from config import AppConfig, EmbeddingConfig, LLMConfig, OrchestratorConfig
from memory.context_memory import ContextMemory
from models.hypothesis import Hypothesis, HypothesisStatus
from orchestrator import Orchestrator

MODE_SIM = "Mô phỏng — không cần API key"
MODE_REAL = "Thật — dùng .env"

STATUS_LABEL = {
    HypothesisStatus.ACTIVE: "đang dùng",
    HypothesisStatus.DUPLICATE: "trùng lặp",
    HypothesisStatus.ARCHIVED: "bị loại",
    HypothesisStatus.UNSAFE: "không an toàn",
    HypothesisStatus.EVOLVED: "đã tiến hoá",
}
HEADERS = ["ID", "Trạng thái", "Nguồn", "Elo", "Trận", "Đúng đắn", "Mới", "Khả thi", "Giả thuyết"]
# Log của các module trong dự án (bỏ qua log của gradio, httpx, urllib3...)
PROJECT_LOGGERS = ("orchestrator", "generation_agent", "proximity_agent", "reflection_agent",
                   "ranking_agent", "evolution_agent", "meta_review_agent", "retriever")


# --------------------------------------------------------------- thu log
class QueueLogHandler(logging.Handler):
    def __init__(self, q: "queue.SimpleQueue[str]"):
        super().__init__(level=logging.INFO)
        self.q = q
        self.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))

    def filter(self, record):
        return record.name.split(".")[0] in PROJECT_LOGGERS

    def emit(self, record):
        try:
            self.q.put(self.format(record))
        except Exception:
            pass


def _drain(q: "queue.SimpleQueue[str]") -> List[str]:
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


# --------------------------------------------------------------- hiển thị
def _rows(memory: ContextMemory) -> Tuple[List[list], List[str]]:
    order = {HypothesisStatus.ACTIVE: 0, HypothesisStatus.DUPLICATE: 1,
             HypothesisStatus.ARCHIVED: 2, HypothesisStatus.UNSAFE: 3, HypothesisStatus.EVOLVED: 4}
    hyps = sorted(memory.hypotheses.values(), key=lambda h: (order.get(h.status, 9), -h.elo_rating))
    rows, ids = [], []
    for h in hyps:
        ids.append(h.id)
        rows.append([h.id, STATUS_LABEL.get(h.status, h.status.value), h.strategy.value,
                     round(h.elo_rating), h.matches_played,
                     round(h.average_score("correctness"), 1), round(h.average_score("novelty"), 1),
                     round(h.average_score("feasibility"), 1), h.content])
    return rows, ids


def _detail_md(h: Hypothesis) -> str:
    parts = [f"### {h.content}",
             f"`{h.id}` · **{STATUS_LABEL.get(h.status, h.status.value)}** · nguồn: `{h.strategy.value}` · "
             f"Elo **{h.elo_rating:.0f}** sau {h.matches_played} trận"]
    if h.parent_ids:
        parts.append(f"Tiến hoá từ: {', '.join('`' + p + '`' for p in h.parent_ids)}")
    parts.append(f"**Cơ chế đề xuất.** {h.rationale}")
    parts.append(f"**Thí nghiệm đề xuất.** {h.suggested_experiment or '(chưa có)'}")
    if h.reviews:
        parts.append("#### Phản biện")
        for r in h.reviews:
            parts.append(f"**{r.review_type}** — đúng đắn {r.correctness:.0f}/10 · mới {r.novelty:.0f}/10 · "
                         f"khả thi {r.feasibility:.0f}/10\n\n{r.comments}")
            if r.references:
                parts.append("Tài liệu đã tra cứu: " + "; ".join(r.references))
    else:
        parts.append("_Chưa có phản biện nào._")
    return "\n\n".join(parts)


def _stats_md(memory: ContextMemory, extra: str = "") -> str:
    n = {s: sum(1 for h in memory.hypotheses.values() if h.status == s) for s in HypothesisStatus}
    edges = sum(len(v) for v in memory.proximity_graph.values()) // 2
    line = (f"**{len(memory.hypotheses)}** giả thuyết · **{n[HypothesisStatus.ACTIVE]}** đang dùng · "
            f"{n[HypothesisStatus.DUPLICATE]} trùng lặp · {n[HypothesisStatus.ARCHIVED]} bị loại · "
            f"{n[HypothesisStatus.UNSAFE]} không an toàn  \n"
            f"**{len(memory.match_history)}** trận đấu · **{edges}** cặp đã so tương đồng · "
            f"**{len(memory.papers)}** bài báo tra cứu · **{memory.iteration}** vòng lặp")
    return line + (f"  \n{extra}" if extra else "")


def _notes_md(memory: ContextMemory) -> str:
    parts = []
    if memory.safety_alerts:
        parts.append("**Cảnh báo an toàn**\n" + "\n".join(f"- {a}" for a in memory.safety_alerts[-10:]))
    if memory.meta_review_notes:
        parts.append("**Mẫu hình phản biện (Meta-review)**\n" + "\n".join(f"- {n}" for n in memory.meta_review_notes[-10:]))
    if memory.research_overview:
        ov = memory.research_overview
        parts.append("**Research overview**\n\n" + (ov[:600] + "…" if len(ov) > 600 else ov))
    return "\n\n".join(parts) if parts else "_Chưa có ghi chú nào._"


def _outputs(memory: ContextMemory, out_dir: Path, extra_stats: str = ""):
    """Trả về (stats, rows, ids, details, report_md, files) cho các ô kết quả."""
    rows, ids = _rows(memory)
    details = {hid: _detail_md(memory.hypotheses[hid]) for hid in ids}
    report_path = out_dir / "final_report.md"
    state_path = out_dir / "state.json"
    report_md = report_path.read_text(encoding="utf-8") if report_path.exists() else "_Chưa có báo cáo._"
    files = [str(p) for p in (report_path, state_path) if p.exists()]
    return _stats_md(memory, extra_stats), rows, ids, details, report_md, (files or None)


# --------------------------------------------------------------- chạy pipeline
async def run_pipeline(goal, constraints, mode, model, emb_model, iterations, n_hyp, n_matches,
                       top_k, max_checks, out_dir):
    log_lines: List[str] = []
    empty = ("", [], [], {}, "_Chưa có báo cáo._", None)   # stats, rows, ids, details, report, files

    if not (goal or "").strip():
        yield ("⚠️ Hãy nhập mục tiêu nghiên cứu trước khi chạy.", "", *empty, "", "")
        return

    out_path = Path((out_dir or "output").strip())
    cfg = AppConfig(
        llm=LLMConfig(model=(model or "GLM-5.2").strip()),
        orchestrator=OrchestratorConfig(
            n_iterations=int(iterations), hypotheses_per_iteration=int(n_hyp),
            matches_per_iteration=int(n_matches), top_k_for_evolution=int(top_k),
            proximity_max_llm_checks_per_iteration=int(max_checks), output_dir=str(out_path)),
        embedding=EmbeddingConfig(model_name=(emb_model or EmbeddingConfig.model_name).strip()),
    )
    orch = Orchestrator(research_goal=goal.strip(), constraints=(constraints or "").strip(), config=cfg)
    sim = mode == MODE_SIM
    if sim:
        sim_backend.install(orch)

    q: "queue.SimpleQueue[str]" = queue.SimpleQueue()
    handler = QueueLogHandler(q)
    logging.getLogger().addHandler(handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    started = time.time()
    head = "🔄 Đang chạy ở chế độ mô phỏng…" if sim else "🔄 Đang chạy (gọi LLM thật)…"
    yield (head, "", *empty, "", "")

    task = asyncio.create_task(orch.run())
    try:
        while True:
            done = task.done()
            new = _drain(q)
            if new:
                log_lines.extend(new)
                yield (f"{head}  ·  {time.time() - started:.0f}s", "\n".join(log_lines[-400:]), *empty, "", "")
            if done:
                break
            await asyncio.sleep(0.3)

        log_lines.extend(_drain(q))
        task.result()   # ném lại lỗi nếu có
        took = time.time() - started
        extra = ("_Chế độ mô phỏng: nội dung giả thuyết do LLM giả sinh ra, không có giá trị khoa học._"
                 if sim else "")
        stats, rows, ids, details, report_md, files = _outputs(orch.memory, out_path, extra)
        yield (f"✅ Hoàn tất sau {took:.0f}s. Kết quả nằm trong `{out_path}`.",
               "\n".join(log_lines[-400:]), stats, rows, ids, details, report_md, files,
               _notes_md(orch.memory), "_Chọn một dòng trong bảng để xem chi tiết._")

    except UnsafeResearchGoalError as e:
        log_lines.extend(_drain(q))
        yield (f"🚫 Mục tiêu nghiên cứu bị từ chối vì lý do an toàn: {e}",
               "\n".join(log_lines[-400:]), *empty, "", "")
    except asyncio.CancelledError:
        raise
    except Exception as e:   # lỗi LLM, thiếu thư viện embedding, thiếu token...
        log_lines.extend(_drain(q))
        stats, rows, ids, details, report_md, files = _outputs(orch.memory, out_path)
        yield (f"❌ Lỗi: {e}", "\n".join(log_lines[-400:]), stats, rows, ids, details, report_md, files,
               _notes_md(orch.memory), "")
    finally:
        logging.getLogger().removeHandler(handler)
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def load_state(file_obj, path_text):
    blank = ("", [], [], {}, "_Chưa có báo cáo._", None, "", "")
    raw = (file_obj.name if file_obj else (path_text or "").strip())
    if not raw:
        return ("⚠️ Hãy chọn file state.json hoặc nhập đường dẫn.", *blank)
    path = Path(raw)
    if not path.exists():
        return (f"❌ Không tìm thấy `{path}`.", *blank)
    try:
        memory = ContextMemory.load(str(path))
    except Exception as e:
        return (f"❌ Không đọc được state: {e}", *blank)
    stats, rows, ids, details, report_md, files = _outputs(memory, path.parent)
    return (f"✅ Đã nạp `{path}` — mục tiêu: {memory.research_goal}",
            stats, rows, ids, details, report_md, files, _notes_md(memory),
            "_Chọn một dòng trong bảng để xem chi tiết._")


# --------------------------------------------------------------- giao diện
THEME = gr.themes.Soft(primary_hue="emerald", secondary_hue="emerald", neutral_hue="slate")


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Research Co-Scientist", fill_height=True) as demo:
        gr.Markdown(
            "# Research Co-Scientist\n"
            "Hệ thống nhiều agent cùng sinh, phản biện, xếp hạng và cải tiến giả thuyết khoa học. "
            "Chọn **chế độ mô phỏng** để xem toàn bộ luồng chạy mà không cần API key."
        )
        memory_ids = gr.State([])      # id giả thuyết theo thứ tự dòng trong bảng
        details_state = gr.State({})   # id -> markdown chi tiết

        with gr.Tabs():
            # ------------------------------------------------ tab chạy
            with gr.Tab("Chạy nghiên cứu"):
                with gr.Row():
                    with gr.Column(scale=3):
                        goal = gr.Textbox(label="Mục tiêu nghiên cứu", lines=3, max_lines=6,
                                          placeholder="Ví dụ: Tìm cơ chế phân tử mới để ức chế sự già hoá tế bào thần kinh")
                        constraints = gr.Textbox(label="Ràng buộc / bối cảnh (tuỳ chọn)", lines=2,
                                                 placeholder="Ví dụ: chỉ dùng mô hình in vitro, ưu tiên thuốc đã được cấp phép")
                    with gr.Column(scale=2):
                        mode = gr.Radio([MODE_SIM, MODE_REAL], value=MODE_SIM, label="Chế độ chạy",
                                        info="Chế độ thật cần ANTHROPIC_AUTH_TOKEN trong .env và thư viện embedding.")
                        model = gr.Textbox(label="Model LLM", value="GLM-5.2")
                        emb_model = gr.Textbox(label="Model embedding (Proximity)",
                                               value=EmbeddingConfig.model_name)

                with gr.Accordion("Tham số vòng lặp", open=False):
                    with gr.Row():
                        iterations = gr.Number(label="Số vòng lặp", value=1, precision=0, minimum=1, maximum=10)
                        n_hyp = gr.Number(label="Giả thuyết mỗi vòng", value=3, precision=0, minimum=1, maximum=20)
                        n_matches = gr.Number(label="Trận đấu mỗi vòng", value=3, precision=0, minimum=1, maximum=50)
                        top_k = gr.Number(label="Top-k cho Evolution", value=2, precision=0, minimum=1, maximum=10)
                        max_checks = gr.Number(label="Số cặp nghi trùng hỏi LLM", value=10, precision=0, minimum=0, maximum=50)
                        out_dir = gr.Textbox(label="Thư mục kết quả", value="output")

                with gr.Row():
                    run_btn = gr.Button("Chạy", variant="primary", scale=2)
                    stop_btn = gr.Button("Dừng", variant="stop", scale=1)
                status = gr.Markdown("Sẵn sàng. Nhập mục tiêu nghiên cứu rồi bấm **Chạy**.")
                log = gr.Textbox(label="Nhật ký chạy", lines=16, max_lines=16, autoscroll=True,
                                 interactive=False)

            # ------------------------------------------------ tab kết quả
            with gr.Tab("Kết quả"):
                stats = gr.Markdown("_Chưa có kết quả._")
                with gr.Row():
                    table = gr.Dataframe(headers=HEADERS, label="Giả thuyết (bấm một dòng để xem chi tiết)",
                                         datatype=["str", "str", "str", "number", "number", "number", "number", "number", "str"],
                                         interactive=False, wrap=True, scale=3,
                                         column_widths=["8%", "10%", "14%", "7%", "6%", "8%", "6%", "7%", "34%"])
                    detail = gr.Markdown("_Chọn một dòng trong bảng để xem chi tiết._")
                with gr.Accordion("Cảnh báo an toàn · ghi chú Meta-review", open=False):
                    notes = gr.Markdown("_Chưa có ghi chú nào._")
                files = gr.File(label="Tải về", interactive=False)

            # ------------------------------------------------ tab báo cáo
            with gr.Tab("Báo cáo"):
                report = gr.Markdown("_Chưa có báo cáo._")

            # ------------------------------------------------ tab nạp state
            with gr.Tab("Nạp kết quả cũ"):
                gr.Markdown("Nạp lại `state.json` của một lần chạy trước để xem kết quả mà không cần chạy lại.")
                with gr.Row():
                    state_file = gr.File(label="Chọn state.json", file_types=[".json"])
                    state_path = gr.Textbox(label="hoặc nhập đường dẫn", value="output/state.json")
                load_btn = gr.Button("Nạp kết quả", variant="primary")
                load_status = gr.Markdown()

        run_outputs = [status, log, stats, table, memory_ids, details_state, report, files, notes, detail]
        run_event = run_btn.click(
            fn=run_pipeline,
            inputs=[goal, constraints, mode, model, emb_model, iterations, n_hyp, n_matches, top_k, max_checks, out_dir],
            outputs=run_outputs,
        )
        stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[run_event])

        load_btn.click(fn=load_state, inputs=[state_file, state_path],
                       outputs=[load_status, stats, table, memory_ids, details_state, report, files, notes, detail])

        def on_select(ids, details, evt: gr.SelectData):
            if not ids or evt.index is None:
                return "_Chọn một dòng trong bảng để xem chi tiết._"
            row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            if row >= len(ids):
                return "_Không tìm thấy giả thuyết._"
            return (details or {}).get(ids[row], "_Không tìm thấy giả thuyết._")

        table.select(fn=on_select, inputs=[memory_ids, details_state], outputs=[detail])
    return demo


def main():
    p = argparse.ArgumentParser(description="Giao diện Gradio cho Research Co-Scientist")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Tạo link chia sẻ tạm thời của Gradio")
    args = p.parse_args()
    build_ui().launch(server_name=args.host, server_port=args.port, share=args.share, theme=THEME)


if __name__ == "__main__":
    main()
