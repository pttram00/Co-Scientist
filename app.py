"""Giao diện Gradio cho Research Co-Scientist.

Chạy:  python app.py            (mở http://127.0.0.1:7860)
       python app.py --share    (tạo link chia sẻ tạm thời)

Các tính năng:
    - Nhập mục tiêu nghiên cứu + ràng buộc, chỉnh tham số vòng lặp.
    - Chạy pipeline 3 pha và xem log trực tiếp; dừng giữa chừng được.
    - Xem bảng giả thuyết (Elo, trạng thái, điểm review) và chi tiết từng giả thuyết.
    - Đọc báo cáo cuối, tải về final_report.md / state.json.
    - Nạp lại một state.json cũ để xem kết quả mà không cần chạy lại.
    - Trò chuyện & góp ý: góp ý vào giả thuyết/agent, rồi chạy lại để hệ thống tiếp thu.
    - Chế độ mô phỏng: chạy thử toàn bộ luồng không cần API key (nội dung là giả).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Tắt gửi thống kê sử dụng của Gradio ra máy chủ ngoài (phải đặt trước khi import gradio).
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

import sim_backend
from chatbot import helpers as chat_helpers
from chatbot.chat_agent import answer as rag_answer
from chatbot.chat_agent import handle_resume_newfinal, handle_review
from chatbot.intent import IntentResult, classify_intent
from config import AppConfig, ChatbotConfig, LLMConfig, OrchestratorConfig
from llm.client import LLMClient
from memory.context_memory import ContextMemory
from models.hypothesis import Hypothesis, HypothesisStatus
from orchestrator import Orchestrator

MODE_SIM = "Mô phỏng — không cần API key"
MODE_REAL = "Thật — dùng .env"

STATUS_LABEL = {
    HypothesisStatus.ACTIVE: "đang dùng",
    HypothesisStatus.DUPLICATE: "trùng lặp",
    HypothesisStatus.ARCHIVED: "bị loại",
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
             HypothesisStatus.ARCHIVED: 2, HypothesisStatus.EVOLVED: 3}
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
    else:
        parts.append("_Chưa có phản biện nào._")
    return "\n\n".join(parts)


def _stats_md(memory: ContextMemory, extra: str = "") -> str:
    n = {s: sum(1 for h in memory.hypotheses.values() if h.status == s) for s in HypothesisStatus}
    edges = sum(len(v) for v in memory.proximity_graph.values()) // 2
    line = (f"**{len(memory.hypotheses)}** giả thuyết · **{n[HypothesisStatus.ACTIVE]}** đang dùng · "
            f"{n[HypothesisStatus.DUPLICATE]} trùng lặp · {n[HypothesisStatus.ARCHIVED]} bị loại  \n"
            f"**{len(memory.match_history)}** trận đấu · **{edges}** cặp đã so tương đồng · "
            f"**{len(memory.papers)}** bài báo tra cứu · **{memory.iteration}** vòng lặp")
    return line + (f"  \n{extra}" if extra else "")


def _notes_md(memory: ContextMemory) -> str:
    parts = []
    if memory.meta_review_notes:
        parts.append("**Mẫu hình phản biện (Meta-review)**\n" + "\n".join(f"- {n}" for n in memory.meta_review_notes[-10:]))
    if memory.agent_feedback:
        lines = []
        for agent, notes in memory.agent_feedback.items():
            for n in notes[-3:]:
                lines.append(f"- **{agent}**: {n}")
        if lines:
            parts.append("**Góp ý cho từng agent** (gồm cả góp ý của bạn)\n" + "\n".join(lines))
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
                       top_k, dup_threshold, out_dir):
    log_lines: List[str] = []
    empty = ("", [], [], {}, "_Chưa có báo cáo._", None)   # stats, rows, ids, details, report, files

    if not (goal or "").strip():
        yield ("⚠️ Hãy nhập mục tiêu nghiên cứu trước khi chạy.", "", *empty, "", "",
               None, gr.update())
        return

    # Ô Number bị xoá trắng -> Gradio gửi None; ô thư mục trống -> Path("") = "." (gốc repo).
    out_path = Path((out_dir or "").strip() or "output")
    try:
        cfg = AppConfig(
            llm=LLMConfig(model=(model or "").strip()) if (model or "").strip() else LLMConfig(),
            orchestrator=OrchestratorConfig(
                n_iterations=int(iterations or 1), hypotheses_per_iteration=int(n_hyp or 3),
                matches_per_iteration=int(n_matches or 3), top_k_for_evolution=int(top_k or 2),
                proximity_duplicate_threshold=float(dup_threshold or 0.85),
                output_dir=str(out_path)),
            # Model embedding chỉ dùng cho kho RAG của chatbot; ProximityAgent ở
            # nhánh này chấm tương đồng bằng LLM nên không cần embedding.
            chatbot=ChatbotConfig(
                embedding_model=(emb_model or ChatbotConfig.embedding_model).strip(),
                vector_store_path=str(out_path / "vectors.json"),
                state_path=str(out_path / "state.json"),
                report_path=str(out_path / "final_report.md")),
        )
    except (TypeError, ValueError) as e:
        yield (f"⚠️ Tham số vòng lặp không hợp lệ: {e}", "", *empty, "", "", None, gr.update())
        return

    if mode == MODE_REAL and not cfg.llm.api_key:
        yield ("⚠️ Chế độ Thật cần ANTHROPIC_AUTH_TOKEN trong file `.env`. "
               "Hãy điền token rồi chạy lại, hoặc chuyển sang chế độ mô phỏng.",
               "", *empty, "", "", None, gr.update())
        return
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
    yield (head, "", *empty, "", "", None, gr.update())

    task = asyncio.create_task(orch.run())
    try:
        while True:
            done = task.done()
            new = _drain(q)
            if new:
                log_lines.extend(new)
                yield (f"{head}  ·  {time.time() - started:.0f}s", "\n".join(log_lines[-400:]),
                       *empty, "", "", None, gr.update())
            if done:
                break
            await asyncio.sleep(0.3)

        log_lines.extend(_drain(q))
        task.result()   # ném lại lỗi nếu có
        took = time.time() - started
        extra = ("_Chế độ mô phỏng: nội dung giả thuyết do LLM giả sinh ra, không có giá trị khoa học._"
                 if sim else "")
        stats, rows, ids, details, report_md, files = _outputs(orch.memory, out_path, extra)
        session = _make_session(orch.memory, out_path, cfg, sim)
        yield (f"✅ Hoàn tất sau {took:.0f}s. Kết quả nằm trong `{out_path}`.",
               "\n".join(log_lines[-400:]), stats, rows, ids, details, report_md, files,
               _notes_md(orch.memory), "_Chọn một dòng trong bảng để xem chi tiết._",
               session, gr.update(choices=_target_choices(session), value=TARGET_AUTO))

    except asyncio.CancelledError:
        raise
    except Exception as e:   # lỗi LLM, thiếu thư viện embedding, thiếu token...
        log_lines.extend(_drain(q))
        stats, rows, ids, details, report_md, files = _outputs(orch.memory, out_path)
        # Vẫn mở phiên trò chuyện trên dữ liệu dở dang: người dùng còn xem/hỏi được.
        session = _make_session(orch.memory, out_path, cfg, sim)
        yield (f"❌ Lỗi: {e}", "\n".join(log_lines[-400:]), stats, rows, ids, details, report_md, files,
               _notes_md(orch.memory), "",
               session, gr.update(choices=_target_choices(session), value=TARGET_AUTO))
    finally:
        logging.getLogger().removeHandler(handler)
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def load_state(file_obj, path_text, mode, model, emb_model):
    blank = ("", [], [], {}, "_Chưa có báo cáo._", None, "", "")
    nosession = (None, gr.update(), _session_banner(None))
    raw = (file_obj.name if file_obj else (path_text or "").strip())
    if not raw:
        return ("⚠️ Hãy chọn file state.json hoặc nhập đường dẫn.", *blank, *nosession)
    path = Path(raw)
    if not path.exists():
        return (f"❌ Không tìm thấy `{path}`.", *blank, *nosession)
    try:
        memory = ContextMemory.load(str(path))
    except Exception as e:
        return (f"❌ Không đọc được state: {e}", *blank, *nosession)

    stats, rows, ids, details, report_md, files = _outputs(memory, path.parent)
    # Mở luôn phiên trò chuyện trên state vừa nạp: góp ý + chạy lại dùng được ngay.
    cfg = AppConfig(
        llm=LLMConfig(model=(model or "").strip()) if (model or "").strip() else LLMConfig(),
        orchestrator=OrchestratorConfig(output_dir=str(path.parent)),
        chatbot=ChatbotConfig(
            embedding_model=(emb_model or ChatbotConfig.embedding_model).strip(),
            vector_store_path=str(path.parent / "vectors.json"),
            state_path=str(path.parent / "state.json"),
            report_path=str(path.parent / "final_report.md")),
    )
    session = _make_session(memory, path.parent, cfg, mode == MODE_SIM)
    return (f"✅ Đã nạp `{path}` — mục tiêu: {memory.research_goal}",
            stats, rows, ids, details, report_md, files, _notes_md(memory),
            "_Chọn một dòng trong bảng để xem chi tiết._",
            session, gr.update(choices=_target_choices(session), value=TARGET_AUTO),
            _session_banner(session))


# --------------------------------------------------------------- phiên trò chuyện
AGENT_NAMES = ["generation_agent", "proximity_agent", "reflection_agent",
               "ranking_agent", "evolution_agent", "meta_review_agent"]
TARGET_AUTO = "— Tự động: để hệ thống tự phân loại —"
TARGET_GENERAL = "Cả hướng nghiên cứu (góp ý chung)"


@dataclass
class ChatSession:
    """Trạng thái sống của một phiên: bộ nhớ nghiên cứu + kho RAG + client LLM.

    Giữ trong gr.State nên mỗi tab trình duyệt có phiên riêng, không giẫm lên nhau.
    """
    memory: ContextMemory
    out_dir: Path
    cfg: AppConfig
    sim: bool
    llm: object                      # LLMClient (đã vá bằng LLM giả nếu sim)
    encoder: Optional[object] = None  # hàm encode cho VectorStore (sim -> giả)
    store: object = None             # VectorStore, dựng lười ở câu hỏi đầu tiên

    @property
    def state_path(self) -> str:
        return str(self.out_dir / "state.json")

    @property
    def report_path(self) -> str:
        return str(self.out_dir / "final_report.md")

    @property
    def store_path(self) -> str:
        return str(self.out_dir / "vectors.json")

    def ensure_store(self):
        """Dựng index RAG lần đầu cần dùng. Ở chế độ thật việc này nạp model
        embedding nên chỉ làm khi người dùng thực sự hỏi."""
        if self.store is None:
            self.store = chat_helpers.rebuild_store(
                self.memory, self.store_path, self.report_path,
                self.cfg.chatbot.embedding_model, self.encoder)
        return self.store

    def invalidate_store(self):
        """Dữ liệu đã đổi -> index cũ hết giá trị, dựng lại ở lần hỏi sau."""
        self.store = None


def _make_session(memory: ContextMemory, out_dir: Path, cfg: AppConfig, sim: bool) -> ChatSession:
    llm = LLMClient(cfg.llm)
    encoder = None
    if sim:
        sim_backend.install_llm(llm)
        encoder = sim_backend.fake_encoder
    return ChatSession(memory=memory, out_dir=Path(out_dir), cfg=cfg, sim=sim,
                       llm=llm, encoder=encoder)


def _target_choices(session: Optional[ChatSession]) -> List[str]:
    """Danh sách đối tượng để gắn góp ý: tự động / chung / từng giả thuyết / từng agent."""
    opts = [TARGET_AUTO, TARGET_GENERAL]
    if session is not None:
        for h in sorted(session.memory.hypotheses.values(),
                        key=lambda x: -x.elo_rating):
            short = h.content.replace("\n", " ")[:55]
            opts.append(f"{h.id} — {short}")
        opts += [f"agent: {a}" for a in AGENT_NAMES]
    return opts


def _session_banner(session: Optional[ChatSession]) -> str:
    if session is None:
        return ("_Chưa có phiên nào._ Hãy chạy ở tab **Chạy nghiên cứu**, "
                "hoặc nạp `state.json` cũ ở tab **Nạp kết quả cũ**.")
    m = session.memory
    has_report = Path(session.report_path).exists()
    mode = " · chế độ mô phỏng" if session.sim else ""
    return (f"**Mục tiêu:** {m.research_goal}  \n"
            f"{len(m.hypotheses)} giả thuyết · vòng lặp {m.iteration} · "
            f"{'đã có báo cáo' if has_report else 'chưa có báo cáo'}{mode}")


def _intent_from_target(target: str, message: str) -> Optional[IntentResult]:
    """Người dùng đã chọn đối tượng cụ thể -> khỏi tốn một lượt gọi LLM phân loại."""
    if target == TARGET_GENERAL:
        return IntentResult(kind="review", comment_text=message)
    if target and target.startswith("agent: "):
        return IntentResult(kind="review", target_agent=target[len("agent: "):],
                            comment_text=message)
    if target and target != TARGET_AUTO:
        return IntentResult(kind="review",
                            target_hypothesis_id=target.split(" — ")[0].strip(),
                            comment_text=message)
    return None


async def on_send(message: str, history: list, session: Optional[ChatSession], target: str):
    """Gửi một câu trong ô chat. Định tuyến: góp ý / hỏi đáp RAG / chạy lại."""
    message = (message or "").strip()
    history = list(history or [])
    if not message:
        yield history, "", session, gr.update()
        return

    history.append({"role": "user", "content": message})
    if session is None:
        history.append({"role": "assistant", "content":
                        "Chưa có phiên nghiên cứu nào. Hãy chạy ở tab **Chạy nghiên cứu** "
                        "hoặc nạp `state.json` ở tab **Nạp kết quả cũ** trước đã."})
        yield history, "", session, gr.update()
        return

    history.append({"role": "assistant", "content": "_(đang xử lý…)_"})
    yield history, "", session, gr.update()

    try:
        intent = _intent_from_target(target, message)
        if intent is None:
            intent = await classify_intent(message, session.llm, session.memory)

        if intent.kind == "question":
            store = await asyncio.to_thread(session.ensure_store)
            reply = await rag_answer(message, store, session.llm,
                                     top_k=session.cfg.chatbot.top_k)
        elif intent.kind == "resume_newfinal":
            reply = ("Để chạy lại, bấm nút **Chạy lại với góp ý** bên cạnh — nút đó "
                     "hiện nhật ký chạy trực tiếp và cập nhật luôn bảng kết quả.")
        elif intent.kind == "new_topic":
            reply = ("Đổi chủ đề thì dùng tab **Chạy nghiên cứu** (nhập mục tiêu mới rồi "
                     "bấm Chạy) — dữ liệu phiên này sẽ được thay bằng phiên mới.")
        elif intent.kind in ("review", "rerun"):
            reply = await handle_review(intent, session.memory, session.llm,
                                        session.state_path)
            session.invalidate_store()
            reply += ("\n\nGóp ý đã được ghi vào bộ nhớ. Bấm **Chạy lại với góp ý** "
                      "khi bạn đã góp ý xong.")
        else:
            reply = ("Mình chưa rõ ý bạn. Bạn có thể: góp ý về một giả thuyết (chọn nó ở ô "
                     "*Gắn góp ý vào*), hỏi về kết quả đã có, hoặc bấm **Chạy lại với góp ý**.")
    except Exception as e:
        reply = f"❌ Lỗi: {e}"

    history[-1] = {"role": "assistant", "content": reply}
    yield history, "", session, gr.update(choices=_target_choices(session))


async def on_rerun(n_iterations, session: Optional[ChatSession], history: list):
    """Nút 'Chạy lại với góp ý': chạy thêm n vòng trên state hiện có rồi viết báo cáo mới."""
    history = list(history or [])
    blank = ("", [], [], {}, "_Chưa có báo cáo._", None)
    if session is None:
        yield ("⚠️ Chưa có phiên nào để chạy lại.", "", *blank, "", history, gr.update())
        return

    n = max(1, int(n_iterations or 1))
    q: "queue.SimpleQueue[str]" = queue.SimpleQueue()
    handler = QueueLogHandler(q)
    logging.getLogger().addHandler(handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    log_lines: List[str] = []
    started = time.time()

    head = f"🔄 Đang chạy thêm {n} vòng, tiếp thu góp ý của bạn…"
    history.append({"role": "assistant", "content": head})
    yield (head, "", *blank, "", history, gr.update())

    # Che do mo phong: gan LLM/retriever/encoder gia vao Orchestrator resume.
    hook = (lambda orch: sim_backend.install(orch)) if session.sim else None
    task = asyncio.create_task(handle_resume_newfinal(
        n, session.cfg, session.memory, session.state_path, session.report_path,
        session.store_path, session.cfg.chatbot.embedding_model, session.encoder,
        on_orchestrator=hook))
    try:
        while True:
            done = task.done()
            new = _drain(q)
            if new:
                log_lines.extend(new)
                yield (f"{head}  ·  {time.time() - started:.0f}s",
                       "\n".join(log_lines[-400:]), *blank, "", history, gr.update())
            if done:
                break
            await asyncio.sleep(0.3)

        log_lines.extend(_drain(q))
        memory2, store, msg = task.result()
        session.memory, session.store = memory2, store
        took = time.time() - started

        extra = ("_Chế độ mô phỏng: nội dung giả thuyết do LLM giả sinh ra, không có giá trị khoa học._"
                 if session.sim else "")
        stats, rows, ids, details, report_md, files = _outputs(memory2, session.out_dir, extra)
        history[-1] = {"role": "assistant", "content":
                       f"{msg}\n\nBáo cáo đã được viết lại — xem tab **Báo cáo**."}
        yield (f"✅ {msg} ({took:.0f}s)", "\n".join(log_lines[-400:]),
               stats, rows, ids, details, report_md, files,
               _notes_md(memory2), history, gr.update(choices=_target_choices(session)))

    except asyncio.CancelledError:
        raise
    except Exception as e:
        log_lines.extend(_drain(q))
        history[-1] = {"role": "assistant", "content": f"❌ Chạy lại thất bại: {e}"}
        stats, rows, ids, details, report_md, files = _outputs(session.memory, session.out_dir)
        yield (f"❌ Lỗi: {e}", "\n".join(log_lines[-400:]), stats, rows, ids, details,
               report_md, files, _notes_md(session.memory), history, gr.update())
    finally:
        logging.getLogger().removeHandler(handler)
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


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
        session_state = gr.State(None)  # ChatSession: bộ nhớ sống cho trò chuyện + chạy lại

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
                        model = gr.Textbox(
                            label="Model LLM", value="",
                            placeholder="de trong -> lay MODEL_DEFAULT trong .env")
                        emb_model = gr.Textbox(label="Model embedding (kho RAG cua chatbot)",
                                               value=ChatbotConfig.embedding_model)

                with gr.Accordion("Tham số vòng lặp", open=False):
                    with gr.Row():
                        iterations = gr.Number(label="Số vòng lặp", value=1, precision=0, minimum=1, maximum=10)
                        n_hyp = gr.Number(label="Giả thuyết mỗi vòng", value=3, precision=0, minimum=1, maximum=20)
                        n_matches = gr.Number(label="Trận đấu mỗi vòng", value=3, precision=0, minimum=1, maximum=50)
                        top_k = gr.Number(label="Top-k cho Evolution", value=2, precision=0, minimum=1, maximum=10)
                        dup_threshold = gr.Number(label="Ngưỡng coi là trùng lặp", value=0.85,
                                                  minimum=0.0, maximum=1.0, step=0.05)
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
                files = gr.File(label="Tải về", interactive=False, file_count="multiple")

            # ------------------------------------------------ tab báo cáo
            with gr.Tab("Báo cáo"):
                report = gr.Markdown("_Chưa có báo cáo._")

            # ------------------------------------------------ tab trò chuyện & góp ý
            with gr.Tab("Trò chuyện & góp ý"):
                gr.Markdown(
                    "Đọc báo cáo xong, góp ý ở đây. Góp ý được ghi thẳng vào bộ nhớ "
                    "nghiên cứu, rồi bấm **Chạy lại với góp ý** để các agent chạy thêm "
                    "vòng mới trên đúng những gì bạn vừa nói và viết lại báo cáo."
                )
                chat_banner = gr.Markdown(_session_banner(None))
                with gr.Row():
                    with gr.Column(scale=3):
                        # Gradio 6: Chatbot chỉ nhận định dạng messages
                        # ([{"role", "content"}]), không còn tham số type.
                        chatbox = gr.Chatbot(label="Hội thoại", height=420)
                        with gr.Row():
                            chat_in = gr.Textbox(
                                label=None, show_label=False, scale=5, lines=2, max_lines=5,
                                placeholder="Ví dụ: cơ chế của giả thuyết này chưa thuyết phục, "
                                            "thiếu nhóm đối chứng…")
                            send_btn = gr.Button("Gửi", variant="primary", scale=1)
                    with gr.Column(scale=2):
                        target = gr.Dropdown(
                            label="Gắn góp ý vào", choices=[TARGET_AUTO, TARGET_GENERAL],
                            value=TARGET_AUTO, interactive=True,
                            info="Chọn đúng giả thuyết thì góp ý vào thẳng nó và không "
                                 "tốn lượt gọi LLM để phân loại.")
                        rerun_n = gr.Number(label="Số vòng chạy thêm", value=1,
                                            precision=0, minimum=1, maximum=5)
                        rerun_btn = gr.Button("Chạy lại với góp ý", variant="primary")
                        gr.Markdown(
                            "Chạy lại **giữ nguyên** giả thuyết, phản biện và góp ý đã có — "
                            "chỉ chạy thêm vòng mới rồi viết báo cáo mới.")
                        chat_status = gr.Markdown("")
                chat_log = gr.Textbox(label="Nhật ký khi chạy lại", lines=8, max_lines=8,
                                      autoscroll=True, interactive=False)

            # ------------------------------------------------ tab nạp state
            with gr.Tab("Nạp kết quả cũ"):
                gr.Markdown("Nạp lại `state.json` của một lần chạy trước để xem kết quả mà không cần chạy lại.")
                with gr.Row():
                    state_file = gr.File(label="Chọn state.json", file_types=[".json"])
                    state_path = gr.Textbox(label="hoặc nhập đường dẫn", value="output/state.json")
                load_btn = gr.Button("Nạp kết quả", variant="primary")
                load_status = gr.Markdown()

        run_outputs = [status, log, stats, table, memory_ids, details_state, report, files,
                       notes, detail, session_state, target]
        run_event = run_btn.click(
            fn=run_pipeline,
            inputs=[goal, constraints, mode, model, emb_model, iterations, n_hyp, n_matches,
                    top_k, dup_threshold, out_dir],
            outputs=run_outputs,
            concurrency_limit=1,   # log gom qua root logger -> 2 lượt song song sẽ lẫn nhau
        )
        stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[run_event])
        # Chạy xong thì cập nhật luôn dòng tóm tắt phiên ở tab trò chuyện.
        run_event.then(fn=_session_banner, inputs=[session_state], outputs=[chat_banner])

        load_btn.click(fn=load_state, inputs=[state_file, state_path, mode, model, emb_model],
                       outputs=[load_status, stats, table, memory_ids, details_state, report,
                                files, notes, detail, session_state, target, chat_banner])

        # ---------------- trò chuyện & góp ý ----------------
        send_event = send_btn.click(
            fn=on_send, inputs=[chat_in, chatbox, session_state, target],
            outputs=[chatbox, chat_in, session_state, target], concurrency_limit=1)
        chat_in.submit(
            fn=on_send, inputs=[chat_in, chatbox, session_state, target],
            outputs=[chatbox, chat_in, session_state, target], concurrency_limit=1)

        rerun_event = rerun_btn.click(
            fn=on_rerun, inputs=[rerun_n, session_state, chatbox],
            outputs=[chat_status, chat_log, stats, table, memory_ids, details_state,
                     report, files, notes, chatbox, target],
            concurrency_limit=1)
        rerun_event.then(fn=_session_banner, inputs=[session_state], outputs=[chat_banner])

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
