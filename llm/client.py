from __future__ import annotations

import asyncio
import json
import re
from typing import Optional

from anthropic import AsyncAnthropic, APIError, APIConnectionError, RateLimitError, AuthenticationError, BadRequestError  # Gọi GLM-5.2 qua interface Anthropic /v1/messages

from config import LLMConfig


def _extract_balanced_json(raw: str) -> str:
    """Bóc object/array JSON đầu tiên khớp ngoặc từ raw text.

    Xử lý 3 trường hợp LLM hay gặp:
      - JSON lồng trong code fence (```json ... ```) -> bóc fence trước.
      - Text thừa trước/sau JSON (vd: "Dưới đây là kết quả: {...} Cảm ơn.")
      - JSON bị cắt giữa (không khớp ngoặc) -> trả nguyên để caller raise lỗi rõ.

    Thử `json.loads` nguyên chuỗi trước (nhanh, đúng trường hợp JSON thuần);
    nếu fail, quét từ `{` / `[` đầu tiên đến ngoặc khớp cuối cùng và `json.loads`
    lại. Trả về chuỗi raw (có thể không hợp lệ) nếu không bóc được đều —
    `complete_json` sẽ raise ValueError kèm raw để debug.
    """
    raw = re.sub(r"^```(?:json)?|^```|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        json.loads(raw)
        return raw
    except Exception:
        pass

    for tok_open, tok_close in ("{}", "[]"):
        start = raw.find(tok_open[0])
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(raw)):
            c = raw[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == tok_open:
                depth += 1
            elif c == tok_close:
                depth -= 1
                if depth == 0:
                    candidate = raw[start : i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        break  # ngoặc khớp nhưng không parse được -> không khớp nữa
    return raw


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config

        # Khởi tạo client Anthropic.
        # Dùng interface /v1/messages. base_url lấy từ config (.env).
        base_url = getattr(config, "api_base", None)

        # Proxy boltz.one authentication: nó kiểm tra header `Authorization: Bearer <token>`,
        # KHÔNG phải `x-api-key` mặc định của SDK anthropic. Nên ta phải đính kèm
        # Authorization vào default_headers. (CoT/demo.py gửi cả 2 header nên chạy được;
        # SDK chỉ gửi x-api-key nên bị 401 "invalid or expired key" dù token vẫn đúng.)
        self._client = AsyncAnthropic(
            api_key=config.api_key,
            base_url=base_url,  # Endpoint của Anthropic thật hoặc proxy (boltz, v.v.) nếu có
            default_headers={
                "Authorization": f"Bearer {config.api_key}",
                # Phòng Cloudflare error 1010: nhiều proxy nằm sau Cloudflare,
                # để UA mặc định sẽ bị chặn như bot.
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36",
                # Ép server nén bằng gzip thay vì brotli: nếu môi trường thiếu lib
                # `brotli`/`brotlicffi` (pip sạch thường không có) thì httpx không
                # giải nén br được → raise APIConnectionError dù HTTP trả 200.
                # "gzip, identity" tắt brotli → tránh "Connection error" âm thầm.
                "Accept-Encoding": "gzip, identity",
            },
        )
        self._semaphore = asyncio.Semaphore(config.max_concurrency)

    async def complete(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Gọi model qua interface Anthropic /v1/messages bao gồm cả system prompt và user"""
        last_err = None
        async with self._semaphore:
            for attempt in range(self.config.max_retries):
                try:
                    # Interface Anthropic tách system ra khỏi messages.
                    # CHỈ giữ các param mà messages.create thực sự chấp nhận, phần còn lại
                    # (temperature, top_p, top_k, reasoning_effort, ...) gửi qua extra_body.
                    # Lý do: SDK anthropic 1.5.0 (và proxy GLM/boltz) KHÔNG có tham số
                    # `temperature` trong signature AsyncMessages.create -> truyền trực tiếp
                    # sẽ raise TypeError: unexpected keyword argument 'temperature',
                    # bị bắt bởi `except Exception` và retry 3 lần vô nghĩa tới
                    # "Model call thất bại sau 3 lần". Đây chính là lỗi query expansion đã gặp.
                    extra_body: dict = {}
                    # Tham số tuỳ chọn sampling gửi qua extra_body (an toàn với mọi phiên bản SDK
                    # và mọi proxy; nếu server bỏ qua thì cũng không lỗi).
                    sampling = {
                        "temperature": temperature if temperature is not None else self.config.temperature,
                    }
                    for k_out, v_in in (("top_p", getattr(self.config, "top_p", None)),
                                       ("top_k", getattr(self.config, "top_k", None))):
                        if v_in is not None:
                            sampling[k_out] = v_in
                    extra_body.update(sampling)

                    # reasoning_effort (GLM/reasoning model) cũng đi qua extra_body.
                    reasoning_effort = getattr(self.config, "reasoning_effort", None)
                    if reasoning_effort:
                        extra_body["reasoning_effort"] = reasoning_effort

                    kwargs = {
                        "model": self.config.model,
                        "system": system,
                        "messages": [
                            {"role": "user", "content": user}
                        ],
                        "max_tokens": max_tokens or self.config.max_tokens,
                    }
                    if extra_body:
                        kwargs["extra_body"] = extra_body

                    resp = await self._client.messages.create(**kwargs)
                    # Anthropic trả về content là danh sách các block; gộp text lại.
                    return "".join(
                        block.text for block in resp.content if getattr(block, "type", None) == "text"
                    ) or ""

                except AuthenticationError as e:
                    # 401/403: sai token hoặc base_url không khớp interface.
                    raise RuntimeError(
                        f"Auth lỗi (kiểm tra ANTHROPIC_AUTH_TOKEN và base_url): {e}"
                    )
                except BadRequestError as e:
                    # 400/404: sai path, sai model, body sai cú pháp -> thử lại cũng vậy.
                    raise RuntimeError(
                        f"Lỗi request 400/404 (kiểm tra base_url={base_url} "
                        f"và model {self.config.model}): {e}"
                    )
                except TypeError as e:
                    # Lỗi do thừa/thiếu tham số trong khi build request (vd: SDK/proxy
                    # không chấp nhận tham số nào đó) -> KHÔNG retry, raise ngay để
                    # thông báo lỗi chính xác, tránh "thất bại sau 3 lần" gây hiểu nhầm.
                    raise RuntimeError(f"Lỗi tham số request (không retry được): {e}")
                except RateLimitError as e:
                    last_err = e
                    await asyncio.sleep(min(2 ** attempt, 20))
                except APIConnectionError as e:
                    last_err = e
                    await asyncio.sleep(1.5 * (attempt + 1))
                except APIError as e:
                    # Các lỗi HTTP khác (5xx, v.v.).
                    last_err = e
                    await asyncio.sleep(1.5 * (attempt + 1))
                except Exception as e:
                    last_err = e
                    await asyncio.sleep(1.5 * (attempt + 1))
            raise RuntimeError(f"Model call thất bại sau {self.config.max_retries} lần: {last_err}")

    async def complete_json(self, system: str, user: str, retries_on_parse_fail: int = 2, **kwargs) -> dict | list:
        """Yêu cầu model trả về JSON thuần và parse. Với đầu vào là System prompt và User prompt.

        Hardening so với bản cũ:
          - Dùng `_extract_balanced_json` để bóc JSON ra khỏi text thừa / code fence,
            thay vì chỉ `re.sub` code fence.
          - Khi parse fail, gửi lại cho model kèm lỗi cụ thể và yêu cầu sửa (≤
            `retries_on_parse_fail` lần) — LLM thường tự chữa được JSON sai của mình.
          - Vẫn giũ signature `(system, user, **kwargs)` để không vỡ FakeLLM /
            chatbot / các agent gọi hiện tại.
        """
        json_system = (
            system
            + "\n\nQUAN TRỌNG: Chỉ trả về JSON hợp lệ, bắt đầu bằng { hoặc [, "
            "kết thúc bằng } hoặc ], không thêm lời dẫn, không dùng markdown code fence."
        )
        cur_user = user
        last_err: Exception | None = None
        raw = ""
        for attempt in range(retries_on_parse_fail + 1):
            raw = await self.complete(json_system, cur_user, **kwargs)
            candidate = _extract_balanced_json(raw)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as e:
                last_err = e
                if attempt < retries_on_parse_fail:
                    # Self-correction: báo lỗi cho model và yêu cầu trả lại JSON đúng.
                    cur_user = (
                        f"{user}\n\n"
                        f"Lần trả trước KHÔNG phải JSON hợp lệ (lỗi: {e.msg}). "
                        f"Hãy trả lại CHÍNH XÁC một JSON hợp lệ, bắt đầu bằng {{ hoặc [ "
                        f"và kết thúc bằng }} hoặc ], không thêm gì khác."
                    )
                    await asyncio.sleep(0.5)
        raise ValueError(
            f"Không parse được JSON sau {retries_on_parse_fail + 1} lần: {last_err}\nRaw: {raw[:500]}"
        )

    async def complete_json_tool(self, system: str, user: str, tool: dict, **kwargs) -> dict:
        """Gọi model với 1 tool + `tool_choice` ép cụ thể (Cách 2 — structured output).

        Schema thật do `tool["input_schema"]` định nghĩa và Anthropic ép presence/type
        ngay tại inference -> không cần parse text, không cần kẹt `data["key"]` trần.
        Trả về `block.input` (dict đã validate theo schema).

        Fallback: nếu model không gọi tool (trả text thuần — có thể do proxy bỏ qua
        `tool_choice`), gom text và gọi `complete_json` (hardened) để parse; nếu
        fallback cũng fail -> raise ValueError kèm raw. Như vậy wrapper này tương
        thích với cả proxy hỗ trợ và không hỗ trợ tool use.
        """
        async with self._semaphore:
            # Build kwargs giống complete(): temperature/top_p/top_k/reasoning_effort
            # đi qua extra_body (an toàn với mọi phiên bản SDK / proxy).
            extra_body: dict = {}
            sampling = {
                "temperature": kwargs.pop("temperature", None) or self.config.temperature,
            }
            for k_out, v_in in (("top_p", getattr(self.config, "top_p", None)),
                               ("top_k", getattr(self.config, "top_k", None))):
                if v_in is not None:
                    sampling[k_out] = v_in
            extra_body.update(sampling)
            reasoning_effort = getattr(self.config, "reasoning_effort", None)
            if reasoning_effort:
                extra_body["reasoning_effort"] = reasoning_effort

            req_kwargs = {
                "model": self.config.model,
                "system": system,
                "messages": [{"role": "user", "content": user}],
                "max_tokens": kwargs.pop("max_tokens", None) or self.config.max_tokens,
                "tools": [tool],
                "tool_choice": {"type": "tool", "name": tool["name"]},
            }
            if extra_body:
                req_kwargs["extra_body"] = extra_body

            resp = await self._client.messages.create(**req_kwargs)

        # tool_use block đầu khớp tên tool -> trả input (dict đã validate schema).
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == tool["name"]:
                return block.input

        # Fallback: model không gọi tool (trả text) -> parse text qua complete_json.
        text = "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        ).strip()
        try:
            return await self.complete_json(system, user + f"\n\n{text}", **kwargs)
        except ValueError as e:
            raise ValueError(
                f"Model không gọi tool '{tool['name']}' và text không parse được JSON: {e}\nRaw: {text[:500]}"
            )
