from __future__ import annotations

import asyncio
import json
import re
from typing import Optional

from anthropic import AsyncAnthropic, APIError, APIConnectionError, RateLimitError, AuthenticationError, BadRequestError  # Gọi GLM-5.2 qua interface Anthropic /v1/messages

from config import LLMConfig


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

    async def complete_json(self, system: str, user: str, **kwargs) -> dict | list:
        """Yêu cầu model trả về JSON thuần và parse. Với đầu vào là System prompt và User prompt"""
        json_system = (
            system
            + "\n\nQUAN TRỌNG: Chỉ trả về JSON hợp lệ, không thêm lời dẫn, "
              "không dùng markdown code fence."
        )
        # Model đôi khi trả JSON lỗi/bị cắt cụt -> gọi lại (tối đa max_retries lần)
        # thay vì làm hỏng cả pha. Lỗi auth/request vẫn raise ngay từ complete().
        attempts = max(1, self.config.max_retries)
        last_err: Optional[Exception] = None
        for _ in range(attempts):
            raw = await self.complete(json_system, user, **kwargs)
            try:
                return _parse_json(raw)
            except ValueError as e:
                last_err = e
        raise ValueError(f"Không parse được JSON sau {attempts} lần: {last_err}")


def _parse_json(raw: str) -> dict | list:
    """Parse JSON từ output của model: bỏ code fence, nếu vẫn lỗi thì cắt từ '{'/'['
    đầu tiên tới '}'/']' cuối cùng (model hay thêm lời dẫn trước/sau JSON)."""
    cleaned = re.sub(r"^```json|^```|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        first_err = e
    starts = [i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1]
    if starts:
        start = min(starts)
        end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        if end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass
    raise ValueError(f"Không parse được JSON: {first_err}\nRaw: {raw[:500]}")
