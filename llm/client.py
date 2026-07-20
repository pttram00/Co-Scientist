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
        """Gọi model qua interface Anthropic /v1/messages."""
        last_err = None
        async with self._semaphore:
            for attempt in range(self.config.max_retries):
                try:
                    # Interface Anthropic tách system ra khỏi messages.
                    kwargs = {
                        "model": self.config.model,
                        "system": system,
                        "messages": [
                            {"role": "user", "content": user}
                        ],
                        "max_tokens": max_tokens or self.config.max_tokens,
                        "temperature": temperature if temperature is not None else self.config.temperature,
                    }

                    # Một số model/proxy hỗ trợ mức độ suy luận qua tham số reasoning_effort.
                    reasoning_effort = getattr(self.config, "reasoning_effort", None)
                    if reasoning_effort:
                        kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}

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
        """Yêu cầu model trả về JSON thuần và parse."""
        json_system = (
            system
            + "\n\nQUAN TRỌNG: Chỉ trả về JSON hợp lệ, không thêm lời dẫn, "
              "không dùng markdown code fence."
        )
        raw = await self.complete(json_system, user, **kwargs)
        cleaned = re.sub(r"^```json|^```|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Không parse được JSON: {e}\nRaw: {raw[:500]}")
