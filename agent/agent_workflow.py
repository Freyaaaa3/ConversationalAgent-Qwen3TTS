import os
import json
from typing import Optional

import urllib.request
import urllib.error

from .prompts import (
    DEFAULT_SYSTEM_PROMPT,
    build_attribution_prompt,
    build_comfort_prompt,
    build_context_summary_prompt,
    build_followup_comfort_prompt,
    build_followup_no_principle_prompt,
    build_no_principle_comfort_prompt,
)


class DeepSeekAgent:
    """
    Ollama-backed 文本 Agent。

    对后端保持向后兼容：
    - 属性: ollama_model (str)
    - 方法: generate_response(subject_id: str, text: str) -> str

    同时新增一套“情绪归因 + 安慰话术”工作流：
    - analyze_driver_state(...)
    - generate_comforting_message(...)
    """

    def __init__(self, ollama_model: Optional[str] = None, ollama_host: Optional[str] = None) -> None:
        self.ollama_model = (ollama_model or os.environ.get("OLLAMA_MODEL") or "qwen3:8b").strip()
        self.ollama_host = (ollama_host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")

    # === 底层 LLM 调用封装 ===

    def _call_llm(self, prompt: str, timeout: int = 120) -> str:
        prompt = (prompt or "").strip()
        if not prompt:
            return ""

        url = f"{self.ollama_host}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
        }
        data_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                parsed = json.loads(body) if body else {}
                return (parsed.get("response") or "").strip()
        except urllib.error.HTTPError as e:
            # HTTPError 也包含 response body，有助于排查
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            raise RuntimeError(f"Ollama HTTP error: {e.code} {detail}".strip()) from e

    def _print_prompt(self, *, title: str, prompt: str, max_chars: int = 8000) -> None:
        """
        仅用于调试：把即将发送给大模型的 prompt 打到终端。
        """
        try:
            text = (prompt or "").strip()
            if not text:
                print(f">> [LLM Prompt] {title}: <empty>")
                return
            clipped = text if len(text) <= max_chars else (text[:max_chars] + f"\n... (truncated, total_chars={len(text)})")
            print("\n" + "=" * 88)
            print(f">> [LLM Prompt] {title}")
            print(f">> [LLM Prompt] model={self.ollama_model} host={self.ollama_host} chars={len(text)}")
            print("-" * 88)
            print(clipped)
            print("=" * 88 + "\n")
        except Exception:
            # debug printing must never break main flow
            pass

    # === 兼容旧接口 ===

    def generate_response(self, subject_id: str, text: str) -> str:
        """
        向后兼容的简单接口：直接把 text 当成完整 prompt。
        subject_id 暂预留用于未来做 persona/话题路由。
        """
        _ = subject_id  # 当前未使用
        return self._call_llm(text)

    # === 情绪归因 + 安慰话术工作流 ===

    def analyze_driver_state(
        self,
        scene_prompt: str,
        driver_profile_prompt: str,
        user_prompt: str,
        extra_instruction: Optional[str] = None,
    ) -> str:
        """
        第一步：情绪归因分析。

        - 将【场景提示词】和【司机特征提示词】嵌入到归因分析提示词中，
        - 再把【用户提示词】作为当前事件，交给大模型进行归因分析。

        返回值：模型的自然语言分析结果（可直接展示或作为后续安慰话术的输入）。
        """
        prompt = build_attribution_prompt(
            scene_prompt=scene_prompt,
            driver_profile_prompt=driver_profile_prompt,
            user_prompt=user_prompt,
            extra_instruction=extra_instruction,
        )
        self._print_prompt(title="Attribution analysis", prompt=prompt)
        return self._call_llm(prompt)

    def generate_comforting_message(
        self,
        attribution_result: str,
        user_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        第二步：基于归因结果 + 系统提示词生成安慰话术。

        - 输入：
          - attribution_result: 上一步归因分析的文字结果
          - user_prompt: 当前司机的话 / 触发 Wake Lumina 的那句话
          - system_prompt: 系统级干预提示词（不传则使用 DEFAULT_SYSTEM_PROMPT）
        - 输出：一段直接可读出的安慰话语，满足安全、共情、简洁等约束。
        """
        sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        prompt = build_comfort_prompt(
            system_prompt=sys_prompt,
            attribution_result=attribution_result,
            user_prompt=user_prompt,
        )
        self._print_prompt(title="Comfort message", prompt=prompt)
        return self._call_llm(prompt)

    def generate_comfort_no_principle(self, scene_prompt: str, user_prompt: str) -> str:
        """
        无原则模式：单次调用，仅根据情景 + 司机话语生成安慰话术（不走归因与系统干预原则）。
        """
        prompt = build_no_principle_comfort_prompt(scene_prompt=scene_prompt, user_prompt=user_prompt)
        self._print_prompt(title="Comfort message (no-principle mode)", prompt=prompt)
        return self._call_llm(prompt)

    def generate_followup_no_principle(
        self,
        *,
        scene_prompt: str,
        last_assistant_comfort: str,
        user_reply: str,
    ) -> str:
        """无原则模式后续轮次。"""
        prompt = build_followup_no_principle_prompt(
            scene_prompt=scene_prompt,
            last_assistant_comfort=last_assistant_comfort,
            user_reply=user_reply,
        )
        self._print_prompt(title="Followup comfort (no-principle mode)", prompt=prompt)
        return self._call_llm(prompt)

    def summarize_intervention_context(
        self,
        *,
        principle_prompt: str,
        first_comfort_text: str,
        role_doc_prompt: str,
    ) -> str:
        """
        第一次安慰语生成后：对“原则提示词(system) + 第一次安慰语 + 角色文档(driver_profile)”做上下文总结。
        """
        prompt = build_context_summary_prompt(
            principle_prompt=principle_prompt,
            first_comfort_text=first_comfort_text,
            role_doc_prompt=role_doc_prompt,
        )
        self._print_prompt(title="Context summary", prompt=prompt)
        return self._call_llm(prompt)

    def generate_followup_comfort_message(
        self,
        *,
        context_summary: str,
        user_reply: str,
    ) -> str:
        """
        后续对话：只使用上下文总结 + 用户的新回复，生成后续安慰话术。
        """
        prompt = build_followup_comfort_prompt(context_summary=context_summary, user_reply=user_reply)
        self._print_prompt(title="Followup comfort message", prompt=prompt)
        return self._call_llm(prompt)

