import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple


TEXT_EXTS = {".txt", ".md", ".markdown"}


def _safe_read_text(path: str, max_chars: int) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(max_chars)
    except Exception:
        return ""


def _iter_text_files(root_dir: str) -> Iterable[str]:
    for base, _, files in os.walk(root_dir):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in TEXT_EXTS:
                yield os.path.join(base, fn)


def _split_into_chunks(text: str, chunk_chars: int = 800, overlap: int = 120) -> List[str]:
    if not text:
        return []
    text = re.sub(r"\r\n?", "\n", text).strip()
    if not text:
        return []
    chunks: List[str] = []
    i = 0
    n = len(text)
    step = max(1, chunk_chars - overlap)
    while i < n:
        chunk = text[i : i + chunk_chars].strip()
        if chunk:
            chunks.append(chunk)
        i += step
    return chunks


def _tokenize(s: str) -> List[str]:
    s = (s or "").lower()
    # 中英文混合：对英文按词，对中文按连续字串/字符，保证无需额外依赖也能做粗检索
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", s)
    return tokens


def _score_chunk(query_tokens: List[str], chunk_tokens: List[str]) -> float:
    if not query_tokens or not chunk_tokens:
        return 0.0
    qset = set(query_tokens)
    cset = set(chunk_tokens)
    overlap = len(qset & cset)
    if overlap <= 0:
        return 0.0
    # 简单归一化：更偏向“覆盖率”，并轻微惩罚过长 chunk
    return (overlap / max(1, len(qset))) * (1.0 / (1.0 + 0.002 * len(chunk_tokens)))


@dataclass
class DriverRagResult:
    driver_id: str
    snippets: List[Tuple[str, str]]  # (source_path, snippet)


def build_driver_profile_prompt_from_dir(
    *,
    drivers_root: str,
    driver_id: str,
    query: str,
    top_k: int = 4,
    max_chars_per_file: int = 30_000,
) -> DriverRagResult:
    driver_dir = os.path.join(drivers_root, driver_id)
    snippets: List[Tuple[str, str]] = []

    if not driver_id or not os.path.isdir(driver_dir):
        return DriverRagResult(driver_id=driver_id or "", snippets=snippets)

    q_tokens = _tokenize(query)
    scored: List[Tuple[float, str, str]] = []  # (score, source_path, chunk)

    for path in _iter_text_files(driver_dir):
        text = _safe_read_text(path, max_chars=max_chars_per_file)
        for chunk in _split_into_chunks(text):
            c_tokens = _tokenize(chunk)
            score = _score_chunk(q_tokens, c_tokens)
            if score > 0:
                scored.append((score, path, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    for score, path, chunk in scored[: max(1, top_k)]:
        _ = score
        snippets.append((path, chunk))

    return DriverRagResult(driver_id=driver_id, snippets=snippets)


def format_driver_profile_prompt(result: DriverRagResult) -> str:
    if not result.snippets:
        return ""
    parts: List[str] = []
    parts.append("以下内容来自已选择司机的档案文档摘录，请用于理解司机特征（不要逐字复述档案原文）：")
    for src, snippet in result.snippets:
        parts.append(f"\n[来源] {os.path.basename(src)}\n{snippet}".strip())
    return "\n\n".join(parts).strip()
