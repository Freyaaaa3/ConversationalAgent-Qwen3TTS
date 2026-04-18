import os
import json
import uuid
from typing import Dict, Any, List, Optional, Tuple
import sys
import shutil

import torch
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

import io
import base64
import time
import threading

# Ensure local repo modules are importable regardless of CWD.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "backend", "pkg"))


def _ensure_sox_in_path() -> None:
    """
    如果系统 PATH 找不到 sox.exe，音频预处理可能会失败/变慢。
    尝试自动把本项目 .env/Library/bin 加入 PATH。
    """
    try:
        if shutil.which("sox"):
            return
        candidates = [
            os.path.join(REPO_ROOT, ".env", "Library", "bin"),
            os.path.join(REPO_ROOT, ".venv", "Library", "bin"),
        ]
        for c in candidates:
            if os.path.exists(os.path.join(c, "sox.exe")):
                os.environ["PATH"] = c + os.pathsep + os.environ.get("PATH", "")
                print(f">> [TTS] Auto-added to PATH: {c}")
                return
        print(">> [TTS] Warning: sox not found in PATH; audio clone may fail or be slow.")
    except Exception:
        pass


_ensure_sox_in_path()

from agent.agent_workflow import DeepSeekAgent
from agent.prompts import (
    DEFAULT_SCENE_PROMPT,
    DEFAULT_DRIVER_PROFILE_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_WAKE_USER_PROMPT,
)
from backend.app.rag_driver_profile import (
    build_driver_profile_prompt_from_dir,
    format_driver_profile_prompt,
)


BASE_DIR = REPO_ROOT
PERSONA_ROOT = os.path.join(REPO_ROOT, "voices", "personas")
os.makedirs(PERSONA_ROOT, exist_ok=True)

DRIVERS_ROOT = os.path.join(REPO_ROOT, "drivers")
os.makedirs(DRIVERS_ROOT, exist_ok=True)

PREVIEW_AUDIO_ROOT = os.path.join(REPO_ROOT, "voices", "audio_previews")
os.makedirs(PREVIEW_AUDIO_ROOT, exist_ok=True)
PREVIEW_MANIFEST_PATH = os.path.join(PREVIEW_AUDIO_ROOT, "manifest.json")
# 所有 Music/preview 声线在 TTS 克隆时统一使用的参考文本（/tts/clone_base64 + preview_item_id）
PREVIEW_CLONE_REF_TEXT = "你好呀，我是你的车载智能语音助手"

MODEL_PATH = os.path.join(REPO_ROOT, "backend", "model", "Qwen3_TTS_12Hz_0.6B_Base")

SETTINGS_PATH = os.path.join(REPO_ROOT, "backend", "config", "runtime_settings.json")

ALLOWED_OLLAMA_MODELS = ["qwen3:8b", "deepseek-r1:8b", "qwen3:4b"]

def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


# 默认值：来自环境变量（仍然支持用 env 一键配置），但可被 /settings 动态更新覆盖
_tts_runtime_settings: Dict[str, Any] = {
    "use_flash_attention": _env_flag("QWEN_TTS_FLASH_ATTN", "0"),
    "use_4bit": _env_flag("QWEN_TTS_4BIT", "0"),
    "use_torch_compile": _env_flag("QWEN_TTS_TORCH_COMPILE", "0"),
}

_llm_runtime_settings: Dict[str, Any] = {
    "ollama_model": os.environ.get("OLLAMA_MODEL", "qwen3:8b"),
}


def _load_settings_from_disk() -> None:
    global _tts_runtime_settings, _llm_runtime_settings
    try:
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            for k in ("use_flash_attention", "use_4bit", "use_torch_compile"):
                if k in data:
                    _tts_runtime_settings[k] = bool(data[k])
            if "ollama_model" in data and isinstance(data["ollama_model"], str):
                _llm_runtime_settings["ollama_model"] = data["ollama_model"].strip()
    except Exception as e:
        print(f">> [Settings] 读取 {SETTINGS_PATH} 失败（忽略）: {e}")


def _save_settings_to_disk() -> None:
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    **_tts_runtime_settings,
                    **_llm_runtime_settings,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception as e:
        print(f">> [Settings] 写入 {SETTINGS_PATH} 失败（忽略）: {e}")


def _ensure_preview_manifest() -> None:
    if os.path.exists(PREVIEW_MANIFEST_PATH):
        return
    try:
        with open(PREVIEW_MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": 1,
                    "items": [],
                    "note": "Put audio files under audio_previews/ and add entries here: {id,name,description,filename}.",
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception as e:
        print(f">> [PreviewAudio] 初始化 manifest 失败（忽略）: {e}")


def _load_preview_manifest() -> Dict[str, Any]:
    _ensure_preview_manifest()
    with open(PREVIEW_MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f) or {"version": 1, "items": []}


def _guess_media_type(filename: str) -> str:
    fn = (filename or "").lower()
    if fn.endswith(".wav"):
        return "audio/wav"
    if fn.endswith(".mp3"):
        return "audio/mpeg"
    if fn.endswith(".ogg"):
        return "audio/ogg"
    if fn.endswith(".flac"):
        return "audio/flac"
    if fn.endswith(".m4a"):
        return "audio/mp4"
    return "application/octet-stream"

def _effective_attn_impl(use_flash_attention: bool) -> str:
    if not torch.cuda.is_available():
        return "cpu_default"
    if not use_flash_attention:
        return "sdpa"
    # FlashAttention-2：Ampere 8.0+ 用 CUDA 版；Turing 7.5（如 2080 Ti）需安装 flash-attn-triton + triton-windows
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return "sdpa"
    cap = torch.cuda.get_device_capability(0)
    if cap[0] >= 8:
        return "flash_attention_2_cuda"
    if cap[0] == 7 and cap[1] >= 5:
        return "flash_attention_2_triton"
    return "sdpa"


def _load_tts_model(*, use_flash_attention: bool, use_4bit: bool, use_torch_compile: bool):
    cuda_available = torch.cuda.is_available()
    device_map = "cuda:0" if cuda_available else "cpu"
    load_kwargs: Dict[str, Any] = {
        "device_map": device_map,
        # transformers 更推荐 torch_dtype；同时保留 dtype 兼容 qwen_tts wrapper 的参数名
        "torch_dtype": torch.bfloat16 if cuda_available else torch.float32,
        "dtype": torch.bfloat16 if cuda_available else torch.float32,
    }

    if cuda_available:
        attn_impl = _effective_attn_impl(use_flash_attention)
        if attn_impl.startswith("flash_attention_2"):
            load_kwargs["attn_implementation"] = "flash_attention_2"
            print(">> [TTS] FlashAttention: ON")
        else:
            load_kwargs["attn_implementation"] = "sdpa"
            print(">> [TTS] FlashAttention: OFF (SDPA)")
        try:
            cap = torch.cuda.get_device_capability(0)
            print(f">> [TTS] GPU compute capability: {cap[0]}.{cap[1]}")
        except Exception:
            pass

    # 4-bit 量化：省显存、加速；单卡用 cuda:0 可避免部分 pickle 错误
    if use_4bit and cuda_available:
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = "cuda:0"  # 单卡用 cuda:0 减少 pickle 问题
            # FlashAttention2 需要明确 torch_dtype；4-bit 下同样保留 torch_dtype
            load_kwargs["torch_dtype"] = torch.bfloat16
            # dtype 不是 transformers 官方参数名，避免触发“未指定 torch dtype”的告警
            load_kwargs.pop("dtype", None)
            print(">> [TTS] 使用 4-bit 量化加载。")
        except Exception as e:
            print(f">> [TTS] 4-bit 加载失败，回退全精度: {e}")
            load_kwargs.pop("quantization_config", None)
            load_kwargs["device_map"] = device_map

    print("Loading Qwen3-TTS model...")
    try:
        from qwen_tts import Qwen3TTSModel  # local import: avoid heavy deps at module import time

        model = Qwen3TTSModel.from_pretrained(MODEL_PATH, **load_kwargs)
    except Exception as e:
        if load_kwargs.pop("quantization_config", None) is not None:
            print(f">> [TTS] 4-bit 加载报错，改用全精度重试: {e}")
            load_kwargs["device_map"] = "cuda:0" if cuda_available else "cpu"
            load_kwargs["torch_dtype"] = torch.bfloat16 if cuda_available else torch.float32
            load_kwargs["dtype"] = torch.bfloat16 if cuda_available else torch.float32
            from qwen_tts import Qwen3TTSModel  # local import

            model = Qwen3TTSModel.from_pretrained(MODEL_PATH, **load_kwargs)
        else:
            raise

    if use_torch_compile and cuda_available and hasattr(torch, "compile"):
        try:
            # Windows 下 inductor/缓存偶尔会抛异常；允许自动回退 eager，避免启动失败
            try:
                import torch._dynamo as _dynamo  # 用 as 避免把 torch 变成局部变量导致 UnboundLocalError

                _dynamo.config.suppress_errors = True
            except Exception:
                pass

            # 避免多个进程/会话共享 temp cache 导致 WinError 183
            try:
                cache_root = os.path.join(BASE_DIR, ".torchinductor_cache", str(os.getpid()))
                os.makedirs(cache_root, exist_ok=True)
                os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", cache_root)
            except Exception:
                pass

            # reduce-overhead 侧重降低单次推理延迟
            model.model = torch.compile(model.model, mode="reduce-overhead")
            print(">> [TTS] 已启用 torch.compile(mode='reduce-overhead')，首次推理会编译，后续加速。")
        except Exception as e:
            print(f">> [TTS] torch.compile 启用失败，使用未编译模型: {e}")

    print("Qwen3-TTS model loaded.")
    return model


_load_settings_from_disk()
tts_model_lock = threading.RLock()
tts_model = None


def _ensure_tts_model():
    global tts_model
    with tts_model_lock:
        if tts_model is None:
            tts_model = _load_tts_model(
                use_flash_attention=_tts_runtime_settings["use_flash_attention"],
                use_4bit=_tts_runtime_settings["use_4bit"],
                use_torch_compile=_tts_runtime_settings["use_torch_compile"],
            )
        return tts_model


def _reload_tts_model() -> None:
    global tts_model
    with tts_model_lock:
        print(">> [TTS] ====== Re-loading model due to settings change ======")
        tts_model = _load_tts_model(
            use_flash_attention=_tts_runtime_settings["use_flash_attention"],
            use_4bit=_tts_runtime_settings["use_4bit"],
            use_torch_compile=_tts_runtime_settings["use_torch_compile"],
        )
        print(">> [TTS] ====== Model reload done ======")

text_agent_lock = threading.RLock()
text_agent = DeepSeekAgent()

# Wake 工作流上下文总结存储（用于“第一次安慰后总结，之后再使用”）
_context_summary_store: Dict[str, str] = {}
_context_summary_events: Dict[str, threading.Event] = {}
_context_summary_store_lock = threading.RLock()
_context_summary_inputs: Dict[str, Dict[str, str]] = {}
_context_summary_started: Dict[str, bool] = {}
try:
    # 启动时应用持久化的模型选择
    if _llm_runtime_settings.get("ollama_model"):
        text_agent.ollama_model = _llm_runtime_settings["ollama_model"]
except Exception:
    pass

app = FastAPI(title="Lumina Drive Voice Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Backend is running. Open /docs to test APIs.",
        "endpoints": [
            "GET /docs",
            "GET /openapi.json",
            "GET /personas",
            "POST /personas",
            "PATCH /personas/{persona_id}",
            "POST /driver/attribution",
            "POST /driver/wake",
            "POST /chat",
            "POST /tts/clone",
            "POST /chat_and_tts/clone",
            "GET /personas/{persona_id}/audio",
            "GET /audio_previews",
            "GET /audio_previews/{item_id}",
        ],
    }


@app.get("/drivers")
def list_drivers():
    """
    列出可选司机档案目录（drivers/<driver_id>/）。
    """
    try:
        if not os.path.isdir(DRIVERS_ROOT):
            return {"drivers": []}
        driver_ids: List[str] = []
        for name in os.listdir(DRIVERS_ROOT):
            path = os.path.join(DRIVERS_ROOT, name)
            if os.path.isdir(path) and not name.startswith("."):
                driver_ids.append(name)
        driver_ids.sort()
        return {"drivers": driver_ids, "drivers_root": DRIVERS_ROOT}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Failed to list drivers: {e}"})


def _read_driver_profile_full_text(driver_id: str) -> str:
    if not driver_id:
        return ""
    profile_path = os.path.join(DRIVERS_ROOT, driver_id, "profile.md")
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def _resolve_driver_profile_prompt(
    *,
    explicit_profile_prompt: str,
    driver_id: str,
    scene_prompt: str,
    user_prompt: str,
) -> Dict[str, Any]:
    """
    解析最终用于归因分析的司机特征提示词，确保与前端所选 driver_id 对齐。
    返回:
      - profile_prompt: 最终提示词
      - rag: DriverRagResult | None
      - source: explicit | rag | full_profile | default
    """
    explicit = (explicit_profile_prompt or "").strip()
    if explicit:
        return {"profile_prompt": explicit, "rag": None, "source": "explicit"}

    did = (driver_id or "").strip()
    if did:
        rag = build_driver_profile_prompt_from_dir(
            drivers_root=DRIVERS_ROOT,
            driver_id=did,
            query=f"{scene_prompt}\n\n{user_prompt}",
            top_k=4,
        )
        rag_profile = format_driver_profile_prompt(rag).strip()
        if rag_profile:
            return {"profile_prompt": rag_profile, "rag": rag, "source": "rag"}

        full_profile = _read_driver_profile_full_text(did)
        if full_profile:
            return {"profile_prompt": full_profile, "rag": rag, "source": "full_profile"}

    return {"profile_prompt": DEFAULT_DRIVER_PROFILE_PROMPT, "rag": None, "source": "default"}


class TTSRuntimeSettings(BaseModel):
    use_flash_attention: bool
    use_4bit: bool
    use_torch_compile: bool


class TTSRuntimeSettingsUpdate(BaseModel):
    use_flash_attention: Optional[bool] = None
    use_4bit: Optional[bool] = None
    use_torch_compile: Optional[bool] = None
    reload_model: bool = True


class DriverContext(BaseModel):
    """
    描述一次驾驶场景 + 司机特征 + 当前话语，用于情绪归因 / Wake 流程。
    """

    scene_prompt: str
    driver_profile_prompt: str
    user_prompt: str
    driver_id: Optional[str] = None


class WakeRequest(DriverContext):
    """
    Wake Lumina 请求：
    - 可以选择覆盖系统提示词（否则使用默认系统提示）。
    - use_principles=False 时跳过归因与系统干预原则，仅按情景+无原则提示词单次生成安慰话；
      不返回可用的 context_id，不做上下文总结，也不进入 comfort_continue 工作流。
    """

    system_prompt_override: Optional[str] = None
    use_principles: bool = True


class WakeResponse(BaseModel):
    attribution_text: str
    comfort_text: str
    # 有原则：第一次安慰结束后上下文总结在后台异步计算；前端用 context_id 走 comfort_continue。
    # 无原则：单次话术，context_id 为空，不做总结、不走该后续链。
    context_id: str


class ComfortContinueRequest(BaseModel):
    """
    后续安慰话术请求：
    - context_id：Wake 后返回的上下文总结 id（服务端异步生成并存储）
    - user_reply：用户在被安慰之后的下一句话
    - persona_id / preview_item_id：用于 TTS 克隆（可选；均不传则只返回文本）
    - preview_item_id 与 /tts/clone_base64 一致，优先于 persona_id，ref_text 使用 PREVIEW_CLONE_REF_TEXT
    """

    context_id: str
    user_reply: str
    persona_id: Optional[str] = None
    preview_item_id: Optional[str] = None
    autoplay: bool = False


class ComfortContinueResponse(BaseModel):
    assistant_text: str
    audio_wav_base64: Optional[str] = None


class ContextSummaryStartRequest(BaseModel):
    context_id: str


class ContextSummaryStartResponse(BaseModel):
    status: str


@app.get("/settings/tts")
def get_tts_settings():
    cap = None
    if torch.cuda.is_available():
        try:
            cap = list(torch.cuda.get_device_capability(0))
        except Exception:
            cap = None
    return {
        "settings": _tts_runtime_settings,
        "gpu_compute_capability": cap,
        "effective_attn": _effective_attn_impl(_tts_runtime_settings["use_flash_attention"]),
        "settings_path": SETTINGS_PATH,
    }


@app.post("/settings/tts")
def update_tts_settings(body: TTSRuntimeSettingsUpdate):
    changed = False
    for k in ("use_flash_attention", "use_4bit", "use_torch_compile"):
        v = getattr(body, k)
        if v is not None and _tts_runtime_settings.get(k) != bool(v):
            _tts_runtime_settings[k] = bool(v)
            changed = True

    _save_settings_to_disk()

    if changed and body.reload_model:
        try:
            _reload_tts_model()
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "detail": f"Settings updated but model reload failed: {e}",
                    "settings": _tts_runtime_settings,
                    "effective_attn": _effective_attn_impl(_tts_runtime_settings["use_flash_attention"]),
                },
            )

    return {
        "settings": _tts_runtime_settings,
        "effective_attn": _effective_attn_impl(_tts_runtime_settings["use_flash_attention"]),
        "reloaded": bool(changed and body.reload_model),
    }


@app.post("/driver/attribution")
def driver_attribution(ctx: DriverContext):
    """
    情绪归因分析接口：
    - 输入：场景提示词 + 司机特征提示词 + 用户当前话语
    - 输出：一段自然语言的情绪归因分析结果
    """
    try:
        scene = ctx.scene_prompt or DEFAULT_SCENE_PROMPT
        user_text = ctx.user_prompt or DEFAULT_WAKE_USER_PROMPT
        driver_id = (ctx.driver_id or "").strip()
        profile_info = _resolve_driver_profile_prompt(
            explicit_profile_prompt=ctx.driver_profile_prompt,
            driver_id=driver_id,
            scene_prompt=scene,
            user_prompt=user_text,
        )
        profile = profile_info["profile_prompt"]
        with text_agent_lock:
            analysis = text_agent.analyze_driver_state(
                scene_prompt=scene,
                driver_profile_prompt=profile,
                user_prompt=user_text,
            )
        return {"attribution_text": analysis}
    except Exception as e:
        print(f">> [Attribution] 分析失败: {e}")
        return JSONResponse(status_code=500, content={"detail": f"Attribution failed: {e}"})


@app.post("/driver/wake", response_model=WakeResponse)
def driver_wake(req: WakeRequest):
    """
    Wake Lumina 工作流接口：

    1. 先调用大模型做情绪归因分析（场景 + 司机特征 + 当前话语）。
    2. 再将归因结果与系统提示词（干预规则）组合，生成一段安慰话术。

    返回：
    - attribution_text: 归因分析结果（可前端展示或保存在上下文中）
    - comfort_text: 一段适合由 TTS 播报的安慰话
    - context_id: 有原则模式用于后续 comfort_continue 与异步上下文总结；无原则模式为空字符串
    """
    try:
        scene = req.scene_prompt or DEFAULT_SCENE_PROMPT
        user_text = req.user_prompt or DEFAULT_WAKE_USER_PROMPT

        if not req.use_principles:
            try:
                print(">> [Wake] ====== No-principle mode (skip attribution & system principles) ======")
                print(f">> [Wake] scene_chars={len(scene)} user_chars={len(user_text)}")
                print(">> [Wake] scene_prompt:\n" + scene)
                print(">> [Wake] user_prompt:\n" + user_text)
            except Exception:
                pass
            with text_agent_lock:
                comfort = text_agent.generate_comfort_no_principle(scene, user_text)
            analysis = ""
            # 无原则：单次安慰话术，不做上下文总结、不提供后续 Wake 对话用的 context_id
            try:
                print(">> [Wake] ====== LLM Output (no-principle) ======")
                print(f">> [Wake] comfort_chars={len(comfort or '')}")
                print(">> [Wake] comfort_text:\n" + (comfort or ""))
                print(">> [Wake] context_id=(none, single-shot mode)")
                print(">> [Wake] ========================")
            except Exception:
                pass
            return WakeResponse(
                attribution_text=analysis,
                comfort_text=comfort,
                context_id="",
            )

        driver_id = (req.driver_id or "").strip()
        profile_info = _resolve_driver_profile_prompt(
            explicit_profile_prompt=req.driver_profile_prompt,
            driver_id=driver_id,
            scene_prompt=scene,
            user_prompt=user_text,
        )
        profile = profile_info["profile_prompt"]
        rag = profile_info["rag"]
        profile_source = profile_info["source"]

        # Debug prints: show what will be sent to LLM in workflow
        try:
            print(">> [Wake] ====== LLM Input Summary ======")
            print(
                f">> [Wake] driver_id={driver_id or '(none)'} source={profile_source} "
                f"scene_chars={len(scene)} profile_chars={len(profile)} user_chars={len(user_text)}"
            )
            if rag is not None:
                print(f">> [Wake] RAG snippets={len(rag.snippets)}")
                if rag.snippets:
                    for i, (src, snip) in enumerate(rag.snippets[:4]):
                        print(f">> [Wake]  - snippet[{i}] src={os.path.basename(src)} chars={len(snip)}")
            print(">> [Wake] scene_prompt:\n" + scene)
            print(">> [Wake] driver_profile_prompt:\n" + profile)
            print(">> [Wake] user_prompt:\n" + user_text)
            print(">> [Wake] ===============================")
        except Exception:
            pass

        with text_agent_lock:
            analysis = text_agent.analyze_driver_state(
                scene_prompt=scene,
                driver_profile_prompt=profile,
                user_prompt=user_text,
            )
            comfort = text_agent.generate_comforting_message(
                attribution_result=analysis,
                user_prompt=user_text,
                system_prompt=req.system_prompt_override,
            )
            # 打印首轮“第一次安慰语句”，保证你能在生成 context_summary 前就看到干预内容
            try:
                print(">> [Wake] ====== First comfort_text ======")
                print(f">> [Wake] comfort_chars={len(comfort or '')}")
                print((comfort or "").strip())
                print(">> [Wake] =====================================")
            except Exception:
                pass
            system_prompt = req.system_prompt_override or DEFAULT_SYSTEM_PROMPT
        context_id = str(uuid.uuid4())

        summary_event = threading.Event()
        with _context_summary_store_lock:
            _context_summary_events[context_id] = summary_event
            _context_summary_inputs[context_id] = {
                "principle_prompt": system_prompt,
                "first_comfort_text": comfort,
                "role_doc_prompt": profile,
            }
            _context_summary_started[context_id] = False

        try:
            print(">> [Wake] ====== LLM Output ======")
            print(f">> [Wake] attribution_chars={len(analysis or '')}")
            print(">> [Wake] attribution_text:\n" + (analysis or ""))
            print(f">> [Wake] comfort_chars={len(comfort or '')}")
            print(">> [Wake] comfort_text:\n" + (comfort or ""))
            print(f">> [Wake] context_id={context_id} (context summary async)")
            print(">> [Wake] ========================")
        except Exception:
            pass
        return WakeResponse(
            attribution_text=analysis,
            comfort_text=comfort,
            context_id=context_id,
        )
    except Exception as e:
        print(f">> [Wake] 工作流失败: {e}")
        return JSONResponse(status_code=500, content={"detail": f"Wake flow failed: {e}"})


@app.post("/driver/context_summary_start", response_model=ContextSummaryStartResponse)
def driver_context_summary_start(req: ContextSummaryStartRequest):
    """
    当客户端完成“第一次安慰语音生成”后调用此接口，后台开始做 context summary（不影响首次音频）。
    """
    context_id = (req.context_id or "").strip()
    if not context_id:
        return JSONResponse(status_code=400, content={"detail": "context_id is empty"})

    with _context_summary_store_lock:
        if _context_summary_started.get(context_id):
            return ContextSummaryStartResponse(status="already_started")
        if context_id not in _context_summary_inputs:
            return JSONResponse(status_code=404, content={"detail": "context_id not found"})

        _context_summary_started[context_id] = True
        summary_event = _context_summary_events.get(context_id)
        if summary_event is None:
            summary_event = threading.Event()
            _context_summary_events[context_id] = summary_event

        inputs = _context_summary_inputs.get(context_id) or {}

    def _bg_do_summary() -> None:
        summary_text = ""
        try:
            summary_text = text_agent.summarize_intervention_context(
                principle_prompt=inputs.get("principle_prompt", ""),
                first_comfort_text=inputs.get("first_comfort_text", ""),
                role_doc_prompt=inputs.get("role_doc_prompt", ""),
            )
        except Exception as e:
            print(f">> [ContextSummaryStart] async context summary 失败: {e}")
        with _context_summary_store_lock:
            _context_summary_store[context_id] = summary_text
            if context_id in _context_summary_inputs:
                _context_summary_inputs.pop(context_id, None)
            if context_id in _context_summary_events:
                _context_summary_events.pop(context_id, None)
            summary_event.set()

    threading.Thread(target=_bg_do_summary, daemon=True).start()
    return ContextSummaryStartResponse(status="starting")


@app.post("/driver/comfort_continue", response_model=ComfortContinueResponse)
def driver_comfort_continue(req: ComfortContinueRequest):
    """
    Wake 后续对话安慰话术：
    1) 根据 context_id 从服务端取回上下文总结 + user_reply 生成后续安慰语
    2) 若提供 persona_id，则对后续安慰语做 TTS 克隆并返回音频 base64
    """
    try:
        user_reply = (req.user_reply or "").strip()
        if not user_reply:
            return JSONResponse(status_code=400, content={"detail": "user_reply is empty"})

        # 等待上下文总结生成完成（Wake 有原则模式时异步计算）
        summary_text = ""
        with _context_summary_store_lock:
            summary_text = _context_summary_store.get(req.context_id, "")
            ev = _context_summary_events.get(req.context_id)

        if not summary_text and ev is not None:
            # 最多等待 20s：确保用户回复后通常可以取到总结
            ev.wait(timeout=20)
            with _context_summary_store_lock:
                summary_text = _context_summary_store.get(req.context_id, "")

        if not summary_text:
            return JSONResponse(status_code=425, content={"detail": "context summary not ready yet"})

        with text_agent_lock:
            followup_text = text_agent.generate_followup_comfort_message(
                context_summary=summary_text,
                user_reply=user_reply,
            )

        if not req.persona_id and not req.preview_item_id:
            return ComfortContinueResponse(assistant_text=followup_text, audio_wav_base64=None)

        try:
            ref_audio, ref_text, language, _xvec = _resolve_clone_ref_for_tts(req.persona_id, req.preview_item_id)
        except FileNotFoundError as e:
            return JSONResponse(status_code=404, content={"detail": str(e)})
        except ValueError as e:
            msg = str(e)
            code = 500 if "manifest" in msg else 400
            return JSONResponse(status_code=code, content={"detail": msg})

        print(">> [ComfortContinue] 开始调用 Qwen3-TTS 克隆合成……")
        with tts_model_lock:
            model = _ensure_tts_model()
            wavs, sr = model.generate_voice_clone(
                text=followup_text,
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=_xvec,
                max_new_tokens=512,
            )

        wav = wavs[0]
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV")
        audio_bytes = buf.getvalue()
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        if req.autoplay:
            print(">> [ComfortContinue] autoplay=true，准备在后端机器播放音频……")
            _play_wav_bytes_on_server(audio_bytes)

        return ComfortContinueResponse(assistant_text=followup_text, audio_wav_base64=audio_b64)
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"detail": str(e)})
    except Exception as e:
        print(f">> [ComfortContinue] 失败: {e}")
        return JSONResponse(status_code=500, content={"detail": f"Comfort continue failed: {e}"})


class LLMSettings(BaseModel):
    ollama_model: str


@app.get("/settings/llm")
def get_llm_settings():
    with text_agent_lock:
        current = getattr(text_agent, "ollama_model", _llm_runtime_settings.get("ollama_model"))
    return {
        "ollama_model": current,
        "allowed_models": ALLOWED_OLLAMA_MODELS,
        "settings_path": SETTINGS_PATH,
    }


@app.post("/settings/llm")
def update_llm_settings(body: LLMSettings):
    model = (body.ollama_model or "").strip()
    if model not in ALLOWED_OLLAMA_MODELS:
        return JSONResponse(status_code=400, content={"detail": f"Unsupported model: {model}", "allowed_models": ALLOWED_OLLAMA_MODELS})
    with text_agent_lock:
        try:
            text_agent.ollama_model = model
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": f"Failed to set model: {e}"})
    _llm_runtime_settings["ollama_model"] = model
    _save_settings_to_disk()
    return {"ollama_model": model, "allowed_models": ALLOWED_OLLAMA_MODELS}


@app.get("/audio_previews")
def list_audio_previews():
    """
    音色预览音频列表。音频文件放在 audio_previews/ 目录，元数据在 audio_previews/manifest.json。
    """
    try:
        manifest = _load_preview_manifest()
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Failed to load manifest: {e}"})

    items = manifest.get("items") or []
    out: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        item_id = str(it.get("id") or "").strip()
        filename = str(it.get("filename") or "").strip()
        if not item_id or not filename:
            continue
        full_path = os.path.join(PREVIEW_AUDIO_ROOT, filename)
        out.append(
            {
                "id": item_id,
                "name": it.get("name") or item_id,
                "description": it.get("description") or "",
                "filename": filename,
                "exists": os.path.exists(full_path),
                "media_type": _guess_media_type(filename),
            }
        )
    return {"items": out}


@app.get("/audio_previews/{item_id}")
def get_audio_preview(item_id: str):
    try:
        manifest = _load_preview_manifest()
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Failed to load manifest: {e}"})

    items = manifest.get("items") or []
    target = None
    for it in items:
        if isinstance(it, dict) and str(it.get("id")) == item_id:
            target = it
            break
    if not target:
        return JSONResponse(status_code=404, content={"detail": f"Preview item {item_id} not found"})

    filename = str(target.get("filename") or "").strip()
    if not filename:
        return JSONResponse(status_code=500, content={"detail": "Invalid manifest entry: missing filename"})

    full_path = os.path.join(PREVIEW_AUDIO_ROOT, filename)
    if not os.path.exists(full_path):
        return JSONResponse(status_code=404, content={"detail": f"Audio file not found: {filename}"})

    media_type = _guess_media_type(filename)

    def _iter():
        with open(full_path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(_iter(), media_type=media_type)

def _persona_dir(persona_id: str) -> str:
    return os.path.join(PERSONA_ROOT, persona_id)


def _persona_meta_path(persona_id: str) -> str:
    return os.path.join(_persona_dir(persona_id), "meta.json")


def _load_persona(persona_id: str) -> Dict[str, Any]:
    meta_path = _persona_meta_path(persona_id)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Persona {persona_id} not found")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 兼容旧目录结构：历史 meta.json 可能把 audio_path 写到 <repo>/personas/...，
    # 现在统一存放在 <repo>/voices/personas/<id>/ref.wav
    default_audio_path = os.path.join(_persona_dir(persona_id), "ref.wav")
    audio_path = meta.get("audio_path")
    if (not audio_path) or (not os.path.exists(str(audio_path))):
        # 尝试把旧路径映射到新路径
        if isinstance(audio_path, str) and ("\\personas\\" in audio_path or "/personas/" in audio_path):
            mapped = os.path.join(_persona_dir(persona_id), "ref.wav")
            if os.path.exists(mapped):
                meta["audio_path"] = mapped
            else:
                meta["audio_path"] = default_audio_path
        else:
            meta["audio_path"] = default_audio_path

    return meta


def _resolve_clone_ref_for_tts(
    persona_id: Optional[str], preview_item_id: Optional[str]
) -> Tuple[str, str, str, bool]:
    """
    解析语音克隆用的参考音频路径与 ref_text。
    preview_item_id 优先于 persona_id（与 /tts/clone_base64 一致）。
    """
    x_vector_only_mode = False
    if preview_item_id:
        manifest = _load_preview_manifest()
        items = manifest.get("items") or []
        target = None
        for it in items:
            if isinstance(it, dict) and str(it.get("id")) == str(preview_item_id):
                target = it
                break
        if not target:
            raise FileNotFoundError(f"Preview item {preview_item_id} not found")
        filename = str(target.get("filename") or "").strip()
        if not filename:
            raise ValueError("Invalid preview manifest entry: missing filename")
        ref_audio = os.path.join(PREVIEW_AUDIO_ROOT, filename)
        if not os.path.exists(ref_audio):
            raise FileNotFoundError(f"Preview audio file not found: {filename}")
        return ref_audio, PREVIEW_CLONE_REF_TEXT, "Chinese", x_vector_only_mode
    if not persona_id:
        raise ValueError("Either persona_id or preview_item_id is required")
    persona = _load_persona(persona_id)
    return (
        str(persona["audio_path"]),
        persona["ref_text"],
        str(persona.get("language", "Chinese")),
        x_vector_only_mode,
    )


def _play_wav_bytes_on_server(audio_bytes: bytes) -> None:
    """
    在后端运行所在机器直接播放 WAV（Windows 优先用 winsound）。
    注意：这不是“推给前端播放”，而是服务器本机扬声器播放。
    """
    try:
        import winsound  # type: ignore

        # SND_MEMORY：直接播放内存中的 WAV 数据；SND_ASYNC：异步播放不阻塞接口返回
        winsound.PlaySound(audio_bytes, winsound.SND_MEMORY | winsound.SND_ASYNC)
        print(">> [PLAY] 已在后端机器发起异步播放。")
    except Exception as e:
        print(f">> [PLAY] 后端自动播放失败（可忽略，仅影响本机播放）: {e}")


@app.get("/personas")
def list_personas() -> List[Dict[str, Any]]:
    personas: List[Dict[str, Any]] = []
    for pid in os.listdir(PERSONA_ROOT):
        meta_path = _persona_meta_path(pid)
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                personas.append(json.load(f))
    return personas


@app.post("/personas")
async def create_persona(
    name: str = Form(...),
    description: str = Form(""),
    ref_text: str = Form(...),
    language: str = Form("Chinese"),
    audio: UploadFile = File(...),
):
    import uuid

    persona_id = str(uuid.uuid4())
    p_dir = _persona_dir(persona_id)
    os.makedirs(p_dir, exist_ok=True)

    audio_path = os.path.join(p_dir, "ref.wav")
    content = await audio.read()
    with open(audio_path, "wb") as f:
        f.write(content)

    meta = {
        "id": persona_id,
        "name": name,
        "description": description,
        "ref_text": ref_text,
        "language": language,
        "audio_path": audio_path,
    }
    with open(_persona_meta_path(persona_id), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


class PersonaUpdate(BaseModel):
    """可选的 persona 字段更新（只传要改的）"""
    name: Optional[str] = None
    description: Optional[str] = None
    ref_text: Optional[str] = None
    language: Optional[str] = None


@app.patch("/personas/{persona_id}")
def update_persona(persona_id: str, body: PersonaUpdate):
    """更新 persona 元数据（如 Content Instructions 存为 description），刷新后仍保留"""
    try:
        meta = _load_persona(persona_id)
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"detail": f"Persona {persona_id} not found"})
    if body.name is not None:
        meta["name"] = body.name
    if body.description is not None:
        meta["description"] = body.description
    if body.ref_text is not None:
        meta["ref_text"] = body.ref_text
    if body.language is not None:
        meta["language"] = body.language
    meta_path = _persona_meta_path(persona_id)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


class TTSCloneRequest(BaseModel):
    persona_id: Optional[str] = None
    preview_item_id: Optional[str] = None
    text: str


@app.post("/tts/clone")
def tts_clone(req: TTSCloneRequest):
    t0 = time.time()
    print(">> [TTS] ==== 新请求开始 ====")
    print(f">> [TTS] 文本长度: {len(req.text)} 字符, persona_id: {req.persona_id}")
    print(">> [TTS] 开始加载说话人与文本信息……")
    try:
        persona = _load_persona(req.persona_id)
    except FileNotFoundError as e:
        print(">> [TTS] 说话人(persona)不存在，终止。")
        return JSONResponse(status_code=404, content={"detail": str(e)})

    t1 = time.time()
    print(f">> [TTS] 说话人配置加载完成，用时 {(t1 - t0) * 1000:.1f} ms")

    ref_audio = persona["audio_path"]
    ref_text = persona["ref_text"]
    language = persona.get("language", "Chinese")

    print(">> [TTS] 文本准备就绪，开始调用 Qwen3-TTS 进行克隆合成（约 0% → 70%）……")
    with tts_model_lock:
        model = _ensure_tts_model()
        wavs, sr = model.generate_voice_clone(
            text=req.text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            max_new_tokens=512,
        )
    t2 = time.time()
    print(f">> [TTS] 克隆语音生成完成（70%），TTS 推理耗时 {(t2 - t1) * 1000:.1f} ms，开始编码为 WAV 并返回（70% → 100%）……")
    wav = wavs[0]

    def _iter():
        import io

        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV")
        buf.seek(0)
        yield from buf

    t3 = time.time()
    print(f">> [TTS] WAV 编码完成并开始流式返回（100%），编码耗时 {(t3 - t2) * 1000:.1f} ms，总耗时 {(t3 - t0):.2f} s")
    print(">> [TTS] ==== 请求结束 ====")
    return StreamingResponse(_iter(), media_type="audio/wav")


class TTSCloneBase64Response(BaseModel):
    audio_wav_base64: str
    sample_rate: int


@app.post("/tts/clone_base64", response_model=TTSCloneBase64Response)
def tts_clone_base64(req: TTSCloneRequest):
    """
    只做语音克隆：给定 text + persona_id，返回 base64 音频，不做任何 LLM 文本生成。
    用于 Wake 首轮安慰语的快速语音合成，避免重复调用 /chat 触发大模型生成。
    """
    t0 = time.time()
    print(">> [TTS] ==== /tts/clone_base64 新请求开始 ====")
    print(f">> [TTS] 文本长度: {len(req.text)} 字符, persona_id: {req.persona_id}, preview_item_id: {req.preview_item_id}")

    try:
        ref_audio, ref_text, language, x_vector_only_mode = _resolve_clone_ref_for_tts(req.persona_id, req.preview_item_id)
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"detail": str(e)})
    except ValueError as e:
        msg = str(e)
        code = 500 if "manifest" in msg else 400
        return JSONResponse(status_code=code, content={"detail": msg})
    try:
        ref_size = os.path.getsize(ref_audio) if ref_audio and os.path.exists(ref_audio) else -1
    except Exception:
        ref_size = -1
    print(f">> [TTS] ref_audio={ref_audio} size={ref_size} ref_text_chars={len(ref_text or '')}")

    with tts_model_lock:
        print(">> [TTS] calling _ensure_tts_model() / generate_voice_clone() ...")
        model = _ensure_tts_model()
        print(">> [TTS] generate_voice_clone() start ...")
        wavs, sr = model.generate_voice_clone(
            text=req.text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
            max_new_tokens=512,
        )
        print(">> [TTS] generate_voice_clone() done.")

    wav = wavs[0]
    buf = io.BytesIO()
    print(">> [TTS] encoding WAV ...")
    sf.write(buf, wav, sr, format="WAV")
    audio_bytes = buf.getvalue()
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    t1 = time.time()
    print(f">> [TTS] /tts/clone_base64 完成，耗时 {(t1 - t0) * 1000:.1f} ms")
    return TTSCloneBase64Response(audio_wav_base64=audio_b64, sample_rate=sr)


class ChatRequest(BaseModel):
    text: str
    subject_id: Optional[str] = None
    # 兼容：如果传 persona_id，则 /chat 自动做克隆 TTS，并返回音频 base64
    persona_id: Optional[str] = None
    # 若为 True，后端机器直接播放（Windows 下生效）；前端仍可用 base64 自行播放
    autoplay: bool = False


class ChatAndTTSCloneRequest(BaseModel):
    """
    一次性完成：
    1）调用文本 Agent 生成回答
    2）用 Qwen3-TTS 做克隆语音
    返回：回答文本 + WAV 音频的 base64
    """

    persona_id: str
    text: str
    subject_id: Optional[str] = None


@app.post("/chat")
def chat(req: ChatRequest):
    """
    使用 Agent/agent.py 中的 DeepSeekAgent 生成文本回复。
    - 默认只返回文本
    - 若请求中携带 persona_id，则自动进行 Qwen3-TTS 克隆，并返回音频 base64
    """
    subject_id = req.subject_id or "default"
    t0 = time.time()
    print(">> [Chat] ==== 新请求开始 ====")
    print(f">> [Chat] subject_id={subject_id}, persona_id={req.persona_id}, 文本长度={len(req.text)} 字符")
    try:
        t_llm_start = time.time()
        with text_agent_lock:
            reply = text_agent.generate_response(subject_id, req.text)
        t_llm_end = time.time()
        print(f">> [Chat] 文本生成完成，LLM 耗时 {(t_llm_end - t_llm_start) * 1000:.1f} ms")
    except Exception as e:
        print(f">> [Chat] 文本生成失败: {e}")
        return JSONResponse(status_code=500, content={"detail": f"Text generation failed: {e}"})

    # 不做 TTS：保持原接口行为
    if not req.persona_id:
        t_end = time.time()
        print(f">> [Chat] 无 persona_id，仅返回文本，总耗时 {(t_end - t0) * 1000:.1f} ms")
        print(">> [Chat] ==== 请求结束 ====")
        return {"assistant_text": reply}

    # 做 TTS：生成完文本立刻克隆语音
    print(">> [Chat] 检测到 persona_id，开始进行语音克隆（文本完成后自动触发）……")
    try:
        t_persona_start = time.time()
        persona = _load_persona(req.persona_id)
        t_persona_end = time.time()
        print(f">> [Chat] 说话人配置加载完成，用时 {(t_persona_end - t_persona_start) * 1000:.1f} ms")
    except FileNotFoundError as e:
        print(">> [Chat] 说话人(persona)不存在，终止。")
        return JSONResponse(status_code=404, content={"detail": str(e)})

    ref_audio = persona["audio_path"]
    ref_text = persona["ref_text"]
    language = persona.get("language", "Chinese")

    print(">> [Chat] 开始调用 Qwen3-TTS 克隆合成（约 0% → 70%）……")
    try:
        t_tts_start = time.time()
        with tts_model_lock:
            model = _ensure_tts_model()
            wavs, sr = model.generate_voice_clone(
                text=reply,
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text,
                max_new_tokens=512,
            )
        t_tts_end = time.time()
        print(f">> [Chat] 语音生成完成（70%），TTS 推理耗时 {(t_tts_end - t_tts_start) * 1000:.1f} ms，开始编码 WAV（70% → 100%）……")
    except Exception as e:
        print(f">> [Chat] 语音克隆失败: {e}")
        return JSONResponse(status_code=500, content={"detail": f"TTS clone failed: {e}"})

    wav = wavs[0]
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    audio_bytes = buf.getvalue()
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    t_encode_end = time.time()
    print(f">> [Chat] WAV 编码完成，耗时 {(t_encode_end - t_tts_end) * 1000:.1f} ms")

    if req.autoplay:
        print(">> [Chat] autoplay=true，准备在后端机器播放音频……")
        _play_wav_bytes_on_server(audio_bytes)

    t_end = time.time()
    print(f">> [Chat] 完成，总耗时 {(t_end - t0):.2f} s")
    print(">> [Chat] ==== 请求结束 ====")
    return {
        "assistant_text": reply,
        "audio_wav_base64": audio_b64,
        "sample_rate": sr,
        "persona_id": req.persona_id,
    }


@app.post("/chat_and_tts/clone")
def chat_and_tts_clone(req: ChatAndTTSCloneRequest):
    """
    先生成文本，再立刻用 Qwen3-TTS 克隆语音。
    - 后台可通过打印日志看到大致进度。
    - 前端拿到文本和 base64 音频后可以立即自动播放。
    """
    subject_id = req.subject_id or "default"

    print(">> [Chat+TTS] 收到一体化请求，开始生成文本（0% → 40%）……")
    try:
        reply = text_agent.generate_response(subject_id, req.text)
    except Exception as e:
        print(f">> [Chat+TTS] 文本生成失败: {e}")
        return JSONResponse(status_code=500, content={"detail": f"Text generation failed: {e}"})

    print(">> [Chat+TTS] 文本生成完成（40%），内容如下：")
    try:
        print(reply)
    except Exception:
        pass

    print(">> [Chat+TTS] 开始加载说话人配置并进行语音克隆（40% → 90%）……")
    try:
        persona = _load_persona(req.persona_id)
    except FileNotFoundError as e:
        print(">> [Chat+TTS] 说话人(persona)不存在，终止。")
        return JSONResponse(status_code=404, content={"detail": str(e)})

    ref_audio = persona["audio_path"]
    ref_text = persona["ref_text"]
    language = persona.get("language", "Chinese")

    try:
        with tts_model_lock:
            model = _ensure_tts_model()
            wavs, sr = model.generate_voice_clone(
            text=reply,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            )
    except Exception as e:
        print(f">> [Chat+TTS] 语音克隆失败: {e}")
        return JSONResponse(status_code=500, content={"detail": f"TTS clone failed: {e}"})

    print(">> [Chat+TTS] 语音克隆完成（90%），开始编码 WAV 并打包 base64（90% → 100%）……")
    wav = wavs[0]
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    audio_bytes = buf.getvalue()
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

    print(">> [Chat+TTS] 完成（100%），返回文本 + 音频。")

    return {
        "assistant_text": reply,
        "audio_wav_base64": audio_b64,
        "sample_rate": sr,
        "persona_id": req.persona_id,
    }


@app.get("/personas/{persona_id}/audio")
def get_persona_audio(persona_id: str):
    try:
        persona = _load_persona(persona_id)
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"detail": str(e)})

    audio_path = persona.get("audio_path")
    if not audio_path or not os.path.exists(audio_path):
        return JSONResponse(status_code=404, content={"detail": "Reference audio not found"})

    def _iter():
        with open(audio_path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(_iter(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

