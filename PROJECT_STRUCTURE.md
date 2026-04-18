# QwenTTS 项目结构说明

本文档说明当前仓库中四个核心部分（前端、后端、Agent 实现、声线存储）的目录组织，以及主要子目录与代码文件的大致功能，便于理解和维护。

> 约定：不逐一罗列第三方依赖目录（如 `node_modules/`、Python 的 `site-packages/` 等），仅说明项目自身代码与配置。

---

## 一、frontend/ —— 前端（Lumina Drive Web UI）

路径：`frontend/lumina-drive---intelligent-voice-assistant/`

- `index.html`  
  - 前端单页入口 HTML，挂载点为 `#root`，引入 Tailwind CDN 和自定义样式 `index.css`。

- `index.css`  
  - 全局样式补充：页面高度、深色主题、滚动条样式（`custom-scrollbar`）、淡入动画（`animate-fade-in`）等。

- `server.ts`  
  - 使用 Express + Vite 中间件启动本地开发服务器（端口 3000）。  
  - 暴露 `/api/dialog-turn` 占位接口，转发到 `QWEN_TTS_URL`（用于对接 Qwen TTS 的 HTTP 服务）。

- `vite.config.ts`  
  - Vite 构建配置，启用 React 插件、别名 `@ -> src`、HMR 端口配置等。

- `tsconfig.json`  
  - TypeScript 编译选项（`target`、`jsx`、paths 别名等），配合 Vite 使用。

- `package.json` / `package-lock.json`  
  - 前端依赖与脚本定义（`npm run dev/build` 等）。

- `src/`（前端源码根目录）
  - `src/index.tsx`  
    - React 入口文件，创建 React 根节点并渲染 `<App />`。

  - `src/app/App.tsx`  
    - 前端应用的主组件，仅负责：
      - Tab 切换（`home / voices / music / settings`）。
      - 管理全局状态（连接状态、当前 voice、ASR 文本、设置、预览音频等）。
      - 将状态与事件处理函数下发给各 Tab 组件。
      - 与后端 HTTP 接口交互（通过 `services/backend.ts`）。

  - `src/components/Sidebar.tsx`  
    - 左侧导航栏组件：包含 Home / Voices / Music / Settings 图标按钮，以及 API Key 图标按钮（目前占位）。

  - `src/components/Visualizer.tsx`  
    - 中央 Siri 风格音频可视化组件，使用 `<canvas>` 绘制多条彩色波形，随 `isActive`、`isSpeaking` 状态变化。

  - `src/components/modals/CloneVoiceModal.tsx`  
    - “Clone New Voice” 弹窗组件：用于上传参考音频、填写 persona 文本与参考文本，提交克隆请求。

  - `src/pages/HomeTab.tsx`  
    - 主页 Tab：展示视觉波形、最新对话文本、Wake/Play/Disconnect 按钮、状态提示与实时 ASR 文本。

  - `src/pages/VoicesTab.tsx`  
    - 人格 / 声线管理 Tab：展示所有 persona，支持：
      - 选择当前 persona。
      - 编辑 persona 描述（前端防抖 + 后端 `PATCH /personas/{id}`）。
      - 预听 persona 参考音频 `GET /personas/{id}/audio`。
      - 打开克隆弹窗。

  - `src/pages/MusicTab.tsx`  
    - 预览音色 Tab：从 `GET /audio_previews` 拉取列表，点击播放 `GET /audio_previews/{id}`，管理当前正在播放的预览项。

  - `src/pages/SettingsTab.tsx`  
    - 设置 Tab：展示并可修改：
      - TTS 设置（`GET/POST /settings/tts`）：`use_flash_attention`、`use_4bit`、`use_torch_compile`，支持“Apply & Reload”重载模型。
      - 文本 LLM 设置（`GET/POST /settings/llm`）：当前模型与候选模型列表。

  - `src/hooks/useAudioPlayer.ts`  
    - 抽象音频播放逻辑：内部维护 `audioRef`、`isSpeaking`，提供 `playUrl` 和 `stop`。

  - `src/hooks/useSpeechRecognition.ts`  
    - 抽象浏览器 SpeechRecognition 状态机：负责开始/停止监听、处理中间/最终结果、回调提交最终文本，支持忽略播放期间的“回声 ASR”。

  - `src/services/apiClient.ts`  
    - 统一封装 `fetch`，提供：
      - `requestJson<T>()`：带状态码检查与错误信息提取的 JSON 请求。
      - `requestArrayBuffer()`：用于获取二进制音频数据与 MIME 类型。

  - `src/services/backend.ts`  
    - 前端访问后端 HTTP API 的封装层：集中定义
      - `listPersonas / createPersona / patchPersonaDescription / getPersonaAudio`
      - `chat`（调用后端 `/chat`）
      - `listAudioPreviews / getAudioPreview`
      - `getTtsSettings / setTtsSettings`
      - `getLlmSettings / setLlmSettings`

  - `src/types/index.ts`  
    - 前端使用的基础类型：`AppStatus`、`TranscriptionEntry`、`VoiceProfile`、`AppTab` 等。

---

## 二、backend/ —— 后端（TTS + 对话服务）

路径：`backend/`

- `backend/app/backend_tts.py`  
  - **主后端服务入口**，基于 FastAPI：
    - 管理模型加载与运行时设置：
      - 使用 `Qwen3_TTS_12Hz_0.6B_Base` 模型（路径 `backend/model/...`）。
      - 支持 FlashAttention、4-bit 量化、`torch.compile` 模式。
      - 使用 `runtime_settings.json` 持久化 TTS/LLM 运行时配置。
    - 管理 personas（声线配置）：
      - `voices/personas/` 中存储 persona 的元数据与参考音频。
      - `GET /personas`：列出所有 persona。
      - `POST /personas`：上传音频 + 文本克隆新 persona。
      - `PATCH /personas/{id}`：更新 name/description/ref_text/language。
      - `GET /personas/{id}/audio`：返回参考音频。
    - 管理 audio previews：
      - `voices/audio_previews/manifest.json` ：预览音频清单。
      - `GET /audio_previews`：列出可用预览。
      - `GET /audio_previews/{item_id}`：返回对应预览音频。
    - 设置接口：
      - `GET/POST /settings/tts`：读写 TTS 运行时设置并可选择是否重载模型。
      - `GET/POST /settings/llm`：读写文本 Agent 的 Ollama 模型名及候选列表。
    - 对话 + TTS：
      - 使用 `agent.DeepSeekAgent` 通过 Ollama 生成文本回复：
        - `POST /chat`：纯文本对话；若传入 `persona_id`，则自动克隆语音并返回 `audio_wav_base64`。
        - `POST /tts/clone`：直接根据 persona + 文本克隆生成语音，返回流式 WAV。
        - `POST /chat_and_tts/clone`：一步完成文本生成 + 语音克隆并返回文本 + base64 音频。
    - 模型加载逻辑：
      - 通过 `_ensure_tts_model()` 懒加载模型，避免模块导入时阻塞。
      - 尝试开启 FlashAttention / 4-bit 量化，失败时自动回退全精度。

- `backend/app/qwen_tts_server.py`  
  - 独立的简化版 `/tts` FastAPI 服务：
    - 使用 `Qwen3TTSModel.generate_voice_clone`，仅支持简单文本合成（无 persona，`x_vector_only_mode=True`）。
    - 端口由 `QWEN_TTS_PORT` 控制（默认 8001），可供 `server.ts` 的 `/api/dialog-turn` 使用。

- `backend/pkg/qwen_tts/`  
  - Qwen 官方 `qwen-tts` Python 包的本地副本：  
    - `core/`：模型架构与 tokenizer 等实现。  
    - `inference/qwen3_tts_model.py`：`Qwen3TTSModel` 推理封装。  
  - 供 `backend_tts.py` 和 `qwen_tts_server.py` 导入使用。

- `backend/model/Qwen3_TTS_12Hz_0.6B_Base/`  
  - 对应 Qwen3-TTS 模型权重与配置（从官方仓库或 ModelScope/HF 下载）。

- `backend/config/runtime_settings.json`  
  - 存储 TTS 和 LLM 的运行时配置（是否开 FlashAttention/4-bit/compile、当前 Ollama 模型名等）。

- `backend/docs/`  
  - `FLASH_ATTN_WINDOWS_2080TI.md` 等文档：针对 Windows + 2080Ti 的 FlashAttention 配置说明。

- `backend/examples/`  
  - 官方 Quickstart 示例脚本：  
    - `test_model_12hz_base.py`：基础 TTS 推理示例。  
    - `test_model_12hz_custom_voice.py`：自定义声音示例。  
    - `test_model_12hz_voice_design.py`：声线设计示例。  
    - `test_tokenizer_12hz.py`：Tokenizer 使用示例。

- `backend/finetuning/`  
  - 微调相关代码：
    - `dataset.py`：微调数据集定义。
    - `prepare_data.py`：将原始数据转为微调格式。
    - `sft_12hz.py`：基于 12Hz 模型的 SFT 训练脚本。
    - `README.md`：微调流程说明。

- `backend/scripts/`  
  - `clone_tts.py`：简单的命令行脚本，加载模型并使用参考音频进行语音克隆，输出 WAV 文件。
  - `test.py`：项目级测试脚本（通常用于快速验证安装与模型加载情况）。

- `backend/requirements.txt`  
  - 后端 Python 环境依赖列表：FastAPI、uvicorn、torch、soundfile、librosa、transformers、onnxruntime、python-multipart、requests 等。

---

## 三、agent/ —— 对话 Agent 实现

路径：`agent/`

- `agent/__init__.py`  
  - 简单的包初始化文件，使 `agent` 可被 `backend_tts.py` 通过 `from agent.agent import DeepSeekAgent` 导入。

- `agent/agent.py`  
  - 封装 LLM 调用的文本 Agent，实现类：
    - `DeepSeekAgent`：
      - 属性：
        - `ollama_model`：当前使用的 Ollama 模型名称（默认从环境变量或 `qwen3:8b`）。
        - `ollama_host`：Ollama 服务地址（默认 `http://127.0.0.1:11434`）。
      - 方法：
        - `generate_response(subject_id: str, text: str) -> str`：  
          构造请求调用 Ollama `POST /api/generate`，返回文本回复。`subject_id` 暂为预留字段，可用于未来的个性化/话题管理。
  - 此 Agent 被 `backend/app/backend_tts.py` 中的 `/chat` 和 `/chat_and_tts/clone` 复用。

- `agent/examples/14_SenceVoice_QWen2VL_edgeTTS_realTime.py`  
  - 复杂的多模态/实时语音 Demo 脚本（摄像头 + 麦克风 + VAD + Qwen2-VL + EdgeTTS 等）。  
  - 与当前 Lumina Drive 主流程解耦，作为实验/示例脚本保留在 `agent/examples/` 下。

---

## 四、voices/ —— 声线与音频资源存储

路径：`voices/`

- `voices/personas/`  
  - 后端管理的 persona 数据目录，每个子目录对应一个 persona ID（UUID）：  
    - 内含参考音频、meta JSON（包含 name、description、ref_text、language 等），由 `backend/app/backend_tts.py` 的 `/personas` 系列接口读写。

- `voices/audio_previews/`  
  - 存放可供前端“音色预览”（Music Tab）使用的音频，以及清单文件：
    - `manifest.json`：包含 `items` 列表（每个元素含 `id/name/description/filename` 等），由后端的 `GET /audio_previews` 与 `GET /audio_previews/{id}` 读取。

- `voices/samples/`  
  - `voice_03.wav` 等手动测试/示例用的参考音频文件（例如早期在 `clone_tts.py` 中使用的样本）。

- `voices/output/`  
  - `output_voice_clone.wav` 等脚本输出的合成语音文件（如根目录原 `clone_tts.py` 示例生成的文件）。

---

## 五、根目录其它文件

- `.github/`  
  - CI/CD、GitHub 配置相关（若存在）。

- `.gitignore`  
  - Git 忽略规则。

- `pyproject.toml` / `MANIFEST.in`  
  - 原 Qwen3-TTS Python 包的打包配置（保留用于参考或二次打包）。

- `README.md`  
  - 官方 Qwen3-TTS 仓库的说明文档（模型介绍、Quickstart、下载方式等）。

- `env/`  
  - 你的本地 Conda/虚拟环境目录（建议不要提交到远程仓库，作为本机环境即可）。

---

如果后续你新增模块（例如新的 Agent 类型、新的声线管理工具或前端页面），建议继续按上述四大板块扩展，并在本文件中补充相应目录与文件说明，保持结构清晰。+
