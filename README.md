# QwenTTS 座舱实时聊天智能体

注：这只是很初步的一个中间版本！这只是很初步的一个中间版本！这只是很初步的一个中间版本！应该有些小bug。
仅做到语音克隆、实时对话。
非完整版——智能体的prompt、memory和检索增强都没有加入，放这里仅仅用作中间状态存储。

## 概述
座舱实时聊天智能体是一款基于先进的Qwen3-TTS 语音生成模型开发的智能交互系统，专为座舱环境设计。本项目致力于为用户提供自然、流畅、个性化的语音交互体验，实现了从语音识别到智能回复再到语音生成的完整流程。

### 核心价值

- **无缝语音交互**：通过集成先进的语音识别和语音生成技术，实现与智能体的自然对话
- **个性化语音体验**：支持语音克隆功能，让智能体使用偏好的声音进行交流
- **本地部署保障隐私**：所有处理均在本地完成，无需将敏感数据上传至云端

### 应用场景

- **智能座舱助手**：提供导航、娱乐、信息查询等语音交互服务
- **个人语音助手**：根据用户喜好定制专属语音形象
- **教育培训**：为语言学习、朗读练习等场景提供高质量语音输出

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/qwen3_tts_logo.png" width="400"/>
<p>

<p align="center">
&nbsp&nbsp🤗 <a href="https://huggingface.co/collections/Qwen/qwen3-tts">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/collections/Qwen/Qwen3-TTS">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://qwen.ai/blog?id=qwen3tts-0115">技术博客</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2601.15621">论文</a>&nbsp&nbsp
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen3-TTS">Hugging Face 演示</a>&nbsp&nbsp | &nbsp&nbsp 🖥️ <a href="https://modelscope.cn/studios/Qwen/Qwen3-TTS">ModelScope 演示</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png">微信</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://help.aliyun.com/zh/model-studio/qwen-tts-realtime">API</a>

</p>

## 项目简介

本项目是一个基于 Qwen3-TTS 模型的本地座舱实时聊天智能体，提供以下核心功能：

- **语音克隆**：通过短音频快速克隆说话人的声音
- **实时对话**：集成语音识别、智能模型回复、语音生成的完整对话流程
- **本地部署**：所有功能均可在本地环境运行，保护隐私
- **座舱适配**：针对座舱环境优化的交互体验

## 核心特性

- **低延迟响应**：基于 Qwen3-TTS 的流式生成能力，实现实时语音交互
- **多语言支持**：覆盖中文、英语、日语、韩语、德语、法语、俄语、葡萄牙语、西班牙语和意大利语
- **情感表达**：支持通过自然语言指令控制语音的情感、语速和语调
- **高质量语音**：采用先进的语音生成技术，提供自然流畅的语音输出

## 快速开始

### 环境准备

推荐使用 Python 3.12 环境，并创建一个隔离的虚拟环境：

```bash
conda create -n qwen-tts python=3.12 -y
conda activate qwen-tts
```

安装必要的依赖：

```bash
pip install -U qwen-tts
# 推荐安装 FlashAttention 3 以减少 GPU 内存使用
pip install -U flash-attn --no-build-isolation
```

如果您的机器 RAM 小于 96GB 但有较多 CPU 核心，请使用：

```bash
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
```

### 基本使用

#### 语音克隆

```python
import torch
import soundfile as sf
from qwen_tts import QwenTTSModel

# 加载模型
model = QwenTTSModel.from_pretrained(
    "Qwen/QwenTTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# 参考音频和文本
ref_audio = "path/to/reference_audio.wav"  # 本地音频文件路径
ref_text = "这是一段参考音频的文本内容"

# 生成克隆语音
wavs, sr = model.generate_voice_clone(
    text="你好，这是克隆后的语音示例",
    language="Chinese",
    ref_audio=ref_audio,
    ref_text=ref_text,
)

# 保存生成的音频
sf.write("output_clone.wav", wavs[0], sr)
```

#### 实时对话流程

1. **语音识别**：使用本地语音识别模型将用户语音转换为文本
2. **智能回复**：将识别的文本输入到对话模型中获取回复
3. **语音生成**：将回复文本转换为语音输出

```python
# 完整的实时对话示例
import torch
import soundfile as sf
import speech_recognition as sr
from qwen_tts import QwenTTSModel
from your_chat_model import ChatModel  # 替换为实际的对话模型

# 初始化模型
recognizer = sr.Recognizer()
chat_model = ChatModel()  # 初始化对话模型
tts_model = QwenTTSModel.from_pretrained(
    "Qwen/QwenTTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# 实时对话循环
while True:
    # 1. 语音识别
    with sr.Microphone() as source:
        print("请说话...")
        audio = recognizer.listen(source)
    
    try:
        user_text = recognizer.recognize_google(audio, language="zh-CN")
        print(f"你: {user_text}")
        
        # 2. 智能回复
        response = chat_model.generate_response(user_text)
        print(f"智能体: {response}")
        
        # 3. 语音生成
        wavs, sr = tts_model.generate_custom_voice(
            text=response,
            language="Chinese",
            speaker="Vivian",  # 可选择不同的预设语音
            instruct="自然友好的语气",
        )
        
        # 播放生成的语音
        # 这里需要根据您的环境选择合适的音频播放方式
        # 例如使用 sounddevice 库播放
        import sounddevice as sd
        sd.play(wavs[0], sr)
        sd.wait()
        
    except sr.UnknownValueError:
        print("抱歉，我没有听清楚，请再说一遍。")
    except sr.RequestError as e:
        print(f"语音识别服务出错: {e}")
```

## 座舱适配

针对座舱环境的特殊优化：

- **噪音抑制**：增强对座舱环境噪音的抵抗能力
- **唤醒词检测**：支持自定义唤醒词，实现免触控交互
- **多模态输入**：支持语音、手势等多种输入方式
- **驾驶安全**：优化交互流程，减少驾驶员分心

## 模型下载

您可以通过以下方式下载模型：

### 通过 ModelScope 下载（推荐中国大陆用户）

```bash
pip install -U modelscope
modelscope download --model Qwen/QwenTTS-Tokenizer-12Hz  --local_dir ./QwenTTS-Tokenizer-12Hz 
modelscope download --model Qwen/QwenTTS-12Hz-1.7B-CustomVoice --local_dir ./QwenTTS-12Hz-1.7B-CustomVoice
modelscope download --model Qwen/QwenTTS-12Hz-1.7B-VoiceDesign --local_dir ./QwenTTS-12Hz-1.7B-VoiceDesign
modelscope download --model Qwen/QwenTTS-12Hz-1.7B-Base --local_dir ./QwenTTS-12Hz-1.7B-Base
```

### 通过 Hugging Face 下载

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/QwenTTS-Tokenizer-12Hz --local_dir ./QwenTTS-Tokenizer-12Hz
huggingface-cli download Qwen/QwenTTS-12Hz-1.7B-CustomVoice --local_dir ./QwenTTS-12Hz-1.7B-CustomVoice
huggingface-cli download Qwen/QwenTTS-12Hz-1.7B-VoiceDesign --local_dir ./QwenTTS-12Hz-1.7B-VoiceDesign
huggingface-cli download Qwen/QwenTTS-12Hz-1.7B-Base --local_dir ./QwenTTS-12Hz-1.7B-Base
```

## 系统要求

- **硬件**：
  - GPU：至少 8GB 显存（推荐 16GB+）
  - CPU：至少 4 核
  - 内存：至少 16GB
- **软件**：
  - Python 3.10+
  - PyTorch 2.0+
  - CUDA 11.7+

## 常见问题

### 1. 语音识别准确率不高

- 确保使用高质量的麦克风
- 在安静的环境中使用
- 考虑使用更专业的语音识别模型

### 2. 语音生成延迟较高

- 确保使用 FlashAttention 2/3
- 考虑使用较小的模型（如 0.6B 版本）
- 优化硬件配置，使用更强大的 GPU

### 3. 语音克隆效果不理想

- 提供更长的参考音频（建议 3-5 秒）
- 确保参考音频清晰，无背景噪音
- 尝试调整生成参数

---

