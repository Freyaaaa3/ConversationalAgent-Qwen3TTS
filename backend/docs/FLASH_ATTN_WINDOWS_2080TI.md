# 在 Windows 上为 RTX 2080 Ti（Turing）启用 FlashAttention（Triton 版）

RTX 2080 Ti 属于 Turing 架构（算力 7.5），官方 flash-attn 的 CUDA 实现只支持 Ampere 8.0+。  
**flash-attn-triton** 支持 Turing（含 2080 Ti），但在 Windows 上依赖的 **Triton 官方没有 Windows wheel**，需要先装社区版 **triton-windows**。

## 步骤（在项目 conda/venv 下执行）

### 1. 安装 Triton（Windows 版，建议固定 3.2.x）

```bash
pip install "triton-windows==3.2.0.post21"
```

说明：`flash-attn-triton` 目前会 `import triton.tools.experimental_descriptor`，在部分较新的 `triton-windows` 版本中该模块可能不存在，因此建议固定到 `3.2.0.post21`（已验证可用）。

### 2. 安装 flash-attn-triton（不拉官方 triton，避免冲突）

```bash
pip install flash-attn-triton --no-deps
```

再补装它声明的其它依赖（不含 triton），例如：

```bash
pip install torch einops
```

注意：`flash-attn-triton` 的元数据要求 `torch>=2.6.0`。如果你当前环境的 torch 较低，可能仍可 import，但建议升级到满足要求的版本以避免运行期问题。

### 3. 验证

```bash
python -c "import triton; import flash_attn; print('triton', triton.__version__); print('flash_attn ok')"
```

### 4. 启动后端

```bash
uvicorn backend_tts:app --host 0.0.0.0 --port 8000
```

日志中应出现：`>> [TTS] 使用 FlashAttention-2（Triton，Turing 7.5+）。`

## 说明

- **triton-windows**：社区维护的 Triton Windows 版，见 [triton-windows](https://github.com/woct0rdho/triton-windows)。
- **flash-attn-triton**：Triton 实现的 FlashAttention 2，支持算力 7.5+（Turing/Ampere）。
- 若 `pip install flash-attn-triton` 仍尝试拉官方 `triton` 并报错，用 `--no-deps` 后手动安装 `triton-windows` 与其它依赖即可。
