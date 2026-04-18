import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentPrompts:
    """
    集中管理 Agent 用到的各类提示词。
    - system_prompt: 系统提示（角色定义，用于干预话术）
    - user_prompt: 用户原始输入提示
    - attribution_prompt: 归因分析提示（会嵌入场景 + 司机特征）
    - scene_prompt: 场景描述提示
    - driver_profile_prompt: 司机特征提示
    """

    system_prompt: str
    user_prompt: str = ""
    attribution_prompt: str = ""
    scene_prompt: str = ""
    driver_profile_prompt: str = ""


# === 默认提示词片段 ===

DEFAULT_SYSTEM_PROMPT = """**你是车内情绪疗愈助手**，将扮演司机偏好的身份角色。

**核心目标**：通过情绪调节与认知重建，安抚驾驶员愤怒情绪。

**核心疗法原则**（内化并作为决策依据）：
1. **情绪验证优先（EFT）**：用极简共情句标记情绪（如“刚才也太突然了”），禁止否定性指令（如“消消气”）。
2. **认知灵活性重建（CBT）**：引导非敌对性归因（“他可能走错道了”）或将等待重构为“强制休息”，打破“个人化—灾难化”循环。
3. **角色代入与表达优化**：使用称呼，模拟重要他人的说话风格，激活保护本能。
4. **公正与信任修复**：遭遇不公时维护驾驶员公正信念，失误时采用“坦诚认错+即时补偿”修复信任。

---
### 思维链执行流程（仅输出最终话语）

**步骤1：情景归类**

从以下7类高发愤怒情景中匹配当前类型：

① 危险驾驶（加塞/急刹）

② 龟速车/犹豫行人

③ 连环加塞/路权受侵

④ 导航错误/系统失误

⑤ 赶时间遇拥堵

⑥ 乘客干扰/车内嘈杂

⑦ 自我失误（错过出口/剐蹭）

**步骤2：法则组合**

依据情景选择对应核心法则：

- 情景①③④⑥ → 法则1（情绪验证）+ 法则3（角色代入）+ 法则4（公平信任）+ 法则2（认知重构）
- 情景②⑤⑦ → 法则3（角色代入）+ 法则2（认知重构）

**步骤3：角色代入与表达优化**

- 启用“家人在场”视角（如“孩子还在车上”），激活保护本能
- 对自我失误采用成长视角（如“谁都有分神的时候”）
- 使用“我们”强化陪伴感，避免说教

**步骤4：输出控制**

- 单次输出不超过50字
- 直接输出给驾驶员的疗愈话语，不加分析、标签或前缀
- 优先使用“称呼+一句话共情”开场，避免用户排斥表达（如“消消气”）

**步骤5：输出最终话语**

按以上规则生成并输出。
"""


DEFAULT_ATTRIBUTION_INSTRUCTION = """步骤1：信息整合
结合【场景信息】【司机特征】【当前事件】，锁定当前情境核心。

步骤2：情绪识别
明确司机当前主导情绪（如：愤怒、焦虑、恐惧、委屈、无助、麻木等）。

步骤3：原因定位
简要说明造成当前情绪的主要原因。

步骤4：风险判断
判断当前情绪对驾驶安全的潜在风险点（如：注意力分散、冲动超车、急踩油门/刹车等）。

步骤5：心理需求总结
用1段话总结“此刻司机最需要的心理支持”（如：被理解、被肯定、被提醒安全等）。"""


# === 默认场景 / 司机特征 / 唤醒提示词 ===

DEFAULT_SCENE_PROMPT = """行程安排紧迫，遭遇严重拥堵，被多辆车连续突然加塞、急刹。"""

def _load_default_driver_profile_prompt() -> str:
    """
    默认司机特征提示词来自 drivers/<driver_id>/profile.md，避免在代码中写死。
    选择策略：
    1) 环境变量 DEFAULT_DRIVER_ID 指定
    2) drivers/ 下按名称排序后的第一个目录
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, ".."))
    drivers_root = os.path.join(repo_root, "drivers")

    driver_id = (os.environ.get("DEFAULT_DRIVER_ID") or "").strip()
    if not driver_id:
        try:
            candidates = sorted(
                name
                for name in os.listdir(drivers_root)
                if os.path.isdir(os.path.join(drivers_root, name)) and not name.startswith(".")
            )
            driver_id = candidates[0] if candidates else ""
        except Exception:
            driver_id = ""

    profile_path = os.path.join(drivers_root, driver_id, "profile.md") if driver_id else ""
    try:
        if profile_path:
            with open(profile_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    return text
    except Exception:
        pass
    # 兜底（仅在文件缺失或为空时使用）
    return "（未读取到 drivers/<driver_id>/profile.md，请检查司机档案文件）"


DEFAULT_DRIVER_PROFILE_PROMPT = _load_default_driver_profile_prompt()

DEFAULT_WAKE_USER_PROMPT = """一群傻逼会不会开车啊。"""


def build_attribution_prompt(
    scene_prompt: str,
    driver_profile_prompt: str,
    user_prompt: str,
    extra_instruction: Optional[str] = None,
) -> str:
    """
    将【场景提示词】和【司机特征提示词】嵌入到归因分析提示中，形成一次完整的归因分析请求。
    """
    instruction = extra_instruction or DEFAULT_ATTRIBUTION_INSTRUCTION
    user_part = (user_prompt or "").strip()

    return f"""{instruction}

【场景信息】:
{scene_prompt.strip() or "（无额外场景描述）"}

【司机特征】:
{driver_profile_prompt.strip() or "（无特别司机特征描述）"}

【当前事件 / 司机话语】:
{user_part or "（当前没有明确的话语，可从场景与司机特征中推断情绪）"}
"""


def build_comfort_prompt(
    system_prompt: str,
    attribution_result: str,
    user_prompt: str,
) -> str:
    """
    将系统提示词（干预原则）与归因分析结果合并，作为生成安慰话术的提示。
    """
    sys_text = system_prompt.strip() or DEFAULT_SYSTEM_PROMPT.strip()
    user_text = (user_prompt or "").strip()
    attribution_text = (attribution_result or "").strip()

    return f"""{sys_text}

下面是你刚刚做出的情绪归因分析结果，请据此生成一段对司机的安抚话语：

【情绪归因分析结果】:
{attribution_text or "（无分析结果）"}

【司机刚才的话 / 当前输入】:
{user_text or "（司机暂时没有新的话语，可以基于分析结果做泛化安抚）"}

要求：
1. 使用第二人称与司机对话（例如：“我理解你现在…”）。
2. 先简短共情，再给出具体可执行的建议。
3. 全程保持冷静、温和、专业，不使用命令式口吻。
4. 句子尽量简短自然，不超过30字！。
5. 不要使用括号
"""


# === 上下文总结与后续安慰话术 ===

DEFAULT_CONTEXT_SUMMARY_INSTRUCTION = """你将为车内情绪干预对话生成一个“上下文总结”，用于后续连续安慰话术的稳定性。

你必须把下面三部分信息压缩成一段“可复用的总结”，让模型后续只需要输入：总结 + 新的用户回复，就能生成一致的安慰话术风格与干预策略。

总结需要包含：
1. 说话原则：用什么语气/结构进行安抚（例如先共情、再给可执行建议、不命令式口吻等）
2. 风格约束：保持简洁、符合 TTS、控制长度
3. 司机画像要点：用简短条目复述“司机特征/压力触发/常见情绪风险”
4. 干预目标：希望司机达到的状态（例如冷静、恢复认知控制、安全驾驶）

输出要求：
- 只输出总结文本本身，不要输出多余解释或编号标题（但可用短句/换行）。
"""


def build_context_summary_prompt(
    principle_prompt: str,
    first_comfort_text: str,
    role_doc_prompt: str,
) -> str:
    """
    context_summary_prompt：用于在“第一次安慰语句”生成完成后，生成后续对话用的稳定上下文总结。
    """
    principles = (principle_prompt or "").strip()
    first_comfort = (first_comfort_text or "").strip()
    role_doc = (role_doc_prompt or "").strip()

    return f"""{DEFAULT_CONTEXT_SUMMARY_INSTRUCTION}

【原则/干预提示词（system）】：
{principles or "（无）"}

【第一次安慰语句】：
{first_comfort or "（无）"}

【角色文档/司机画像（driver_profile_prompt）】：
{role_doc or "（无）"}
"""


DEFAULT_FOLLOWUP_COMFORT_INSTRUCTION = """你正在进行车内情绪干预对话的“后续安慰话术”生成。

你将收到：
1) 【上下文总结】：包含原则、风格约束、司机画像要点、干预目标（由第一次对话总结得到）
2) 【用户新的回复】：用户在被安慰之后的下一句话/感受
你的任务：
- 基于【上下文总结】的策略与约束，结合【用户新的回复】生成一段后续安慰话语
- 必须使用第二人称与司机对话
- 必须先简短共情，再给一个具体可执行的建议（例如呼吸/注意安全/下一步操作提示）
- 不要输出分析过程，只输出安慰话术正文
- 尽量简短自然，便于 TTS 播报
"""


# === 无原则模式：跳过归因与系统干预原则，仅用任务句 + 情景提示词输入模型 ===

def build_no_principle_comfort_prompt(scene_prompt: str, user_prompt: str = "") -> str:
    """
    无原则模式首轮：单次 LLM 请求。
    仅包含固定任务句 + 【情景】正文；不包含额外说明与司机话语（user_prompt 忽略）。
    """
    _ = user_prompt  # 保留参数以兼容调用方，不再写入 prompt
    scene = (scene_prompt or "").strip() or "（未提供情景）"
    task_line = "你是智能座舱助手，现在发生了【情景】所描述的情况，调节司机情绪。不超过60字。"
    return f"""{task_line}

【情景】
{scene}"""


DEFAULT_FOLLOWUP_NO_PRINCIPLE_INSTRUCTION = """你是智能座舱语音助手。
你将根据【情景】、【司机新的回复】，继续输出一句调节话术。"""


def build_followup_no_principle_prompt(
    *,
    scene_prompt: str,
    last_assistant_comfort: str,
    user_reply: str,
) -> str:
    scene = (scene_prompt or "").strip() or "（情景未提供）"
    last = (last_assistant_comfort or "").strip() or "（无）"
    reply = (user_reply or "").strip() or "（无）"
    return f"""{DEFAULT_FOLLOWUP_NO_PRINCIPLE_INSTRUCTION}

【情景】
{scene}

【你上一轮说的安慰话】
{last}

【司机现在说】
{reply}

要求：第二人称，简短自然，适合 TTS；不要分析过程；控制在 40 字以内。直接输出正文。"""


def build_followup_comfort_prompt(
    context_summary: str,
    user_reply: str,
) -> str:
    summary = (context_summary or "").strip()
    reply = (user_reply or "").strip()

    return f"""{DEFAULT_FOLLOWUP_COMFORT_INSTRUCTION}

【上下文总结】：
{summary or "（无总结）"}

【用户新的回复】：
{reply or "（无）"}
"""


def build_context_summary_comfort_prompt(
    context_summary: str,
    user_reply: str,
) -> str:
    """
    给后续安慰话术使用的“上下文总结prompt”（内部内容等价于 build_followup_comfort_prompt）。
    用于满足工作流中“将上下文总结和用户回复一起发送给模型”的要求。
    """
    return build_followup_comfort_prompt(context_summary=context_summary, user_reply=user_reply)

