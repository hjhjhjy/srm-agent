"""多轮对话记忆与指代消解（Phase 4）。

解决的实际问题
--------------
用户第二轮问「它怎么申请？」时，Agent 并不知道「它」指什么，容易检索跑偏或给出
与上下文无关的答案。本模块提供确定性（零依赖、可离线、可复现）的能力：

1. ``has_coref``：检测当前问题是否含有指代 / 省略，需要结合上文才能理解。
2. ``build_coref_context``：从最近几轮对话提炼「指代消解上下文」，拼到检索 / 回答
   查询里，让单轮模型也能拿到必要的背景。
3. ``MemoryStore``：按 session 保存对话历史，长会话下对旧消息做**确定性摘要压缩**，
   避免 ``messages`` 无限膨胀吃掉 token 预算。不引入 LLM 摘要，纯规则折叠。
4. ``compress_messages``：超长对话时保留系统/最新若干条，丢弃中间（确定性）。

设计约束：离线确定性、零额外依赖、CI 常绿 —— 因此全部用 stdlib，核心词表可解释。
"""
from __future__ import annotations

# 中文指代 / 省略的高频标记。命中即认为该问题需要上文才能理解。
_COREF_MARKERS = [
    "它", "他", "她", "它们", "他们", "她们",
    "这个", "那个", "这", "那", "该", "其", "此",
    "上述", "上面", "前面", "之前", "刚刚", "前面提到的",
    "前者", "后者", "上一步", "刚才", "同样", "还是", "也是",
]


def has_coref(text: str) -> bool:
    """判断文本是否含有指代 / 省略成分（需要对话上文才能理解）。"""
    if not text:
        return False
    return any(m in text for m in _COREF_MARKERS)


def _role_label(role: str) -> str:
    return "用户" if role == "user" else ("助手" if role == "assistant" else role)


def build_coref_context(
    history: list[dict[str, str]],
    *,
    max_turns: int = 4,
    max_len: int = 160,
) -> str:
    """从对话历史提炼指代消解上下文。

    Args:
        history: 形如 ``[{"role": "user"/"assistant", "content": "..."}, ...]``。
        max_turns: 最多取最近 N 条消息。
        max_len: 单条内容截断长度，避免噪声与超长。

    Returns:
        多行文本；为空时返回 ``""``（调用方据此跳过拼接）。
    """
    if not history:
        return ""
    recent = history[-max_turns:]
    parts: list[str] = []
    for m in recent:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        # 助手答案只保留首句，避免把冗长回答的噪声带进上下文
        first_line = content.split("\n", 1)[0].strip()
        parts.append(f"{_role_label(role)}：{first_line[:max_len]}")
    return "\n".join(parts)


def compress_messages(messages: list, *, max_keep: int = 12) -> list:
    """超长对话时保留「首条 + 最新若干条」，丢弃中间（确定性，零依赖）。

    兼容两种元素形态：
    - LangChain ``BaseMessage``（有 ``.type`` / ``.content`` 属性）；
    - 普通 ``dict``（有 ``role`` / ``content`` 键）。

    返回的是与输入同形态的列表（不强制转换），便于直接回写 ``state["messages"]``。
    """
    if len(messages) <= max_keep:
        return messages
    head = messages[:1]
    keep_tail = max_keep - len(head)
    tail = messages[-keep_tail:]
    return head + tail


class MemoryStore:
    """按 session 维护对话历史，并对旧消息做确定性摘要压缩。

    适用场景：长会话（如多轮排障）下，完整 ``messages`` 会无限增长。本存储把
    较早的消息折叠进 ``summary`` 文本，仅保留最近 ``max_recent`` 条明细，
    从而在「有记忆」与「不爆 token 预算」之间取得平衡。
    """

    def __init__(self, max_recent: int = 6, max_summary_chars: int = 2000):
        self.max_recent = max_recent
        self.max_summary_chars = max_summary_chars
        self._sessions: dict[str, dict] = {}

    def _session(self, session_id: str) -> dict:
        s = self._sessions.get(session_id)
        if s is None:
            s = {"summary": "", "recent": []}
            self._sessions[session_id] = s
        return s

    def _fold(self, summary: str, msg: dict[str, str]) -> str:
        line = f"{_role_label(msg.get('role', ''))}：{(msg.get('content') or '')[:200]}"
        merged = f"{summary}\n{line}" if summary else line
        return merged[: self.max_summary_chars]

    def append(self, session_id: str, role: str, content: str) -> None:
        """追加一条消息；``recent`` 超过上限时把较早的一半折叠进 ``summary``。"""
        s = self._session(session_id)
        s["recent"].append({"role": role, "content": content})
        while len(s["recent"]) > self.max_recent:
            old = s["recent"].pop(0)
            s["summary"] = self._fold(s["summary"], old)

    def context(self, session_id: str) -> str:
        """返回该 session 的可注入上下文（历史摘要 + 近期明细）。"""
        s = self._session(session_id)
        if not s["summary"] and not s["recent"]:
            return ""
        parts: list[str] = []
        if s["summary"]:
            parts.append("=== 历史摘要 ===\n" + s["summary"])
        if s["recent"]:
            parts.append(
                "=== 近期对话 ===\n"
                + build_coref_context(s["recent"], max_turns=self.max_recent, max_len=200)
            )
        return "\n\n".join(parts)

    def get(self, session_id: str) -> dict:
        return dict(self._session(session_id))
