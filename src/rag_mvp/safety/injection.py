"""Small, intent-aware prompt-injection policy for user and evidence text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class InjectionAction(StrEnum):
    """Action the caller must take for assessed content."""

    ALLOW = "allow"
    REFUSE = "refuse"
    TREAT_AS_DATA = "treat_as_data"


@dataclass(frozen=True, slots=True)
class InjectionAssessment:
    """Content-free prompt-injection policy decision."""

    action: InjectionAction
    reason_code: str | None = None
    matched_rules: tuple[str, ...] = ()

    @property
    def allowed(self) -> bool:
        """Whether normal grounded QA may continue."""

        return self.action is not InjectionAction.REFUSE

    @property
    def requires_refusal(self) -> bool:
        """Whether the user request must receive a policy refusal."""

        return self.action is InjectionAction.REFUSE


_QUOTED_SEGMENTS: Final[re.Pattern[str]] = re.compile(
    r'"[^"\r\n]*"|\'[^\'\r\n]*\'|`[^`\r\n]*`|“[^”\r\n]*”|'
    r"\u2018[^\u2019\r\n]*\u2019|「[^」\r\n]*」|『[^』\r\n]*』"
)

_USER_RULES: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "override_policy",
        re.compile(
            r"(?is)(?:ignore|disregard|override|bypass|disable|turn\s+off|do\s+not\s+follow)"
            r".{0,80}(?:previous|higher[- ]priority|system|developer|safety|policy|instruction|"
            r"safeguards?|grounding|citation|redaction|privacy)"
            r"|(?:忽略|无视|绕过|覆盖|关闭|禁用|不要遵循).{0,40}"
            r"(?:之前|上级|系统|开发者|安全|策略|规则|指令|引用|依据|脱敏|隐私)",
        ),
    ),
    (
        "ungrounded_output",
        re.compile(
            r"(?is)(?:answer|respond|reply).{0,50}(?:without|no).{0,30}"
            r"(?:grounding|evidence|citations?|privacy checks?|redaction)"
            r"|(?:不要|无需|关闭|跳过).{0,30}(?:引用|依据|证据|脱敏|隐私检查).{0,20}(?:回答|输出)?",
        ),
    ),
    (
        "hidden_context_disclosure",
        re.compile(
            r"(?is)(?:reveal|show\s+me|print|display|disclose|repeat|give\s+me)"
            r".{0,60}(?:system prompt|developer message|hidden (?:prompt|context|instruction)|"
            r"chain of thought|internal instruction)"
            r"|(?:泄露|展示|显示|输出|告诉我|重复).{0,40}"
            r"(?:系统提示|开发者消息|隐藏(?:提示|上下文|指令)|思维链|内部指令)",
        ),
    ),
    (
        "secret_disclosure",
        re.compile(
            r"(?is)(?:reveal|show\s+me|print|display|disclose|give\s+me|extract)"
            r".{0,60}(?:secrets?|credentials?|api keys?|tokens?|passwords?)"
            r"|(?:泄露|展示|显示|输出|告诉我|提取).{0,40}"
            r"(?:密钥|凭据|令牌|密码|API[ _-]?key)",
        ),
    ),
    (
        "cross_session_access",
        re.compile(
            r"(?is)(?:access|show|read|retrieve|list).{0,60}"
            r"(?:another|other|someone else(?:'s)?).{0,30}(?:session|conversation|chat|history)"
            r"|(?:访问|显示|读取|列出|检索).{0,40}(?:其他|别人|他人).{0,20}"
            r"(?:会话|对话|聊天|历史)",
        ),
    ),
)

_RETRIEVED_DIRECTIVES: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    *_USER_RULES,
    (
        "command_execution",
        re.compile(
            r"(?is)(?:execute|run|launch|invoke).{0,30}(?:command|shell|terminal|tool|function)"
            r"|(?:执行|运行|调用|启动).{0,20}(?:命令|脚本|终端|工具|函数)"
            r"|\b(?:curl|wget)\s+(?:https?://|[-/])"
            r"|\b(?:powershell|bash|sh)\s+[-/]",
        ),
    ),
    (
        "external_access",
        re.compile(
            r"(?is)(?:open|visit|fetch|request|browse|download|upload|send|post)"
            r".{0,30}https?://"
            r"|(?:打开|访问|请求|浏览|下载|上传|发送).{0,20}https?://",
        ),
    ),
)


def _without_quoted_segments(text: str) -> str:
    return _QUOTED_SEGMENTS.sub(lambda match: " " * len(match.group(0)), text)


class InjectionPolicy:
    """Classify override intent without refusing merely quoted trigger phrases."""

    def assess_user_input(self, text: object) -> InjectionAssessment:
        """Refuse direct attempts to disable controls or expose hidden state."""

        if not isinstance(text, str):
            return InjectionAssessment(
                InjectionAction.REFUSE,
                reason_code="invalid_user_input",
                matched_rules=("invalid_input",),
            )
        actionable_text = _without_quoted_segments(text)
        matched = tuple(name for name, pattern in _USER_RULES if pattern.search(actionable_text))
        if matched:
            return InjectionAssessment(
                InjectionAction.REFUSE,
                reason_code="prompt_injection_override",
                matched_rules=matched,
            )
        return InjectionAssessment(InjectionAction.ALLOW)

    def assess_retrieved_content(self, text: object) -> InjectionAssessment:
        """Mark retrieved directives as inert data rather than executing them."""

        if not isinstance(text, str):
            return InjectionAssessment(
                InjectionAction.TREAT_AS_DATA,
                reason_code="invalid_retrieved_content",
                matched_rules=("invalid_input",),
            )
        matched = tuple(name for name, pattern in _RETRIEVED_DIRECTIVES if pattern.search(text))
        if matched:
            return InjectionAssessment(
                InjectionAction.TREAT_AS_DATA,
                reason_code="untrusted_retrieved_instruction",
                matched_rules=matched,
            )
        return InjectionAssessment(
            InjectionAction.TREAT_AS_DATA,
            reason_code="untrusted_retrieved_content",
        )


DEFAULT_INJECTION_POLICY = InjectionPolicy()


def check_user_input(
    text: str, *, policy: InjectionPolicy = DEFAULT_INJECTION_POLICY
) -> InjectionAssessment:
    """Assess a user question with the default prompt-injection policy."""

    return policy.assess_user_input(text)


def check_retrieved_content(
    text: str, *, policy: InjectionPolicy = DEFAULT_INJECTION_POLICY
) -> InjectionAssessment:
    """Assess retrieved evidence with the default prompt-injection policy."""

    return policy.assess_retrieved_content(text)
