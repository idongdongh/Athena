"""CLI 创建和切换 ``AIAgent`` 的生命周期逻辑。"""

from __future__ import annotations

from agent.context_state import ContextSettings
from agent.tool_guardrails import ToolCallGuardrailConfig
from athena_cli.config import MemorySettings
from run_agent import AIAgent
from session_db import SessionDB


class CLIAgentSetupMixin:
    """Agent 生命周期方法，以及对宿主 CLI 属性的类型约定。"""

    model: str
    system_prompt: str
    context_settings: ContextSettings
    _session_db: SessionDB | None
    api_key: str | None
    base_url: str | None
    tool_guardrail_config: ToolCallGuardrailConfig
    memory_settings: MemorySettings
    agent: AIAgent

    def _create_agent(self) -> AIAgent:
        return AIAgent(
            model=self.model,
            system_prompt=self.system_prompt,
            context_settings=self.context_settings,
            session_db=self._session_db,
            model_config={"max_output_tokens": self.context_settings.max_output_tokens},
            api_key=self.api_key,
            base_url=self.base_url,
            tool_guardrail_config=self.tool_guardrail_config,
            memory_settings=self.memory_settings,
        )

    def _resume_agent(
        self,
        session_id: str,
    ) -> tuple[AIAgent, list[dict]]:
        if self._session_db is None:
            raise RuntimeError("会话数据库不可用")
        agent, messages = AIAgent.resume(
            self._session_db,
            session_id,
            context_settings=self.context_settings,
            client=self.agent.client,
            tool_guardrail_config=self.tool_guardrail_config,
            system_prompt=self.system_prompt,
            memory_settings=self.memory_settings,
        )
        # 当前进程的模型与稳定 system prompt 由本次启动配置决定。
        agent.model = self.model
        agent.reset_session_state(messages)
        return agent, messages
