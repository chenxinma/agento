import dataclasses
from typing import Generic, Optional

from pydantic_ai.agent import AgentRunResult
from pydantic_ai.result import StreamedRunResult
from pydantic_ai.tools import AgentDepsT

@dataclasses.dataclass(repr=False)
class HandleContext(Generic[AgentDepsT]):
    """Information about the current call."""

    deps: AgentDepsT
    run_result: Optional[AgentRunResult[str]] = None
    stream_result: Optional[StreamedRunResult[AgentDepsT, str]] = None

