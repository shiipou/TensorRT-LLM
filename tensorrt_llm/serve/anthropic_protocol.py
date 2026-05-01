# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pydantic models for the Anthropic Messages API (https://docs.anthropic.com/en/api/messages)."""

import json
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from tensorrt_llm.inputs.utils import ConversationMessage
from tensorrt_llm.sampling_params import SamplingParams


class AnthropicBaseModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


# ---------------------------------------------------------------------------
# Content block types (input)
# ---------------------------------------------------------------------------


class TextBlock(AnthropicBaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageSource(AnthropicBaseModel):
    type: Literal["base64", "url"]
    media_type: Optional[str] = None
    data: Optional[str] = None
    url: Optional[str] = None


class ImageBlock(AnthropicBaseModel):
    type: Literal["image"] = "image"
    source: ImageSource


class ToolUseBlock(AnthropicBaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(AnthropicBaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Union[str, List[TextBlock]] = ""
    is_error: Optional[bool] = None


# Ordered by frequency-of-use so Union resolution is fast
ContentBlock = Union[TextBlock, ToolResultBlock, ToolUseBlock, ImageBlock]

# System prompt can be a plain string or an array of text blocks (for cache
# control / multi-block prompts as per the API spec).
SystemTextBlock = TextBlock  # alias for clarity
SystemContent = Union[str, List[SystemTextBlock]]


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


class AnthropicMessage(AnthropicBaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[ContentBlock]]


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


class AnthropicTool(AnthropicBaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    # cache_control is accepted but not acted upon
    cache_control: Optional[Dict[str, str]] = None


class ToolChoiceAuto(AnthropicBaseModel):
    type: Literal["auto"] = "auto"
    disable_parallel_tool_use: Optional[bool] = None


class ToolChoiceAny(AnthropicBaseModel):
    type: Literal["any"] = "any"
    disable_parallel_tool_use: Optional[bool] = None


class ToolChoiceSpecific(AnthropicBaseModel):
    type: Literal["tool"] = "tool"
    name: str
    disable_parallel_tool_use: Optional[bool] = None


class ToolChoiceNone(AnthropicBaseModel):
    type: Literal["none"] = "none"


AnthropicToolChoice = Union[ToolChoiceAuto, ToolChoiceAny, ToolChoiceSpecific, ToolChoiceNone]


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class AnthropicMetadata(AnthropicBaseModel):
    user_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

AnthropicStopReason = Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]


class MessagesRequest(AnthropicBaseModel):
    model: str
    messages: List[AnthropicMessage]
    max_tokens: int
    system: Optional[SystemContent] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[AnthropicToolChoice] = None
    metadata: Optional[AnthropicMetadata] = None

    def get_system_text(self) -> Optional[str]:
        if self.system is None:
            return None
        if isinstance(self.system, str):
            return self.system
        return "\n".join(b.text for b in self.system)

    def to_conversation(self) -> List[ConversationMessage]:
        """Convert Anthropic messages to the internal ConversationMessage list.

        System prompt is prepended as a system-role message so that
        apply_chat_template can inject it via the normal path.
        """
        conversation: List[ConversationMessage] = []
        system_text = self.get_system_text()
        if system_text:
            conversation.append(ConversationMessage(role="system", content=system_text))

        for msg in self.messages:
            role = msg.role
            content = msg.content
            if isinstance(content, str):
                conversation.append(ConversationMessage(role=role, content=content))
            else:
                # Flatten content blocks into a plain text string.
                # Tool results are included as plain text so the model can
                # observe them via the standard chat template.
                parts: List[str] = []
                for block in content:
                    if isinstance(block, TextBlock):
                        parts.append(block.text)
                    elif isinstance(block, ToolResultBlock):
                        result = block.content
                        if isinstance(result, str):
                            parts.append(result)
                        else:
                            parts.append("\n".join(b.text for b in result))
                    elif isinstance(block, ToolUseBlock):
                        parts.append(
                            f"[Tool call: {block.name}({json.dumps(block.input)})]"
                        )
                    # ImageBlock: silently ignored (model must have vision support)
                conversation.append(ConversationMessage(role=role, content="\n".join(parts)))

        return conversation

    def to_openai_tools(self) -> Optional[List[Dict[str, Any]]]:
        """Convert Anthropic tool definitions to the OpenAI function-call format
        understood by apply_chat_template."""
        if not self.tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.input_schema,
                },
            }
            for t in self.tools
        ]

    def to_sampling_params(self) -> SamplingParams:
        return SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            stop=self.stop_sequences or [],
        )


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


class AnthropicUsage(AnthropicBaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


# ---------------------------------------------------------------------------
# Non-streaming response
# ---------------------------------------------------------------------------

OutputContentBlock = Union[TextBlock, ToolUseBlock]


class MessagesResponse(AnthropicBaseModel):
    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:24]}")
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[OutputContentBlock] = Field(default_factory=list)
    model: str
    stop_reason: Optional[AnthropicStopReason] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage = Field(default_factory=AnthropicUsage)


# ---------------------------------------------------------------------------
# Error response
# ---------------------------------------------------------------------------


class AnthropicError(AnthropicBaseModel):
    type: str
    message: str


class AnthropicErrorResponse(AnthropicBaseModel):
    type: Literal["error"] = "error"
    error: AnthropicError


# ---------------------------------------------------------------------------
# Streaming SSE event models
# ---------------------------------------------------------------------------


class MessageStartPayload(AnthropicBaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List = Field(default_factory=list)
    model: str
    stop_reason: None = None
    stop_sequence: None = None
    usage: AnthropicUsage


class MessageStartEvent(AnthropicBaseModel):
    type: Literal["message_start"] = "message_start"
    message: MessageStartPayload


class ContentBlockStartEvent(AnthropicBaseModel):
    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: Union[TextBlock, ToolUseBlock]


class PingEvent(AnthropicBaseModel):
    type: Literal["ping"] = "ping"


class TextDelta(AnthropicBaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class InputJsonDelta(AnthropicBaseModel):
    type: Literal["input_json_delta"] = "input_json_delta"
    partial_json: str


ContentBlockDelta = Union[TextDelta, InputJsonDelta]


class ContentBlockDeltaEvent(AnthropicBaseModel):
    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: ContentBlockDelta


class ContentBlockStopEvent(AnthropicBaseModel):
    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageDeltaDetails(AnthropicBaseModel):
    stop_reason: Optional[AnthropicStopReason] = None
    stop_sequence: Optional[str] = None


class MessageDeltaUsage(AnthropicBaseModel):
    output_tokens: int


class MessageDeltaEvent(AnthropicBaseModel):
    type: Literal["message_delta"] = "message_delta"
    delta: MessageDeltaDetails
    usage: MessageDeltaUsage


class MessageStopEvent(AnthropicBaseModel):
    type: Literal["message_stop"] = "message_stop"


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------


def sse_event(event_type: str, data: BaseModel) -> str:
    """Format a single Anthropic SSE event as a string ready to yield."""
    return f"event: {event_type}\ndata: {data.model_dump_json(exclude_none=True)}\n\n"


# ---------------------------------------------------------------------------
# Finish-reason mapping: TRT-LLM → Anthropic
# ---------------------------------------------------------------------------

FINISH_REASON_MAP: Dict[Optional[str], AnthropicStopReason] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


def map_finish_reason(
    finish_reason: Optional[str],
    stop_reason: Optional[Union[int, str]] = None,
) -> tuple[AnthropicStopReason, Optional[str]]:
    """Return ``(anthropic_stop_reason, stop_sequence_str)``."""
    if finish_reason == "stop" and isinstance(stop_reason, str):
        return "stop_sequence", stop_reason
    return FINISH_REASON_MAP.get(finish_reason, "end_turn"), None
