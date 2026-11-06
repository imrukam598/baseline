from agentscope.formatter import FormatterBase
from agentscope.message import (
    Msg,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
)
from agentscope.model import ChatModelBase
from agentscope.tool import Toolkit, ToolResponse

from mini_agemem.prompts import (
    SUMMARY_CONTEXT_PROMPT,
    FILTER_CONTEXT_PROMPT,
)

class SimpleMemoryAgent:
    """具有记忆工具和对话历史的简化 Agent。"""

    finish_function_name = "generate_response"

    def __init__(
            self,
            name: str,
            system_prompt: str,
            model: ChatModelBase,
            formatter: FormatterBase,
            toolkit: Toolkit,
            max_steps: int = 4,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.formatter = formatter
        self.toolkit = toolkit
        self.max_steps = max_steps

        self.context_messages = [
            Msg(
                name="system",
                content=system_prompt,
                role="system",
            ),
        ]

    async def reply(
            self,
            user_text: str,
    ) -> str:
        """接收用户文本并生成最终回答。"""

        user_message = Msg(
            name="user",
            content=user_text,
            role="user",
        )

        self.context_messages.append(
            user_message,
        )

        for step in range(self.max_steps):
            print(
                f"{self.name} 第 {step + 1} 步"
            )

            formatted_messages = (
                await self.formatter.format(
                    msgs=self.context_messages,
                )
            )

            model_response = await self.model(
                formatted_messages,
                tools = self.toolkit.get_json_schemas(),
                tool_choice = "auto",
            ) 

            assistant_message = Msg(
                name=self.name,
                content=list(model_response.content),
                role="assistant",
            )

            self.context_messages.append(
                assistant_message,
            )

            tool_calls = (
                assistant_message.get_content_blocks(
                    "tool_use",
                )
            )

            if not tool_calls:
                final_text = (
                    assistant_message.get_text_content()
                )

                assert final_text is not None
                return final_text

            for tool_call in tool_calls:
                await self._apply_tool(
                    tool_call,
                )

        raise RuntimeError(
            f"{self.name} 在 {self.max_steps} 步内"
            "没有生成最终回答"
        )

    async def summary_context(
            self,
            span: str = "all",
    ) -> ToolResponse:
        """压缩当前短期上下文。
        
         Args:
            span:
                "all" 表示压缩全部旧消息；
                正整数字符串表示压缩最近 N 条旧消息。
        """

        tool_call_index = len(
            self.context_messages
        ) - 1

        candidate_indices = [
            index 
            for index, message in enumerate(
                self.context_messages[
                    :tool_call_index
                ]
            )
            if message.role != "system"
        ]
        if span == "all":
            selected_indices = candidate_indices
        else:      
            count = int(span)
            selected_indices = (
                candidate_indices[-count:]
            )

        if not selected_indices:
            return ToolResponse(
                content = [
                    TextBlock(
                        type = "text",
                        text = "当前没有可以压缩的旧消息。",
                    ),
                ],
                metadata = {
                    "success": False,
                    "span": span,
                },
            )

        selected_messages = [
            self.context_messages[index]
            for index in selected_indices
        ]         

        conversation_text = "\n".join(
            (
                f"{message.role}:"
                f"{message.get_text_content()}"
            )
            for message in selected_messages
        )

        summary_prompt = (
            SUMMARY_CONTEXT_PROMPT.format(
                conversation_text = (
                    conversation_text
                ),
            )
        )

        summary_messages = [
            Msg(
                name = "system",
                content = "你负责压缩对话上下文。",
                role="system",
            ),
            Msg(
                name = "user",
                content=summary_prompt,
                role = "user",
            ),
        ]

        formatted_summary_messages = (
            await self.formatter.format(
                msgs = summary_messages,
            )
        )

        summary_response = await self.model(
            formatted_summary_messages,
        )

        summary_message = Msg(
            name = self.name,
            content = list(
                summary_response.content
            ),
            role = "assistant",
        )

        summary_text = (
            summary_message.get_text_content()
        )

        assert summary_text is not None
        
        selected_index_set = set(
            selected_indices
        )

        self.context_messages = [
            message
            for index, message in enumerate(
                self.context_messages
            )
            if index not in selected_index_set
        ]

        return ToolResponse(
            content = [
                TextBlock(
                    type = "text",
                    text = (
                        f"已经压缩"
                        f"{len(selected_messages)}"
                        f"条上下文消息。"
                        f"摘要:{summary_text}"
                    ),
                ),
            ],
            metadata = {
                "success": True,
                "span": span,
                "summary": summary_text,
            },
        )

    async def filter_context(
            self,
            criteria: str,
    ) -> ToolResponse:
        """
        删除符合指定条件的短期上下文消息。

        Args:
            criteria:
                对需要删除的内容的描述。
        """

        current_tool_call_index = (
            len(self.context_messages) - 1
        )

        filtered_messages = []
        removed_count = 0
        threshold = 0.1

        for index, message in enumerate(
            self.context_messages
        ):
            if message.role == "system":
                filtered_messages.append(
                    message
                )
                continue

            if index == current_tool_call_index:
                filtered_messages.append(
                    message
                )
                continue

            message_text = (
                message.get_text_content()
            )

            if not message_text:
                filtered_messages.append(
                    message
                )
                continue

            similarity_prompt = (
                FILTER_CONTEXT_PROMPT.format(
                    criteria =criteria,
                    message_text = message_text,
                )
            )

            similarity_messages = [
                Msg(
                    name = "user",
                    content = similarity_prompt,
                    role = "user",
                ),
            ]

            formatted_messages = (
                await self.formatter.format(
                    msgs= similarity_messages,
                )
            )

            score_response = await self.model(
                formatted_messages,
            )

            score_message = Msg(
                name = self.name,
                content = list(
                    score_response.content
                ),
                role = "assistant",
            )

            score_text = (
                score_message.get_text_content()
            )

            assert score_text is not None

            similarity_score = float(
                score_text.strip()
            )

            print(
                "检查上下文消息 =",
                message_text,
            )

            print(
                "删除相关度 =",
                similarity_score,
            )

            if similarity_score >= threshold:
                removed_count += 1
                continue

            filtered_messages.append(
                message
            )

        self.context_messages = (
            filtered_messages
        )

        return ToolResponse(
            content = [
                TextBlock(
                    type = "text",
                    text = (
                        f"已经从短期上下文中删除"
                        f"{removed_count}条消息。"
                    ),
                ),
            ],
            metadata = {
                "success": True,
                "criteria": criteria,
                "removed_count": removed_count,
            },
        )
    
    async def _apply_tool(
            self,
            tool_call: ToolUseBlock,
    ) -> None:
        """执行一次工具调用，并把结果加入对话历史。"""

        print(
            "模型调用工具 =",
            tool_call["name"],
        )

        print(
            "工具参数 =",
            tool_call["input"],
        )

        response_stream = (
            await self.toolkit.call_tool_function(
                tool_call,
            )
        )

        tool_response = None

        async for response_chunk in response_stream:
            tool_response = response_chunk

        assert tool_response is not None

        print(
            "工具执行结果 =",
            tool_response.content,
        )

        tool_result_message = Msg(
            name="system",
            content=[
                ToolResultBlock(
                    type="tool_result",
                    id=tool_call["id"],
                    name=tool_call["name"],
                    output=tool_response.content,
                ),
            ],
            role="system",
        )

        self.context_messages.append(
            tool_result_message,
        )