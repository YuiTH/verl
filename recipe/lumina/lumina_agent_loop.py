from datetime import datetime, timedelta

from azure.core.credentials import TokenCredential
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoProcessor
from azure.identity import DefaultAzureCredential

from recipe.lumina.cert_kv_helper import acquire_token_from_kv_cert
from verl.experimental.agent_loop import AgentLoopBase, AsyncLLMServerManager
from verl.experimental.agent_loop.agent_loop import _DummyConfig


class LuminaAgentLoop(AgentLoopBase):

    credential: TokenCredential
    config: DictConfig

    def __init__(
            self,
            trainer_config: _DummyConfig,
            server_manager: AsyncLLMServerManager,
            tokenizer: AutoTokenizer,
            processor: AutoProcessor,
            **kwargs,
    ):
        super().__init__(trainer_config, server_manager, tokenizer, processor, kwargs)
        self._token_expiry = None
        self._token = None

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer, **kwargs):
        if cls._class_initialized:
            return
        cls.credential = DefaultAzureCredential()
        cls.config = config


    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        model_path = self.config.actor_rollout_ref.model.path
        model_name = "/".join(model_path.split("/")[-2:])

        rollout = self.config.actor_rollout_ref.rollout
        model = ChatModel(
            model=model_name,
            client=self.server_manager,
            tokenizer=self.tokenizer,
            max_tokens=rollout.response_length,
            max_parallel_calls=rollout.multi_turn.max_parallel_calls,
            tool_parser=rollout.multi_turn.format,
        )

        model = model.bind_tools(self.tools, tool_choice="any")

        # Calculate recursion_limit dynamically based on max_assistant_turns
        max_assistant_turns = (
            rollout.multi_turn.max_assistant_turns
            if rollout.multi_turn.max_assistant_turns
            else self.DEFAULT_MAX_ASSISTANT_TURNS
        )

        # Formula: nodes_per_turn * max_turns * safety_buffer, with minimum threshold
        recursion_limit = max(
            self.MIN_RECURSION_LIMIT,
            int(max_assistant_turns * self.NODES_PER_TURN * self.RECURSION_LIMIT_SAFETY_FACTOR),
        )
        logger.info(f"Configured recursion_limit={recursion_limit} (max_assistant_turns={max_assistant_turns})")

        config = {
            "configurable": {
                "model": model,
                "sampling_params": sampling_params,
                "max_user_turns": rollout.multi_turn.max_user_turns,
                "max_assistant_turns": rollout.multi_turn.max_assistant_turns,
            },
            "recursion_limit": recursion_limit,
        }

        # TODO: how to handle multiple trajectories in an graph invocation?
        # Each graph node may has its own LLM calls and state, e.g:
        # https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart
        try:
            state = await self.graph.ainvoke(input={"messages": messages}, config=config)
        except Exception as e:
            logger.error(f"Agent loop execution failed: {type(e).__name__}: {e}")
            logger.error("Attempting to recover by extracting last valid trajectory.")

            # Strategy: Find the last valid AI message with complete metadata
            # This preserves any partial progress made before the error
            last_valid_ai_message = None
            for msg in reversed(messages):
                if (
                    msg.type == "ai"
                    and hasattr(msg, "response_metadata")
                    and "prompt_ids" in msg.response_metadata
                    and "response_mask" in msg.response_metadata
                ):
                    last_valid_ai_message = msg
                    break

            if last_valid_ai_message:
                logger.info("Recovered valid trajectory from existing messages.")
                state = {"messages": messages}
            else:
                logger.warning("No valid trajectory found. Creating minimal fallback.")
                fallback_message = AIMessage(
                    content="[Agent execution failed - no valid trajectory]",
                    response_metadata={
                        "request_id": "fallback",
                        "prompt_ids": [],
                        "response_mask": [],
                    },
                )
                state = {"messages": messages + [fallback_message]}

        output = convert_to_agent_output(state["messages"], rollout.response_length)
        return output
