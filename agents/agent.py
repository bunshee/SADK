from agents.client import AgenticOllamaClient


class Agent(AgenticOllamaClient):
    def __init__(
        self,
        name: str,
        description: str,
        model: str,
        system_prompt: str | None = None,
        ollama_uri: str = "http://localhost:11434/api",
        tools: list[callable] | None = None,
        **ollama_client_kwargs: object,
    ) -> None:
        super().__init__(
            base_url=ollama_uri,
            model_name=model,
            system_prompt=system_prompt,
            **ollama_client_kwargs,
        )

        self.name: str = name
        self.description: str = description
        self.model: str = model
        self.system_prompt: str = system_prompt

        self.guardrails: list[object] = []

        if tools:
            self.register_tools(tools)
            print(f"Agent '{self.name}' registered {len(tools)} tools.")
        else:
            self.tools = {}

    def invoke(self, user_input: str) -> dict[str, object]:
        """
        Invokes the agent to generate a non-streaming chat response,
        potentially using orchestration if tools are registered.
        """
        print(f"\n--- Agent '{self.name}' Invoked ---")
        if self.tools:
            return self.chat_with_orchestration(
                user_prompt=user_input,
                model_name=self.default_model_name,
                additional_instructions=self.default_system_prompt,
                max_iterations=self.default_max_iterations,
                **self.default_chat_kwargs,
            )
        else:
            # When no tools are available, revert to a basic chat.
            # In this case, we use the default model name and system prompt from init.
            return self.chat(
                user_prompt=user_input,
                model_name=self.default_model_name,
                system_prompt=self.default_system_prompt,
                **self.default_chat_kwargs,
            )

    def stream(self, user_input: str) -> None:
        """
        Invokes the agent to generate a streaming chat response.
        Note: Streaming with orchestration for tool calls can be complex
        as tool outputs are discrete. This method will currently only
        stream the model's natural language response.
        """
        print(f"\n--- Agent '{self.name}' Streaming Response ---")
        # When streaming, for simplicity, we bypass orchestration and
        # directly call the basic chat with the default model and system prompt.
        self.chat(
            user_prompt=user_input,
            model_name=self.default_model_name,
            system_prompt=self.default_system_prompt,
            stream=True,
            **self.default_chat_kwargs,
        )

    def __call__(
        self, user_input: str, stream: bool = False
    ) -> dict[str, object] | None:
        """
        Allows the agent instance to be called directly like a function.
        If stream is True, returns None as output is printed directly.
        """
        if stream:
            self.stream(user_input)
            return None
        else:
            return self.invoke(user_input)
