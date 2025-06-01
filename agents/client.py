import json

import requests


class OllamaClient:
    """
    A client for interacting with the Ollama API's /api/chat endpoint.
    It handles sending multi-role messages.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        system_prompt: str | None = None,
        **kwargs,  # For other default chat parameters
    ):
        self.base_url = base_url
        self.chat_endpoint = f"{self.base_url}/chat"
        self.default_model_name = model_name
        self.default_system_prompt = system_prompt
        self.default_chat_kwargs = kwargs  # Store other default chat parameters

    def _make_request(self, method: str, url: str, **kwargs) -> dict[str, object]:
        """Internal helper for HTTP requests and error handling."""
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            result = response.json()

            if "error" in result:
                raise ValueError(f"Ollama API Error: {result['error']}")
            return result
        except requests.exceptions.ConnectionError as e:
            raise requests.exceptions.ConnectionError(
                f"Error: Could not connect to Ollama at {self.base_url}. "
                "Please ensure Ollama is installed and running, and the API is accessible."
            ) from e
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(
                f"An error occurred during the API request: {e}"
            ) from e
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Error: Could not decode JSON response from Ollama. Raw response: {response.text}"
            ) from e

    def chat(
        self,
        user_prompt: str,
        model_name: str,  # Explicitly expect model_name
        system_prompt: str,  # Explicitly expect system_prompt
        stream: bool = False,
        **kwargs,
    ) -> str | dict[str, object]:
        """
        Sends a list of messages to the Ollama /api/chat endpoint and returns the model's response message.
        """
        data = {
            "model": model_name,  # Use the passed model_name
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },  # Use the passed system_prompt
                {"role": "user", "content": user_prompt},
            ],
            "stream": stream,
            **kwargs,  # Pass additional kwargs from the caller
        }

        if stream:
            full_response_content = ""
            final_message_role = "assistant"

            try:
                with requests.post(
                    self.chat_endpoint, json=data, stream=True
                ) as response:
                    response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=None):
                        if chunk:
                            try:
                                json_chunk = json.loads(chunk)
                                if "message" in json_chunk:
                                    message_part = json_chunk["message"]
                                    if "content" in message_part:
                                        print(
                                            message_part["content"], end="", flush=True
                                        )
                                        full_response_content += message_part["content"]
                                    if "role" in message_part:
                                        final_message_role = message_part["role"]
                                elif "error" in json_chunk:
                                    raise ValueError(
                                        f"Ollama API Error during stream: {json_chunk['error']}"
                                    )
                            except json.JSONDecodeError:
                                continue

            except requests.exceptions.ConnectionError as e:
                raise requests.exceptions.ConnectionError(
                    f"Error: Could not connect to Ollama at {self.base_url} during streaming. "
                    "Please ensure Ollama is installed and running, and the API is accessible."
                ) from e
            except requests.exceptions.RequestException as e:
                raise requests.exceptions.RequestException(
                    f"An error occurred during the API request: {e}"
                ) from e
            print()
            return {
                "role": final_message_role,
                "content": full_response_content,
            }
        else:
            result = self._make_request("POST", self.chat_endpoint, json=data)
            if "message" in result:
                return result["message"]
            else:
                raise ValueError(
                    f"Unexpected response format from Ollama chat API: {result}"
                )


class AgenticOllamaClient(OllamaClient):
    """
    A client for interacting with the Ollama API's /api/chat endpoint,
    with added capabilities for tool integration and automated orchestration.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        system_prompt: str | None = None,
        max_iterations: int = 3,
        **kwargs,
    ):
        super().__init__(base_url, model_name, system_prompt, **kwargs)
        self.tools: dict[str, callable] = {}
        self.tool_schemas: dict[str, dict[str, object]] = {}
        self.default_max_iterations = max_iterations

    def register_tool(self, func: callable, schema: dict[str, object]):
        """Registers a single tool function and its schema."""
        tool_name = schema.get("name")
        if not tool_name:
            raise ValueError("Tool schema must contain a 'name' field.")
        if tool_name in self.tools:
            print(
                f"Warning: Tool '{tool_name}' is already registered and will be overwritten."
            )

        self.tools[tool_name] = func
        self.tool_schemas[tool_name] = schema
        print(f"Tool '{tool_name}' registered successfully.")

    def register_tools(self, tool_functions: list[callable]):
        """Registers multiple tool functions."""
        for func in tool_functions:
            if not hasattr(func, "_tool_schema"):
                raise ValueError(
                    f"Function '{func.__name__}' is not decorated with @tool "
                    "or its schema is missing. Please use the @tool decorator."
                )
            self.register_tool(func, func._tool_schema)
        print("All specified tools registered.")

    def _format_tools_for_prompt(self) -> str:
        """Formats the registered tool schemas into a JSON string for system prompts."""
        if not self.tool_schemas:
            return ""
        return json.dumps(list(self.tool_schemas.values()), indent=2)

    def _parse_tool_call(
        self, model_response_message: dict[str, object]
    ) -> list[dict[str, object]] | None:
        """
        Parses tool calls from a model response message.
        Prioritizes Ollama's native 'tool_calls' structure.
        """
        tool_calls = []

        if "tool_calls" in model_response_message and isinstance(
            model_response_message["tool_calls"], list
        ):
            for tc in model_response_message["tool_calls"]:
                if (
                    isinstance(tc, dict)
                    and "function" in tc
                    and isinstance(tc["function"], dict)
                ):
                    function_data = tc["function"]
                    if "name" in function_data and "arguments" in function_data:
                        tool_calls.append(
                            {
                                "name": function_data["name"],
                                "arguments": function_data["arguments"],
                                "id": tc.get("id", f"generated_id_{len(tool_calls)}"),
                            }
                        )
            if tool_calls:
                return tool_calls

        # Fallback to parsing JSON from content for older models/less precise output
        content = model_response_message.get("content", "").strip()
        if content.startswith("{") and content.endswith("}"):
            try:
                parsed_json = json.loads(content)
                if "tool_name" in parsed_json and "arguments" in parsed_json:
                    tool_calls.append(
                        {
                            "name": parsed_json["tool_name"],
                            "arguments": parsed_json["arguments"],
                            "id": f"content_call_{hash(content)}",  # Simple unique ID
                        }
                    )
                    return tool_calls
            except json.JSONDecodeError:
                pass

        return None

    def _execute_tool(self, tool_call: dict[str, object]) -> object:
        """Executes a registered tool based on a parsed tool call dictionary."""
        tool_name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})

        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not registered.")

        tool_func = self.tools[tool_name]
        print(f"\n[Tool Call] Executing '{tool_name}' with arguments: {arguments}")
        try:
            return tool_func(**arguments)
        except TypeError as e:
            raise ValueError(
                f"Error executing tool '{tool_name}': Mismatched arguments. {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Error during tool '{tool_name}' execution: {e}")

    def chat_with_orchestration(
        self,
        user_prompt: str,
        model_name: str,
        additional_instructions: str,
        max_iterations: int,
        **kwargs,
    ) -> str:
        """
        Orchestrates a multi-turn conversation with the model, including tool calls.
        """
        messages: list[dict[str, str]] = []

        tool_definitions_json = self._format_tools_for_prompt()
        # Construct the system prompt that includes tool definitions
        system_prompt_for_orchestration = f"""
You have access to the following tools {additional_instructions}:
{tool_definitions_json}

If question is irrelevant to provided tool descriptions then answer without using any tool and respond with plain text following this format:
`As of my base knowledge ...`
Or else when a user asks a question that can be answered by one of the tools, respond with a JSON object in the following format:
{{
    "tool_name": "<name_of_the_tool>",
    "arguments": {{
        "<param1>": "<value1>",
        "<param2>": "<value2>"
    }}
}}
Do NOT output anything else if you are calling a tool.
"""
        messages.append(
            {"role": "system", "content": system_prompt_for_orchestration.strip()}
        )
        messages.append({"role": "user", "content": user_prompt})

        for i in range(max_iterations):
            print(f"\n--- Orchestration Iteration {i + 1} ---")
            print(
                f"Sending messages ({len(messages)} total). Last message: {messages[-1].get('content', '')[:100]}..."
            )

            # CORRECTED: Explicitly pass model_name and the constructed system_prompt
            model_response_message = super().chat(
                user_prompt=user_prompt,
                model_name=model_name,  # Pass the model_name from chat_with_orchestration args
                system_prompt=system_prompt_for_orchestration,  # Pass the specific system prompt for orchestration
                stream=kwargs.get("stream", False),
                **kwargs,
            )
            messages.append(model_response_message)

            model_response_content = model_response_message.get("content", "").strip()
            parsed_tool_calls = self._parse_tool_call(model_response_message)

            if parsed_tool_calls:
                print(f"Model requested tool calls: {parsed_tool_calls}")
                for tool_call in parsed_tool_calls:
                    try:
                        tool_output = self._execute_tool(tool_call)
                        tool_output_str = (
                            json.dumps(tool_output)
                            if not isinstance(tool_output, str)
                            else str(tool_output)
                        )
                        print(
                            f"Tool output for '{tool_call['name']}': {tool_output_str}"
                        )
                        return {
                            "content": tool_output_str,
                            "tool_calling": True,
                            "tool_name": tool_call["name"],
                        }

                    except Exception as e:
                        print(
                            f"Error executing tool '{tool_call.get('name', '')}': {e}"
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "content": json.dumps(
                                    {
                                        "error": f"Tool execution failed for {tool_call.get('name')}: {e}"
                                    }
                                ),
                                "tool_call_id": tool_call["id"],
                            }
                        )
        # If the loop finishes, it means max_iterations was reached without a final answer
        print(
            f"Max iterations ({max_iterations}) reached without a final natural language answer. Last response: {model_response_content}"
        )
        return {
            "content": model_response_content,
            "tool_calling": False,
            "tool_name": "Base knowledge",
        }
