import json

import requests


class Client:
    """
    A client for interacting with a Large Language Model (LLM) API.
    This client is designed to be model-agnostic, supporting various providers
    through a common interface. It handles API key authentication and makes HTTP
    requests to the specified base URL.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        system_prompt: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ):
        """
        Initializes the client with the necessary API information.

        Args:
            base_url (str): The base URL of the LLM API.
            model_name (str): The default model to use for requests.
            system_prompt (str | None): A default system prompt to prepend to messages.
            api_key (str | None): The API key for authentication.
            **kwargs: Additional default parameters for chat requests.
        """
        self.base_url = base_url
        self.chat_endpoint = self.base_url
        self.default_model_name = model_name
        self.default_system_prompt = system_prompt
        self.api_key = api_key
        self.default_chat_kwargs = kwargs

    def _make_request(self, method: str, url: str, **kwargs) -> dict[str, object]:
        """
        Internal helper for making HTTP requests with authentication and error handling.

        Args:
            method (str): The HTTP method (e.g., 'POST', 'GET').
            url (str): The full URL for the request.
            **kwargs: Additional arguments for the request (e.g., json, headers).

        Returns:
            dict[str, object]: The JSON response from the API.

        Raises:
            requests.exceptions.ConnectionError: If the API is unreachable.
            requests.exceptions.RequestException: For other request-related errors.
            ValueError: If the API returns an error in the response.
            json.JSONDecodeError: If the response is not valid JSON.
        """
        headers = kwargs.get("headers", {})
        kwargs["headers"] = headers

        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            result = response.json()

            if "error" in result:
                raise ValueError(f"LLM API Error: {result['error']}")
            return result
        except requests.exceptions.ConnectionError as e:
            raise requests.exceptions.ConnectionError(
                f"Error: Could not connect to LLM at {self.base_url}. "
                "Please ensure the service is running and the API is accessible."
            ) from e
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(
                f"An error occurred during the API request: {e}"
            ) from e
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Error: Could not decode JSON response from LLM. Raw response: {response.text}"
            ) from e

    def chat(
        self,
        user_prompt: str,
        model_name: str,
        system_prompt: str,
        stream: bool = False,
        **kwargs,
    ) -> dict[str, object]:
        """
        Sends a chat request to a compatible LLM API (OpenAI, Google Gemini, Ollama)
        and returns the response in a standardized format.
        """
        is_google_api = "googleapis.com" in self.base_url
        headers = {"Content-Type": "application/json"}
        data = {}
        url = ""

        if is_google_api:
            # --- Google Gemini API Handling ---
            url = f"{self.base_url}/{model_name}:generateContent"
            if self.api_key:
                url += f"?key={self.api_key}"

            full_prompt = (
                f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
            )
            data = {"contents": [{"parts": [{"text": full_prompt}]}]}

            if stream:
                print(
                    "Warning: Streaming is not currently supported for the Google Gemini API in this client."
                )

            result = self._make_request("POST", url, json=data, headers=headers)

            if "candidates" in result and result.get("candidates"):
                content = result["candidates"][0].get("content", {})
                if "parts" in content and content.get("parts"):
                    return {
                        "role": "assistant",
                        "content": content["parts"][0].get("text", ""),
                    }
            raise ValueError(
                f"Unexpected response format from Google Gemini API: {result}"
            )

        else:
            # --- OpenAI, GitHub, and Ollama API Handling ---
            url = self.chat_endpoint
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            data = {
                "model": model_name,
                "messages": messages,
                "stream": stream,
                **kwargs,
            }

            if stream:
                full_response_content = ""
                final_message_role = "assistant"
                try:
                    with requests.post(
                        url, json=data, headers=headers, stream=True
                    ) as response:
                        response.raise_for_status()
                        for chunk in response.iter_lines():
                            if chunk:
                                chunk_str = chunk.decode("utf-8")
                                if chunk_str.startswith("data: "):
                                    chunk_str = chunk_str[6:]
                                if chunk_str == "[DONE]":
                                    break
                                try:
                                    json_chunk = json.loads(chunk_str)
                                    content_part = ""
                                    if (
                                        "choices" in json_chunk
                                        and json_chunk["choices"]
                                        and "delta" in json_chunk["choices"][0]
                                        and "content"
                                        in json_chunk["choices"][0]["delta"]
                                    ):
                                        content_part = (
                                            json_chunk["choices"][0]["delta"]["content"]
                                            or ""
                                        )
                                    elif (
                                        "message" in json_chunk
                                        and "content" in json_chunk["message"]
                                    ):
                                        content_part = json_chunk["message"]["content"]

                                    if content_part:
                                        print(content_part, end="", flush=True)
                                        full_response_content += content_part
                                except json.JSONDecodeError:
                                    continue
                except requests.exceptions.RequestException as e:
                    raise requests.exceptions.RequestException(
                        f"An error occurred during the API request: {e}"
                    ) from e
                print()
                return {"role": final_message_role, "content": full_response_content}
            else:
                result = self._make_request("POST", url, json=data, headers=headers)
                if (
                    "choices" in result
                    and result.get("choices")
                    and "message" in result["choices"][0]
                ):
                    return result["choices"][0]["message"]
                elif "message" in result:
                    return result["message"]
                raise ValueError(f"Unexpected response format from API: {result}")


class AgenticClient(Client):
    """
    A client for interacting with the LLM API's /api/chat endpoint,
    with added capabilities for tool integration and automated orchestration.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        system_prompt: str | None = None,
        max_iterations: int = 3,
        api_key: str | None = None,
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
            model_name=model_name,
            system_prompt=system_prompt,
            api_key=api_key,
            **kwargs,
        )
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
        Prioritizes LLM's native 'tool_calls' structure.
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
