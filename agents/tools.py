import inspect
from functools import wraps
from typing import get_args, get_origin


def tool(
    param_descriptions: dict[str, str] | None = None, required: list[str] | None = None
):
    """
    A decorator to define a function as a tool. It automatically sets parameters
    without default values as 'required'. Explicitly listed 'required' parameters
    with default values will trigger a warning.

    Args:
        param_descriptions (dict[str, str] | None): A dictionary mapping parameter names to their descriptions.
        required (list[str] | None): An *optional* list of parameter names that are explicitly required.
                                     This list primarily serves to provide warnings if it conflicts
                                     with parameters that have default values, as parameters without
                                     defaults are now *automatically* required.
    """
    if param_descriptions is None:
        param_descriptions = {}
    if required is None:
        required = []

    def decorator(func: callable) -> callable:
        tool_name = func.__name__
        tool_description = (
            func.__doc__.strip() if func.__doc__ else f"A tool named {tool_name}."
        )

        parameters_schema: dict[str, any] = {
            "type": "object",
            "properties": {},
            "required": [],  # This will be populated based on default value presence
        }

        # The 'actual_required' list is now primarily used for checking consistency
        # with function defaults, rather than being the sole source of 'required' status.
        actual_required_from_decorator = set(
            required
        )  # Using a set for efficient lookups

        sig = inspect.signature(func)
        for param_name, param in sig.parameters.items():
            # Skip 'self' for instance methods if the decorator is used on them
            if param_name == "self":
                continue

            param_type = "string"  # Default if no hint or unknown hint
            annotation = param.annotation

            if annotation is inspect.Parameter.empty:
                print(
                    f"Warning: Parameter '{param_name}' in tool '{tool_name}' has no type hint. Defaulting to 'string'."
                )
            else:
                # Handle Union types (e.g., str | None)
                if get_origin(annotation) is type(str | int):
                    args = get_args(annotation)
                    non_none_args = [arg for arg in args if arg is not type(None)]
                    if non_none_args:
                        annotation = non_none_args[0]
                    else:
                        print(
                            f"Warning: Union type for '{param_name}' in tool '{tool_name}' contains only NoneType. Defaulting to 'string'."
                        )
                        annotation = str

                # Map Python types to JSON Schema types
                if annotation is str:
                    param_type = "string"
                elif annotation is int:
                    param_type = "integer"
                elif annotation is float:
                    param_type = "number"
                elif annotation is bool:
                    param_type = "boolean"
                elif annotation is list or get_origin(annotation) is list:
                    param_type = "array"
                elif annotation is dict or get_origin(annotation) is dict:
                    param_type = "object"
                else:
                    print(
                        f"Warning: Unsupported type hint '{annotation}' for parameter '{param_name}' in tool '{tool_name}'. Defaulting to 'string'."
                    )

            prop_details: dict[str, any] = {"type": param_type}
            if param_name in param_descriptions:
                prop_details["description"] = param_descriptions[param_name]

            parameters_schema["properties"][param_name] = prop_details

            # --- NEW LOGIC FOR REQUIRED PARAMETERS ---
            if param.default is inspect.Parameter.empty:
                # If a parameter has NO default value, it's considered required by default.
                parameters_schema["required"].append(param_name)
                # No need to check `actual_required_from_decorator` here for adding to required list,
                # as the absence of a default is the primary driver.
            else:
                # If a parameter HAS a default value, it's inherently optional.
                # If it was still listed in the decorator's 'required' argument, it's a conflict.
                if param_name in actual_required_from_decorator:
                    print(
                        f"Warning: Parameter '{param_name}' in tool '{tool_name}' was explicitly listed "
                        f"as 'required' in the decorator, but it has a default value in the function signature. "
                        f"According to JSON Schema rules, parameters with default values are optional. "
                        f"It will be treated as optional in the generated schema."
                    )

        tool_schema = {
            "name": tool_name,
            "description": tool_description,
            "parameters": parameters_schema,
        }

        # Attach schema to the function itself for later registration by the client
        func._tool_schema = tool_schema

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator
