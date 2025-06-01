import json
from typing import Any, Dict, List, Optional  # Added Optional for clarity

from agents.agent import Agent
from agents.client import OllamaClient


class Orchestrator(OllamaClient):
    """
    The Orchestrator is responsible for selecting and routing user requests
    to the most appropriate AI agent based on their capabilities.
    It uses its inherited OllamaClient capabilities for decision-making.
    """

    def __init__(
        self,
        name: str = "Main Orchestrator",
        description: str = "Routes requests to specialized agents.",
        model: str = "llama3",
        system_prompt: str | None = None,
        max_iterations: int = 2,
        ollama_uri: str = "http://localhost:11434/api",
        agents: List[Agent] | None = None,
        **ollama_client_kwargs: object,
    ):
        super().__init__(
            base_url=ollama_uri,
            model_name=model,
            system_prompt=system_prompt,
            **ollama_client_kwargs,
        )

        self.name = name
        self.description = description
        self.orchestrator_model = model

        self.registered_agents: Dict[str, Agent] = {}
        self.max_iterations = max_iterations

        if agents:
            for agent_instance in agents:
                self.register_agent(
                    {
                        "handoff": agent_instance,
                        "name": agent_instance.name,
                        "description": agent_instance.description,
                    }
                )

    def register_agent(self, agent_info: Dict[str, Any]):
        """
        Registers an agent with the orchestrator.
        `agent_info` is expected to be a dictionary with 'handoff' (Agent instance),
        'name', and 'description'.
        """
        agent_instance = agent_info.get("handoff")
        agent_name = agent_info.get("name")
        agent_description = agent_info.get("description")

        if not isinstance(agent_instance, Agent):
            raise ValueError("Provided agent_info must contain an instance of Agent.")
        if not agent_name:
            raise ValueError("Provided agent_info must contain 'name'.")
        if not agent_description:
            raise ValueError("Provided agent_info must contain 'description'.")

        if agent_name in self.registered_agents:
            print(
                f"Warning: Agent '{agent_name}' is already registered and will be overwritten."
            )

        self.registered_agents[agent_name] = agent_instance
        print(f"Agent '{agent_name}' registered successfully with {self.name}.")

    # New method: get_agent
    def get_agent(self, agent_name: str) -> Optional[Agent]:
        """
        Retrieves an Agent instance by its registered name.
        Returns the Agent instance if found, otherwise None.
        """
        return self.registered_agents.get(agent_name)

    def _get_agent_descriptions(self) -> str:
        """Formats descriptions of all registered agents for the LLM prompt."""
        if not self.registered_agents:
            return "No agents are currently registered."

        descriptions = []
        for name, agent in self.registered_agents.items():
            descriptions.append(f"- Name: {name}\n  Description: {agent.description}")
        return "\n".join(descriptions)

    def _select_agent(self, user_input: str) -> str | None:
        """
        Uses the LLM (via inherited chat method) to select the most appropriate agent.
        Returns the name of the selected agent or None if no suitable agent is found.
        """
        agent_descriptions = self._get_agent_descriptions()
        orchestrator_system_prompt = f"""
You are an intelligent orchestrator responsible for routing user requests to the most appropriate AI agent.
Below is a list of available agents, each with a name and a description of its capabilities:

{agent_descriptions}

Your task is to analyze the user's request and determine which agent is best suited to handle it.
Respond ONLY with a JSON object containing the name of the chosen agent.
If no agent is suitable, respond with a JSON object indicating "no_agent".

Example response for choosing an agent:
{{"chosen_agent": "AgentName"}}

Example response if no agent is suitable:
{{"chosen_agent": "no_agent"}}

Do not include any other text or explanation in your response.
"""
        print(f"\n--- Orchestrator: Selecting Agent for '{user_input}' ---")

        for i in range(self.max_iterations):
            try:
                llm_response = self.chat(
                    user_prompt=user_input,
                    model_name=self.orchestrator_model,
                    system_prompt=orchestrator_system_prompt,
                    stream=False,
                )

                response_content = llm_response.get("content", "").strip()
                print(f"Orchestrator LLM Raw Response: {response_content}")

                try:
                    parsed_response = json.loads(response_content)
                    chosen_agent_name = parsed_response.get("chosen_agent")

                    if (
                        chosen_agent_name
                        and chosen_agent_name in self.registered_agents
                    ):
                        print(f"Orchestrator selected agent: '{chosen_agent_name}'")
                        return chosen_agent_name
                    elif chosen_agent_name == "no_agent":
                        print("Orchestrator determined no suitable agent.")
                        return None
                    else:
                        print(
                            f"Orchestrator LLM returned an invalid agent name: '{chosen_agent_name}'. Retrying..."
                        )
                        continue
                except json.JSONDecodeError:
                    print("Orchestrator LLM response was not valid JSON. Retrying...")
                    continue

            except Exception as e:
                print(f"Error during agent selection by orchestrator LLM: {e}")
                return None

        print(
            f"Orchestrator failed to select an agent after {self.max_iterations} iterations."
        )
        return None

    def invoke(self, user_input: str) -> Dict[str, Any]:
        """
        Runs the orchestration process: selects an agent and invokes it.
        """
        if not self.registered_agents:
            return {"error": "No agents registered with the orchestrator."}

        chosen_agent_name = self._select_agent(user_input)

        if chosen_agent_name:
            # Use the new get_agent method here
            selected_agent = self.get_agent(chosen_agent_name)
            if selected_agent:  # Add a check in case get_agent returns None (shouldn't happen if _select_agent is correct)
                print(f"\n--- Orchestrator: Invoking '{selected_agent.name}' ---")
                return selected_agent.invoke(user_input)
            else:
                return {
                    "error": f"Selected agent '{chosen_agent_name}' not found after selection. This indicates an internal logic error."
                }
        else:
            return {
                "response": "I couldn't find a suitable agent to handle your request."
            }
