# Chapter 3: BaseTool (Tool Interface)

In the previous chapter, [BaseChatModel (LLM Interface)](02_basechatmodel__llm_interface__.md), we learned how our agent gets its "brain" – the Large Language Model (LLM). This brain is great at understanding and generating text, but what if we need the agent to *do* something more?

Imagine asking your agent, "What's the weather like in Tokyo right now?". The LLM, trained on past data, doesn't know the *current* weather. It needs a way to find out. This is where **Tools** come in!

## What's a Tool? Giving Agents Superpowers

Think of an agent like a smart assistant. While the LLM gives it language skills, it might need specialized equipment to perform certain tasks. These pieces of equipment are **Tools**.

*   A **calculator** tool (`PythonExecutor`) lets the agent perform precise calculations.
*   A **web browser** tool (`WebSearch`) lets the agent look up current information online.
*   A **document reader** tool (`DocParser`) lets the agent read and understand files.
*   An **image generator** tool (`ImageGen`) lets the agent create pictures.

Just like a carpenter uses different tools (hammer, saw, screwdriver) for different jobs, an agent uses different software tools for specific actions.

**`BaseTool` is the blueprint, the standard design, for all these tools.** It defines *how* any tool should be structured so the agent knows how to use it. It's like a universal adapter socket – any tool designed to fit this socket can be plugged into the agent.

Every tool built using the `BaseTool` blueprint must have a specific method called `call`. This `call` method contains the instructions for what the tool actually *does* when the agent decides to use it.

## Key Ideas Behind `BaseTool`

1.  **Standard Structure:** All tools look the same from the agent's perspective. They have a name, a description of what they do, what inputs they need, and a way to be executed (`call`).
2.  **Discoverability:** The agent (often guided by the LLM's function-calling capability, which we'll cover later in [BaseFnCallModel (LLM Function Calling)](05_basefncallmodel__llm_function_calling__.md)) uses the tool's `name` and `description` to figure out *which* tool to use for a specific request.
3.  **Action Execution:** The `call` method is the core of the tool. It takes the necessary inputs (parameters) and performs the action (e.g., searches the web, runs code).
4.  **Extensibility:** You can easily create new tools by following the `BaseTool` blueprint, extending your agent's capabilities.

## Using a Tool: Equipping Your Agent

Let's equip our agent with a `WebSearch` tool. Qwen Agent comes with several built-in tools.

```python
# You need to set your Serper API key first (for WebSearch):
# export SERPER_API_KEY="your_serper_api_key"
# You also need your LLM API key (e.g., DashScope):
# export DASHSCOPE_API_KEY="your_dashscope_api_key"

# 1. Import necessary classes
# We'll use FnCallAgent, which is designed to use tools
from qwen_agent import FnCallAgent
from qwen_agent.llm import get_chat_model

# 2. Set up the LLM
llm_config = {'model': 'qwen-plus'} # Use a Qwen model capable of function calling
llm = get_chat_model(llm_config)

# 3. Create the agent, providing the LLM and the tool name
#    The agent will automatically find and initialize the 'web_search' tool.
agent = FnCallAgent(llm=llm, function_list=['web_search'])

# 4. Prepare our message asking a question that requires search
messages = [{'role': 'user', 'content': 'What is the latest news about Qwen?'}]

# 5. Run the agent (non-streaming for simplicity here)
response = agent.run_nonstream(messages=messages)

# 6. Print the final response
#    The agent uses the web search tool behind the scenes!
print(response[-1]) # Print the last message from the agent
```

**Explanation:**

1.  **Import:** We import `FnCallAgent` (an agent type specifically good at using tools) and `get_chat_model`.
2.  **LLM Setup:** We configure an LLM that supports function calling (like `qwen-plus` or `gpt-4`).
3.  **Create Agent:** Crucially, when creating `FnCallAgent`, we pass `function_list=['web_search']`. This tells the agent: "You have access to the tool named 'web_search'". The agent looks up this name in its registry of known tools ([`TOOL_REGISTRY`](./tools/__init__.py)) and prepares it.
4.  **Message:** We ask a question that the LLM can't answer from its internal knowledge alone.
5.  **Run:** We call `run_nonstream` which waits for the complete answer. The agent's internal logic ([`_run`](01_agent__base_class__.md#under-the-hood-how-agentrun-works)) will involve:
    *   Sending the request and the available tools (including `web_search`'s description) to the LLM.
    *   The LLM deciding to use `web_search` with a query like "latest news about Qwen".
    *   The agent executing the `web_search` tool's `call` method.
    *   The agent sending the search results back to the LLM.
    *   The LLM generating a final answer based on the search results.
6.  **Print Output:** We print the final message from the agent, which should summarize the latest news found online.

**Expected Output (will vary based on current news):**

```
{'role': 'assistant', 'content': 'Based on the latest search results, [Summary of recent Qwen news found by the web search tool]...'}
```

The agent successfully used the `WebSearch` tool to answer a question it couldn't answer otherwise!

## Under the Hood: How Tools are Called

Let's trace the journey when the agent decides to use a tool:

1.  **User Request:** You send a message like "What's the weather in Tokyo?" to the agent's `run` method.
2.  **Agent Logic (`_run`):** The agent's specific `_run` method (e.g., in `FnCallAgent`) processes the message. It sends the message *and* the descriptions of available tools (like `WebSearch`, `AmapWeather`) to the [LLM](02_basechatmodel__llm_interface__.md).
3.  **LLM Decision:** The LLM analyzes the request and the tool descriptions. It recognizes that "weather in Tokyo" requires the weather tool. It sends back a special instruction, essentially saying: "Call the tool named `AmapWeather` with the parameter `location='Tokyo'`." (This uses the LLM's function-calling ability, detailed in [Chapter 5](05_basefncallmodel__llm_function_calling__.md)).
4.  **Tool Detection:** The agent receives the LLM's instruction and detects the request to call a tool (using its `_detect_tool` helper).
5.  **Helper Call (`_call_tool`):** The agent's logic calls its internal `_call_tool` helper method, passing the tool name (`AmapWeather`) and the arguments (`{'location': 'Tokyo'}`).
6.  **Tool Lookup:** `_call_tool` looks inside its `self.function_map` (a dictionary holding all the initialized tools) to find the `AmapWeather` tool object.
7.  **Execute Tool (`tool.call()`):** `_call_tool` executes the `call` method of the found `AmapWeather` object, passing the arguments `{'location': 'Tokyo'}`.
8.  **Tool Action:** The `AmapWeather.call()` method runs its specific code – in this case, making an API request to a weather service like Amap.
9.  **Tool Result:** The weather service sends back the weather data. `AmapWeather.call()` formats this data into a string (e.g., "The weather in Tokyo is clear, 25°C.").
10. **Return Result:** The result string is returned from `AmapWeather.call()` back to `_call_tool`.
11. **Back to Agent Logic:** `_call_tool` returns the result string ("The weather in Tokyo...") to the main agent logic (`_run`).
12. **Final Processing:** The agent often sends this tool result back to the LLM, asking it to formulate a user-friendly final answer (e.g., "The current weather in Tokyo is clear and 25 degrees Celsius.").
13. **Final Response:** The agent yields the final response back to you.

Here's a simplified diagram of the tool-calling flow:

```mermaid
sequenceDiagram
    participant User
    participant Agent._run()
    participant LLM
    participant Agent._call_tool()
    participant SpecificTool as e.g., AmapWeather.call()
    participant ExternalAPI as e.g., Weather Service

    User->>Agent._run(): Ask weather question
    Agent._run()->>LLM: Send message + tool descriptions
    LLM-->>Agent._run(): Instruct agent to call AmapWeather(location='Tokyo')
    Agent._run()->>Agent._call_tool(): Call 'AmapWeather' with {'location': 'Tokyo'}
    Agent._call_tool()->>SpecificTool: call({'location': 'Tokyo'})
    SpecificTool->>ExternalAPI: Request weather data for Tokyo
    ExternalAPI-->>SpecificTool: Weather data response
    SpecificTool-->>Agent._call_tool(): Return formatted weather string
    Agent._call_tool()-->>Agent._run(): Tool result string
    Agent._run()->>LLM: Send tool result for final response generation
    LLM-->>Agent._run(): Final user-friendly answer
    Agent._run()-->>User: Yield final answer
```

## Diving into the Code (Simplified)

Let's look at where `BaseTool` and the calling mechanism are defined.

**1. The Blueprint (`tools/base.py`)**

This file defines the `BaseTool` abstract class.

```python
# File: tools/base.py (Simplified)
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

# A dictionary to keep track of all registered tools
TOOL_REGISTRY = {}

# Decorator function to register a tool class
def register_tool(name, ...):
    def decorator(cls):
        TOOL_REGISTRY[name] = cls
        cls.name = name
        # ... more setup ...
        return cls
    return decorator

# The abstract base class for all tools
class BaseTool(ABC):
    name: str = ''          # Unique name for the tool
    description: str = ''   # What the tool does (for LLM)
    parameters: Union[List[dict], dict] = [] # Inputs needed

    def __init__(self, cfg: Optional[dict] = None):
        # Stores configuration if provided
        self.cfg = cfg or {}
        # ... validation ...

    @abstractmethod # This method MUST be implemented by subclasses
    def call(self, params: Union[str, dict], **kwargs) -> Union[str, list, dict, ...]:
        # The actual code that performs the tool's action
        raise NotImplementedError

    # Helper to get the tool's definition for the LLM
    @property
    def function(self) -> dict:
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
        }
```

**Key Takeaways:**

*   **`ABC` and `@abstractmethod`:** Like `Agent` and `BaseChatModel`, `BaseTool` is an Abstract Base Class. You can't use it directly. Any class that inherits from it *must* provide its own implementation for the `call` method.
*   **`name`, `description`, `parameters`:** These attributes define the tool's identity and interface, used by the LLM to decide when and how to call it.
*   **`call`:** The heart of the tool, containing its execution logic.
*   **`TOOL_REGISTRY` and `@register_tool`:** A simple way Qwen Agent keeps track of available tools. When you provide `function_list=['web_search']` to the agent, it looks up `'web_search'` in this registry.

**2. A Concrete Tool (`tools/web_search.py`)**

This file shows a real tool implementing the `BaseTool` interface.

```python
# File: tools/web_search.py (Simplified)
import os
import requests # Library to make web requests
from qwen_agent.tools.base import BaseTool, register_tool

# Get API key from environment variable
SERPER_API_KEY = os.getenv('SERPER_API_KEY', '')
SERPER_URL = 'https://google.serper.dev/search'

@register_tool('web_search') # Register this class with the name 'web_search'
class WebSearch(BaseTool): # Inherits from BaseTool
    name = 'web_search'
    description = 'Search for information from the internet.'
    parameters = { # Defines the input: a 'query' string
        'type': 'object',
        'properties': {'query': {'type': 'string'}},
        'required': ['query'],
    }

    # The required implementation of the 'call' method
    def call(self, params: Union[str, dict], **kwargs) -> str:
        # 1. Parse the input parameters
        params = self._verify_json_format_args(params)
        query = params['query']

        # 2. Perform the action (call Serper API)
        headers = {'X-API-KEY': SERPER_API_KEY, ...}
        payload = {'q': query}
        response = requests.post(SERPER_URL, json=payload, headers=headers)
        # ... error handling ...
        search_results = response.json()['organic']

        # 3. Format and return the result as a string
        return self._format_results(search_results)

    def _format_results(self, search_results) -> str:
        # Helper to make results readable
        # ... formatting logic ...
        return formatted_string
```

**Key Takeaways:**

*   **`@register_tool('web_search')`:** Makes this tool available to agents via the name `'web_search'`.
*   **Inheritance:** `WebSearch` inherits from `BaseTool`.
*   **Attributes:** It defines its specific `name`, `description`, and `parameters`.
*   **`call` Implementation:** It provides the concrete logic for `call`, which involves parsing inputs, calling the external Serper search API using `requests`, and formatting the results.

**3. Calling the Tool (`agent.py`)**

Inside the base `Agent` class, the `_call_tool` method handles the execution.

```python
# File: agent.py (Simplified, inside Agent class)

class Agent(ABC):
    # ... ( __init__ stores tools in self.function_map ) ...

    def _call_tool(self, tool_name: str, tool_args: Union[str, dict], **kwargs) -> ...:
        # 1. Check if the tool exists in the agent's map
        if tool_name not in self.function_map:
            return f'Tool {tool_name} does not exists.'

        # 2. Get the tool object
        tool = self.function_map[tool_name]

        # 3. Execute the tool's specific call method
        try:
            tool_result = tool.call(tool_args, **kwargs)
        except Exception as e:
            # Handle errors during tool execution
            logger.warning(f'Error calling tool {tool_name}: {e}')
            return f"Error: {e}" # Return error message

        # 4. Format and return the result
        # ... (converts non-string results to JSON string) ...
        return tool_result # Or formatted string
```

**Key Takeaways:**

*   **Lookup:** It uses the `tool_name` provided by the agent's logic (which got it from the LLM) to find the corresponding tool object in `self.function_map`.
*   **Execution:** It calls the `call` method *on the specific tool object* (`tool.call(...)`). This is polymorphism in action – `_call_tool` doesn't need to know *how* `WebSearch` or `CodeInterpreter` works, it just needs to know they both have a `call` method defined by the `BaseTool` interface.
*   **Error Handling:** It includes basic error handling if the tool execution fails.

## Conclusion

You've now learned about `BaseTool`, the essential interface that allows Qwen Agents to gain new capabilities beyond just talking!

*   `BaseTool` acts as a **standard blueprint** for creating tools like web search, calculators, or document readers.
*   It defines a common structure (`name`, `description`, `parameters`) and requires a `call` method for execution.
*   Agents use the tool's description to decide *when* to use it (often via LLM function calling).
*   The agent's `_call_tool` method finds the right tool and executes its `call` method to perform the action.
*   This system makes agents **extensible**, allowing you to easily add new skills.

We've seen how the Agent, the LLM, and Tools interact. But how do they pass information back and forth? They use a standardized format called `Message`.

**Next Chapter:** [Message (Communication Structure)](04_message__communication_structure__.md)

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)