Parameter | Type | Default | Description
model | Optional[Model] | None | Model to use for this Agent
name | Optional[str] | None | Agent name
agent_id | Optional[str] | None | Agent UUID (autogenerated if not set)
agent_data | Optional[Dict[str, Any]] | None | Metadata associated with this agent
introduction | Optional[str] | None | Agent introduction (added to chat history when run starts)
user_id | Optional[str] | None | ID of the user interacting with this agent
user_data | Optional[Dict[str, Any]] | None | Metadata associated with the user
session_id | Optional[str] | None | Session UUID (autogenerated if not set)
session_name | Optional[str] | None | Session name
session_state | Optional[Dict[str, Any]] | None | Session state (persisted across runs)
context | Optional[Dict[str, Any]] | None | Context available for tools and prompt functions
add_context | bool | False | If True, add the context to the user prompt
resolve_context | bool | True | If True, resolve the context before running
memory | Optional[Memory] | None | Agent Memory
add_history_to_messages | bool | False | Add chat history to messages
num_history_responses | int | 3 | Number of historical responses to add
knowledge | Optional[AgentKnowledge] | None | Agent Knowledge
add_references | bool | False | Enable RAG by adding references
retriever | Optional[Callable[..., Optional[List[Dict]]]] | None | Function to retrieve references
references_format | Literal["json", "yaml"] | "json" | Format of the references
storage | Optional[AgentStorage] | None | Agent Storage
extra_data | Optional[Dict[str, Any]] | None | Extra data stored with agent
tools | Optional[List[Union[Toolkit, Callable, Function]]] | None | Tools available to the agent
show_tool_calls | bool | False | Show tool calls in the response
tool_call_limit | Optional[int] | None | Max number of tool calls allowed
tool_choice | Optional[Union[str, Dict[str, Any]]] | None | Controls which tool is called
reasoning | bool | False | Enable step-by-step reasoning
reasoning_model | Optional[Model] | None | Model to use for reasoning
reasoning_agent | Optional[Agent] | None | Agent to use for reasoning
reasoning_min_steps | int | 1 | Minimum reasoning steps
reasoning_max_steps | int | 10 | Maximum reasoning steps
read_chat_history | bool | False | Allow model to read chat history
search_knowledge | bool | True | Allow model to search knowledge base
update_knowledge | bool | False | Allow model to update knowledge base
read_tool_call_history | bool | False | Allow model to access tool call history
system_message | Optional[Union[str, Callable, Message]] | None | System message
system_message_role | str | "system" | Role of the system message
create_default_system_message | bool | True | Create default system message
description | Optional[str] | None | Description of the Agent
goal | Optional[str] | None | Task goal
instructions | Optional[Union[str, List[str], Callable]] | None | Instructions for the agent
expected_output | Optional[str] | None | Expected output from the Agent
additional_context | Optional[str] | None | Additional system message context
markdown | bool | False | Format output using markdown
add_name_to_instructions | bool | False | Add agent name to instructions
add_datetime_to_instructions | bool | False | Add current datetime to instructions
add_state_in_messages | bool | False | Add session state to messages
add_messages | Optional[List[Union[Dict, Message]]] | None | Extra messages after system message
user_message | Optional[Union[List, Dict, str, Callable, Message]] | None | User message
user_message_role | str | "user" | Role for the user message
create_default_user_message | bool | True | Create default user message
retries | int | 0 | Number of retries
delay_between_retries | int | 1 | Delay between retries
exponential_backoff | bool | False | Exponential retry delay
response_model | Optional[Type[BaseModel]] | None | Pydantic model for response
parse_response | bool | True | Parse response into response_model
use_json_mode | bool | False | Enable JSON mode for model response
save_response_to_file | Optional[str] | None | Save response to file
stream | Optional[bool] | None | Stream agent response
stream_intermediate_steps | bool | False | Stream intermediate steps
team | Optional[List[Agent]] | None | Team of agents available
team_data | Optional[Dict[str, Any]] | None | Shared data between team members
role | Optional[str] | None | Role of the agent (in team)
respond_directly | bool | False | If True, agent responds directly to user
add_transfer_instructions | bool | True | Add task transfer instructions
team_response_separator | str | "\n" | Separator between team member responses
debug_mode | bool | False | Enable debug logging
monitoring | bool | False | Enable monitoring via agno.com
telemetry | bool | True | Log minimal telemetry

Functions
Function | Description
print_response | Run the agent and print the response
run | Run the agent
aprint_response | Run the agent and print response asynchronously
arun | Run the agent asynchronously
get_session_summary | Get session summary (uses current session if no ID provided)
get_user_memories | Get user memories (uses current user if no ID provided)