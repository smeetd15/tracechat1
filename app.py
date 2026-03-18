import json
import gradio as gr
from dotenv import load_dotenv
load_dotenv()  # picks up ANTHROPIC_API_KEY from .env when running locally
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_TOKEN_THRESHOLD = 5000   # traces below this are kept as-is
CHUNK_TARGET_TOKENS   = 1500   # soft target tokens per chunk
CHUNK_OVERLAP_STEPS   = 2      # how many groups to repeat at boundaries
MODEL                 = "claude-sonnet-4-6"

def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Trace ingestion & formatting
# ---------------------------------------------------------------------------

def _extract_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                t = item.get("type", "")
                if t == "text":
                    parts.append(item.get("text", "").strip())
                elif t == "tool_use":
                    name = item.get("name", "")
                    inp = item.get("input", {})
                    parts.append(f"{name}({json.dumps(inp, ensure_ascii=False)})")
                elif t == "tool_result":
                    inner = item.get("content", "")
                    parts.append(f"result: {_extract_text(inner)}")
                else:
                    parts.append(_extract_text(item))
            else:
                parts.append(str(item))
        return " | ".join(p for p in parts if p)
    if isinstance(value, dict):
        for key in ("text", "content", "output", "result", "message"):
            if key in value:
                return _extract_text(value[key])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _infer_step_type(step: dict) -> str:
    for key in ("type", "role", "event", "kind", "step_type"):
        val = step.get(key)
        if val:
            return str(val)
    if "tool_calls" in step or "function_call" in step:
        return "tool_call"
    if "tool_call_id" in step or "tool_result" in step:
        return "tool_result"
    if "content" in step:
        role = step.get("role", "")
        if role:
            return role
    return "step"


def _timestamp(step: dict) -> str:
    for key in ("timestamp", "time", "created_at", "ts", "start_time"):
        val = step.get(key)
        if val is not None:
            return str(val)
    return "-"


def ingest_trace(raw_json: str) -> list[dict]:
    data = json.loads(raw_json)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("steps", "messages", "trace", "events", "turns", "spans"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return list(data.values()) if data else []
    return []


def format_step(i: int, step: dict) -> str:
    ts = _timestamp(step)
    stype = _infer_step_type(step)
    content_raw = step.get(
        "content",
        step.get("output", step.get("result", step.get("message", step.get("text", step))))
    )
    if content_raw is step:
        content_raw = {k: v for k, v in step.items()
                       if k not in ("type", "role", "event", "kind", "step_type",
                                    "timestamp", "time", "created_at", "ts", "start_time")}
    return f"[step {i}] {ts} [{stype}] {_extract_text(content_raw)}"


def format_trace(steps: list[dict]) -> str:
    return "\n\n".join(format_step(i, s) for i, s in enumerate(steps, start=1))


# ---------------------------------------------------------------------------
# Smart chunking
# ---------------------------------------------------------------------------

def _is_tool_call(step: dict) -> bool:
    stype = _infer_step_type(step)
    if stype in ("tool_call", "function_call"):
        return True
    content = step.get("content", [])
    if isinstance(content, list):
        return any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content)
    return "tool_calls" in step or "function_call" in step


def _is_tool_result(step: dict) -> bool:
    stype = _infer_step_type(step)
    if stype in ("tool", "tool_result", "function"):
        return True
    content = step.get("content", [])
    if isinstance(content, list):
        return any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
    return "tool_call_id" in step or "tool_result" in step


def _group_steps(steps: list[dict]) -> list[list[int]]:
    """Bond each tool_call with its immediately following tool_result(s)."""
    groups: list[list[int]] = []
    i = 0
    while i < len(steps):
        if _is_tool_call(steps[i]):
            group = [i]
            j = i + 1
            while j < len(steps) and _is_tool_result(steps[j]):
                group.append(j)
                j += 1
            groups.append(group)
            i = j
        else:
            groups.append([i])
            i += 1
    return groups


def chunk_steps(steps: list[dict]) -> list[list[dict]]:
    groups = _group_steps(steps)
    group_texts = []
    for grp in groups:
        text = "\n".join(format_step(idx + 1, steps[idx]) for idx in grp)
        group_texts.append((_approx_tokens(text), grp))

    chunks: list[list[dict]] = []
    current_indices: list[int] = []
    current_tokens = 0
    g = 0

    while g < len(group_texts):
        tok, grp = group_texts[g]
        if current_tokens + tok > CHUNK_TARGET_TOKENS and current_indices:
            chunks.append([steps[i] for i in current_indices])
            overlap_groups = groups[max(0, g - CHUNK_OVERLAP_STEPS): g]
            current_indices = [i for og in overlap_groups for i in og]
            current_tokens = sum(group_texts[gi][0] for gi in range(max(0, g - CHUNK_OVERLAP_STEPS), g))
        current_indices.extend(grp)
        current_tokens += tok
        g += 1

    if current_indices:
        chunks.append([steps[i] for i in current_indices])
    return chunks


def chunk_trace(steps: list[dict]) -> tuple[list[list[dict]], bool]:
    total_tokens = _approx_tokens(format_trace(steps))
    if total_tokens <= CHUNK_TOKEN_THRESHOLD:
        return [steps], False
    return chunk_steps(steps), True


def format_chunks(chunks: list[list[dict]], steps: list[dict]) -> str:
    step_index = {id(s): i for i, s in enumerate(steps, start=1)}
    parts = []
    for c_idx, chunk in enumerate(chunks, start=1):
        header = f"{'─'*60}\n  CHUNK {c_idx} / {len(chunks)}\n{'─'*60}"
        lines = [header] + [format_step(step_index.get(id(s), "?"), s) for s in chunk]
        parts.append("\n\n".join(lines))
    return "\n\n\n".join(parts)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert AI trace analyzer. A trace is a log of steps recorded during \
an AI agent's execution — it may include user messages, LLM responses, tool calls, \
and tool results.

Your rules:
1. Answer ONLY based on what is explicitly in the trace. Do not hallucinate or infer \
information that isn't there.
2. Cite the exact step(s) supporting your answer using [step N] notation inline.
3. If the trace does not contain enough information to answer, say so clearly.
4. Be concise. Every claim must point back to a specific step.\
"""


def _score_chunk(chunk_text: str, query: str) -> float:
    """BM25-inspired term frequency score (no external deps)."""
    terms = set(query.lower().split())
    if not terms:
        return 0.0
    words = chunk_text.lower().split()
    word_count = len(words) or 1
    tf = sum(words.count(t) for t in terms)
    # Normalise by chunk length to avoid bias toward longer chunks
    return tf / (word_count ** 0.5)


def _make_retrieve_tool(formatted_chunks: list[str], top_k: int = 2):
    """Return a retrieve_trace LangChain tool closed over the formatted chunk texts."""

    @tool
    def retrieve_trace(query: str) -> str:
        """Search the trace for segments relevant to the query.
        Returns the most relevant trace chunks with their step numbers.
        Call this before answering any question about a large trace."""
        scored = sorted(
            enumerate(formatted_chunks),
            key=lambda t: _score_chunk(t[1], query),
            reverse=True,
        )
        top = scored[:top_k]
        # Always include at least one chunk even with zero score
        if not top:
            top = [(0, formatted_chunks[0])]

        results = [
            f"[Chunk {idx + 1} of {len(formatted_chunks)}]\n{text}"
            for idx, text in top
        ]
        return "\n\n---\n\n".join(results)

    return retrieve_trace


def build_agent(steps: list[dict], chunks: list[list[dict]], was_chunked: bool):
    """
    Build a LangGraph ReAct agent.
    - Small trace  → full trace embedded in system prompt, no tools.
    - Large trace  → retrieve_trace tool; agent decides which chunks to read.
    """
    llm = ChatAnthropic(model=MODEL, temperature=0)
    step_index = {id(s): i for i, s in enumerate(steps, start=1)}

    if was_chunked:
        # Pre-format each chunk with global step numbers
        formatted_chunks = []
        for chunk in chunks:
            lines = [format_step(step_index.get(id(s), "?"), s) for s in chunk]
            formatted_chunks.append("\n\n".join(lines))

        retrieve_tool = _make_retrieve_tool(formatted_chunks)
        system = (
            SYSTEM_PROMPT
            + f"\n\nThis trace is large ({len(chunks)} chunks). "
            "Use the retrieve_trace tool to find relevant segments before answering. "
            "Do not answer until you have retrieved and read the relevant steps."
        )
        agent = create_react_agent(llm, [retrieve_tool], prompt=system)
    else:
        full_trace = format_trace(steps)
        system = SYSTEM_PROMPT + f"\n\n<trace>\n{full_trace}\n</trace>"
        agent = create_react_agent(llm, [], prompt=system)

    return agent


def run_agent(agent, history: list[dict], user_query: str) -> tuple[str, list[dict]]:
    """
    Invoke the agent.

    history  — list of {"role": "user"|"assistant", "content": str} dicts
               representing the conversation so far (no LangChain objects stored).
    Returns  (answer_text, updated_history).
    """
    # Convert stored history dicts → proper LangChain ChatMessage objects,
    # then append the new HumanMessage. This is the format LangGraph requires.
    def _to_lc(turn: dict):
        if turn["role"] == "user":
            return HumanMessage(content=turn["content"])
        return AIMessage(content=turn["content"])

    messages = [_to_lc(t) for t in history] + [HumanMessage(content=user_query)]

    result = agent.invoke({"messages": messages})

    # Extract the last assistant text from the returned message list
    answer = "(No response)"
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            c = msg.content
            if isinstance(c, str) and c.strip():
                answer = c.strip()
                break
            if isinstance(c, list):
                parts = [b["text"] for b in c if isinstance(b, dict) and b.get("type") == "text"]
                text = "\n".join(parts).strip()
                if text:
                    answer = text
                    break

    updated_history = history + [
        {"role": "user",      "content": user_query},
        {"role": "assistant", "content": answer},
    ]
    return answer, updated_history


# ---------------------------------------------------------------------------
# Gradio helpers
# ---------------------------------------------------------------------------

SAMPLE_TRACE = json.dumps({
    "steps": [
        {"role": "user", "timestamp": "2024-01-01T10:00:00Z",
         "content": "What's the weather today?"},
        {"role": "assistant", "timestamp": "2024-01-01T10:00:01Z",
         "content": [{"type": "tool_use", "name": "get_weather",
                      "input": {"location": "San Francisco"}}]},
        {"role": "tool", "timestamp": "2024-01-01T10:00:02Z",
         "content": [{"type": "tool_result", "content": "Sunny, 68°F"}]},
        {"role": "assistant", "timestamp": "2024-01-01T10:00:03Z",
         "content": "It's sunny and 68°F in San Francisco today!"},
    ]
}, indent=2)


def _render(steps: list[dict]) -> tuple[str, str, list[list[dict]], bool]:
    chunks, was_chunked = chunk_trace(steps)
    total_tokens = _approx_tokens(format_trace(steps))
    if was_chunked:
        display = format_chunks(chunks, steps)
        info = (f"Chunked into {len(chunks)} chunks  |  "
                f"~{total_tokens:,} tokens  |  overlap={CHUNK_OVERLAP_STEPS} groups  |  "
                f"target={CHUNK_TARGET_TOKENS} tok/chunk  →  agent will use retrieve_trace tool")
    else:
        display = format_trace(steps)
        info = (f"~{total_tokens:,} tokens — fits in context (threshold={CHUNK_TOKEN_THRESHOLD:,})  "
                f"→  full trace passed directly to agent")
    return display, info, chunks, was_chunked


def _load(steps: list[dict]):
    """Shared logic for upload / paste / sample: render trace + build agent state."""
    if not steps:
        return "", "", None, [], []

    display, info, chunks, was_chunked = _render(steps)
    agent = build_agent(steps, chunks, was_chunked)
    # agent_state: [agent_obj, lc_messages_list]
    return display, info, agent, []


def process_upload(file_obj):
    if file_obj is None:
        return "", "", None, [], []
    try:
        raw = open(file_obj, "r", encoding="utf-8").read()
        steps = ingest_trace(raw)
        display, info, agent, lc_msgs = _load(steps)
        return raw, display, info, agent, lc_msgs, []   # also clear chatbot
    except Exception as e:
        return "", f"Error: {e}", "", None, [], []


def process_paste(raw_json: str):
    if not raw_json.strip():
        return raw_json, "", "", None, [], []
    try:
        steps = ingest_trace(raw_json)
        display, info, agent, lc_msgs = _load(steps)
        return raw_json, display, info, agent, lc_msgs, []
    except Exception as e:
        return raw_json, f"Error: {e}", "", None, [], []


def load_sample():
    steps = ingest_trace(SAMPLE_TRACE)
    display, info, agent, lc_msgs = _load(steps)
    return SAMPLE_TRACE, display, info, agent, lc_msgs, []


def chat_submit(user_msg: str, chatbot_history: list, agent, conv_history: list):
    """Handle a chat turn.

    conv_history — plain {"role", "content"} dicts accumulated across turns.
    """
    if not user_msg.strip():
        return chatbot_history, agent, conv_history, ""

    if agent is None:
        new_chatbot = chatbot_history + [
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": "Please load a trace first."},
        ]
        return new_chatbot, agent, conv_history, ""

    try:
        answer, updated_conv = run_agent(agent, conv_history, user_msg)
    except Exception as e:
        answer = f"Error: {e}"
        updated_conv = conv_history

    new_chatbot = chatbot_history + [
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": answer},
    ]
    return new_chatbot, agent, updated_conv, ""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="TraceChat") as demo:
    gr.Markdown("# TraceChat\nIngest a trace, view it step-by-step, then ask questions about it.")

    # Shared state
    agent_state  = gr.State(None)   # LangGraph agent object
    lc_msg_state = gr.State([])     # LangChain message history (for multi-turn)

    # ── Top row: input controls ──────────────────────────────────────────────
    with gr.Row():
        upload_btn  = gr.File(label="Upload trace (.json)", file_types=[".json"], scale=2)
        sample_btn  = gr.Button("Load sample trace", variant="secondary", scale=1)

    raw_box = gr.Code(label="Raw JSON (paste / edit here)", language="json", lines=8)

    # ── Bottom row: trace view | chat ────────────────────────────────────────
    with gr.Row(equal_height=False):

        # Left: formatted trace
        with gr.Column(scale=1):
            gr.Markdown("### Formatted Trace")
            chunk_info    = gr.Textbox(label="Trace info", interactive=False, lines=2)
            formatted_box = gr.Textbox(label="", lines=30, interactive=False)

        # Right: chat
        with gr.Column(scale=1):
            gr.Markdown("### Chat with Trace")
            chatbot = gr.Chatbot(label="", height=520)
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask something about this trace…",
                    label="",
                    scale=4,
                )
                send_btn  = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear chat", variant="secondary")

    # ── Event wiring ──────────────────────────────────────────────────────────
    _trace_outputs = [raw_box, formatted_box, chunk_info, agent_state, lc_msg_state, chatbot]

    upload_btn.change(fn=process_upload, inputs=[upload_btn],  outputs=_trace_outputs)
    raw_box.change(fn=process_paste,   inputs=[raw_box],      outputs=_trace_outputs)
    sample_btn.click(fn=load_sample,   inputs=[],             outputs=_trace_outputs)

    _chat_inputs  = [msg_input, chatbot, agent_state, lc_msg_state]
    _chat_outputs = [chatbot, agent_state, lc_msg_state, msg_input]

    send_btn.click(fn=chat_submit, inputs=_chat_inputs, outputs=_chat_outputs)
    msg_input.submit(fn=chat_submit, inputs=_chat_inputs, outputs=_chat_outputs)
    clear_btn.click(fn=lambda: ([], []), inputs=[], outputs=[chatbot, lc_msg_state])


if __name__ == "__main__":
    demo.launch()
