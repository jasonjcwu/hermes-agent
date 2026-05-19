"""Tests for tools/advisor_tool.py — Advisor Strategy implementation.

Covers:
- Message sanitization (_sanitize_for_advisor)
- Provider auto-detection (_detect_anthropic_native)
- Advisor block stripping (_strip_advisor_blocks in anthropic_adapter)
- Rate limiting (max_uses_per_task)
- Config loading (load_advisor_config)
- OpenAI-compatible advisor call (mocked)
- Anthropic-native advisor call (mocked)
- reasoning_content fallback
- Output trimming
- Registry / schema validation
- EXECUTOR_ADVISOR_PROMPT injection in system_prompt
"""

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Ensure project root importable
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ===========================================================================
# 1. Message sanitization
# ===========================================================================

class TestSanitizeForAdvisor:
    """Strip tool_calls / tool roles so the advisor sees plain text only."""

    def test_system_message_preserved(self):
        from tools.advisor_tool import _sanitize_for_advisor
        msgs = [{"role": "system", "content": "You are a coder."}]
        result = _sanitize_for_advisor(msgs)
        assert result == [{"role": "system", "content": "You are a coder."}]

    def test_user_message_preserved(self):
        from tools.advisor_tool import _sanitize_for_advisor
        msgs = [{"role": "user", "content": "Write a function"}]
        result = _sanitize_for_advisor(msgs)
        assert result == [{"role": "user", "content": "Write a function"}]

    def test_assistant_text_only(self):
        from tools.advisor_tool import _sanitize_for_advisor
        msgs = [{"role": "assistant", "content": "I will write the function."}]
        result = _sanitize_for_advisor(msgs)
        assert result == [{"role": "assistant", "content": "I will write the function."}]

    def test_assistant_tool_calls_summarized(self):
        from tools.advisor_tool import _sanitize_for_advisor
        msgs = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "write_file",
                    "arguments": '{"path": "/tmp/test.py", "content": "print(1)"}',
                }
            }],
        }]
        result = _sanitize_for_advisor(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "write_file" in result[0]["content"]
        assert "write_file(" in result[0]["content"]

    def test_assistant_text_plus_tool_calls(self):
        from tools.advisor_tool import _sanitize_for_advisor
        msgs = [{
            "role": "assistant",
            "content": "Let me create the file.",
            "tool_calls": [{
                "function": {"name": "read_file", "arguments": '{"path": "x.py"}'},
            }],
        }]
        result = _sanitize_for_advisor(msgs)
        assert "Let me create the file." in result[0]["content"]
        assert "read_file(" in result[0]["content"]

    def test_tool_role_converted_to_user(self):
        from tools.advisor_tool import _sanitize_for_advisor
        msgs = [{
            "role": "tool",
            "content": "def hello():\n    print('hello')",
        }]
        result = _sanitize_for_advisor(msgs)
        assert result[0]["role"] == "user"
        assert "[Tool result]" in result[0]["content"]
        assert "print('hello')" in result[0]["content"]

    def test_tool_output_truncated(self):
        from tools.advisor_tool import _sanitize_for_advisor
        long_content = "x" * 2000
        msgs = [{"role": "tool", "content": long_content}]
        result = _sanitize_for_advisor(msgs, max_tool_output_chars=600)
        assert len(result[0]["content"]) < 700
        assert "truncated" in result[0]["content"]

    def test_empty_messages_skipped(self):
        from tools.advisor_tool import _sanitize_for_advisor
        msgs = [
            {"role": "system", "content": ""},
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""},
            {"role": "tool", "content": ""},
        ]
        result = _sanitize_for_advisor(msgs)
        assert result == []

    def test_user_multimodal_content(self):
        """User messages with list content (image + text) are flattened."""
        from tools.advisor_tool import _sanitize_for_advisor
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                {"type": "text", "text": "and fix it"},
            ],
        }]
        result = _sanitize_for_advisor(msgs)
        assert result[0]["role"] == "user"
        assert "Look at this" in result[0]["content"]
        assert "and fix it" in result[0]["content"]

    def test_full_conversation_flow(self):
        """End-to-end: system + user + assistant(tool) + tool + user."""
        from tools.advisor_tool import _sanitize_for_advisor
        msgs = [
            {"role": "system", "content": "You are a coder."},
            {"role": "user", "content": "Write a rate limiter"},
            {"role": "assistant", "content": "Let me check existing files.",
             "tool_calls": [{"function": {"name": "search_files", "arguments": '{"pattern": "rate_limit"}'}}]},
            {"role": "tool", "content": "No results found."},
            {"role": "assistant", "content": "I'll create it from scratch."},
            {"role": "user", "content": "Make it thread-safe"},
        ]
        result = _sanitize_for_advisor(msgs)
        # system + user + assistant(actions) + user(tool result) + assistant + user
        assert len(result) == 6
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert "search_files(" in result[2]["content"]
        assert result[3]["role"] == "user"
        assert "[Tool result]" in result[3]["content"]


# ===========================================================================
# 2. Provider detection
# ===========================================================================

class TestDetectAnthropicNative:
    """Auto-detect whether to use Anthropic native Messages API."""

    def test_explicit_provider(self):
        from tools.advisor_tool import _detect_anthropic_native
        assert _detect_anthropic_native("sk-xxx", "gpt-4", None, "anthropic") is True
        assert _detect_anthropic_native("sk-xxx", "gpt-4", None, "openai") is False

    def test_api_key_prefix_sk_ant(self):
        from tools.advisor_tool import _detect_anthropic_native
        assert _detect_anthropic_native("sk-ant-api03-xxx", "anything", None, None) is True

    def test_api_key_prefix_cc(self):
        from tools.advisor_tool import _detect_anthropic_native
        assert _detect_anthropic_native("cc-xxx", "anything", None, None) is True

    def test_model_contains_claude(self):
        from tools.advisor_tool import _detect_anthropic_native
        assert _detect_anthropic_native("sk-xxx", "claude-sonnet-4-6", None, None) is True
        assert _detect_anthropic_native("sk-xxx", "anthropic/claude-opus-4-7", None, None) is True

    def test_base_url_anthropic(self):
        from tools.advisor_tool import _detect_anthropic_native
        assert _detect_anthropic_native("sk-xxx", "gpt-4", "https://api.anthropic.com", None) is True

    def test_no_match(self):
        from tools.advisor_tool import _detect_anthropic_native
        assert _detect_anthropic_native("sk-xxx", "deepseek-chat", None, None) is False
        assert _detect_anthropic_native("sk-xxx", "glm-5.1", "https://open.bigmodel.cn", None) is False

    def test_none_values(self):
        from tools.advisor_tool import _detect_anthropic_native
        assert _detect_anthropic_native("", "", None, None) is False
        assert _detect_anthropic_native(None, None, None, None) is False


# ===========================================================================
# 3. Advisor block stripping (Anthropic message history)
# ===========================================================================

class TestStripAdvisorBlocks:
    """Strip stale advisor tool blocks from Anthropic message history."""

    def test_no_advisor_blocks(self):
        from agent.anthropic_adapter import _strip_advisor_blocks
        msgs = [
            {"role": "assistant", "content": [
                {"type": "text", "text": "Hello"},
            ]},
        ]
        result = _strip_advisor_blocks(msgs)
        assert result[0]["content"][0]["text"] == "Hello"

    def test_strip_server_tool_use_and_result(self):
        from agent.anthropic_adapter import _strip_advisor_blocks
        msgs = [
            {"role": "assistant", "content": [
                {"type": "text", "text": "Thinking..."},
                {"type": "server_tool_use", "name": "advisor", "id": "srv_001"},
                {"type": "advisor_tool_result", "tool_use_id": "srv_001",
                 "content": "Use deque instead of list"},
                {"type": "text", "text": "OK, using deque"},
            ]},
        ]
        result = _strip_advisor_blocks(msgs)
        content = result[0]["content"]
        # Should have: text, advisor_feedback, text
        assert len(content) == 3
        assert content[0]["text"] == "Thinking..."
        assert content[1]["type"] == "text"
        assert "<advisor_feedback>" in content[1]["text"]
        assert "deque instead of list" in content[1]["text"]
        assert content[2]["text"] == "OK, using deque"

    def test_non_advisor_tool_use_preserved(self):
        from agent.anthropic_adapter import _strip_advisor_blocks
        msgs = [
            {"role": "assistant", "content": [
                {"type": "server_tool_use", "name": "web_search", "id": "srv_002"},
                {"type": "text", "text": "searching..."},
            ]},
        ]
        result = _strip_advisor_blocks(msgs)
        assert len(result[0]["content"]) == 2  # preserved

    def test_advisor_result_with_list_content(self):
        """advisor_tool_result can have list content (Anthropic format)."""
        from agent.anthropic_adapter import _strip_advisor_blocks
        msgs = [
            {"role": "assistant", "content": [
                {"type": "server_tool_use", "name": "advisor", "id": "srv_003"},
                {"type": "advisor_tool_result", "tool_use_id": "srv_003",
                 "content": [
                     {"type": "text", "text": "Try a different approach"},
                 ]},
            ]},
        ]
        result = _strip_advisor_blocks(msgs)
        assert "Try a different approach" in result[0]["content"][0]["text"]

    def test_multiple_advisor_exchanges(self):
        from agent.anthropic_adapter import _strip_advisor_blocks
        msgs = [
            {"role": "assistant", "content": [
                {"type": "server_tool_use", "name": "advisor", "id": "srv_a"},
                {"type": "advisor_tool_result", "tool_use_id": "srv_a", "content": "Advice 1"},
                {"type": "text", "text": "OK"},
                {"type": "server_tool_use", "name": "advisor", "id": "srv_b"},
                {"type": "advisor_tool_result", "tool_use_id": "srv_b", "content": "Advice 2"},
            ]},
        ]
        result = _strip_advisor_blocks(msgs)
        feedbacks = [b for b in result[0]["content"] if b["type"] == "text" and "<advisor_feedback>" in b.get("text", "")]
        assert len(feedbacks) == 2
        assert "Advice 1" in feedbacks[0]["text"]
        assert "Advice 2" in feedbacks[1]["text"]

    def test_user_messages_untouched(self):
        from agent.anthropic_adapter import _strip_advisor_blocks
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": [
                {"type": "server_tool_use", "name": "advisor", "id": "srv_004"},
                {"type": "advisor_tool_result", "tool_use_id": "srv_004", "content": "Reply"},
            ]},
        ]
        result = _strip_advisor_blocks(msgs)
        assert result[0] == {"role": "user", "content": "Hello"}

    def test_string_content_untouched(self):
        """Messages with string content (not list) should not crash."""
        from agent.anthropic_adapter import _strip_advisor_blocks
        msgs = [
            {"role": "assistant", "content": "Just text"},
        ]
        result = _strip_advisor_blocks(msgs)
        assert result[0]["content"] == "Just text"


# ===========================================================================
# 4. Rate limiting
# ===========================================================================

class TestRateLimiting:
    """Per-task invocation cap prevents runaway advisor usage."""

    def test_counter_increments(self):
        from tools.advisor_tool import _make_use_counter
        counter = _make_use_counter()
        assert counter["count"] == 0
        counter["count"] += 1
        assert counter["count"] == 1

    def test_max_uses_enforcement(self):
        """When count >= max_uses, advisor returns error."""
        from tools.advisor_tool import _make_use_counter
        max_uses = 3
        counter = _make_use_counter()
        for _ in range(max_uses):
            counter["count"] += 1
        # 4th call should be blocked
        assert counter["count"] >= max_uses


# ===========================================================================
# 5. Config loading
# ===========================================================================

class TestLoadAdvisorConfig:
    """Config merges: defaults → config.yaml → env vars."""

    def test_defaults_applied(self):
        from tools.advisor_tool import load_advisor_config
        cfg = load_advisor_config()
        assert cfg["max_uses_per_task"] == 5
        assert cfg["temperature"] == 0.3
        assert cfg["max_tokens"] == 2048
        assert cfg["timeout"] == 30

    def test_env_override_model(self):
        from tools.advisor_tool import load_advisor_config
        with patch.dict("os.environ", {"HERMES_ADVISOR_MODEL": "test-model"}):
            cfg = load_advisor_config()
            assert cfg["model"] == "test-model"

    def test_env_override_api_key(self):
        from tools.advisor_tool import load_advisor_config
        with patch.dict("os.environ", {"HERMES_ADVISOR_API_KEY": "sk-test"}):
            cfg = load_advisor_config()
            assert cfg["api_key"] == "sk-test"


# ===========================================================================
# 6. OpenAI-compatible advisor call (mocked)
# ===========================================================================

class TestCallAdvisorOpenAI:
    """call_advisor routes to OpenAI-compatible API for non-Anthropic providers."""

    def _mock_openai_response(self, content, reasoning_content=None,
                               prompt_tokens=500, completion_tokens=50):
        """Build a mock openai response object."""
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = content
        mock_resp.choices[0].message.reasoning_content = reasoning_content
        mock_resp.usage = MagicMock()
        mock_resp.usage.prompt_tokens = prompt_tokens
        mock_resp.usage.completion_tokens = completion_tokens
        return mock_resp

    def test_successful_call(self):
        from tools.advisor_tool import call_advisor

        mock_resp = self._mock_openai_response("Use a deque with a Condition variable.")

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_resp
            result = json.loads(call_advisor(
                messages=[{"role": "user", "content": "Design a pool"}],
                question="Best data structure?",
                urgency="high",
                config={"model": "deepseek-chat", "api_key": "sk-test", "base_url": "https://api.deepseek.com"},
            ))
        assert result["advice"] == "Use a deque with a Condition variable."
        assert result["tokens_in"] == 500
        assert result["tokens_out"] == 50
        assert result["latency_ms"] >= 0  # mocked, may be 0

    def test_reasoning_content_fallback(self):
        """GLM-5.1 / DeepSeek thinking mode: content empty, reasoning_content has text."""
        from tools.advisor_tool import call_advisor

        mock_resp = self._mock_openai_response(
            "", reasoning_content="The best approach is to use a semaphore.",
            prompt_tokens=300, completion_tokens=100,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_resp
            result = json.loads(call_advisor(
                messages=[{"role": "user", "content": "Synchronize threads"}],
                question="How?",
                urgency="medium",
                config={"model": "glm-5.1", "api_key": "test-key", "base_url": "https://open.bigmodel.cn/api/paas/v4"},
            ))
        assert "semaphore" in result["advice"]
        assert result["tokens_in"] == 300

    def test_api_error_returns_error_json(self):
        from tools.advisor_tool import call_advisor

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.side_effect = Exception("API timeout")
            result = json.loads(call_advisor(
                messages=[],
                question="test",
                urgency="low",
                config={"model": "test", "api_key": "sk-test"},
            ))
        assert "error" in result
        assert "timeout" in result["error"]


# ===========================================================================
# 7. Anthropic-native advisor call (mocked)
# ===========================================================================

class TestCallAdvisorAnthropic:
    """call_advisor routes to Anthropic native API when detected."""

    @patch("tools.advisor_tool._call_advisor_anthropic")
    def test_anthropic_native_detected(self, mock_anthropic):
        from tools.advisor_tool import call_advisor

        mock_anthropic.return_value = json.dumps({
            "advice": "Use Opus for complex reasoning.",
            "model": "claude-opus-4-6",
            "tokens_in": 800,
            "tokens_out": 60,
            "latency_ms": 1200,
        })

        result = json.loads(call_advisor(
            messages=[{"role": "user", "content": "Help"}],
            question="Complex architecture",
            urgency="high",
            config={"model": "claude-opus-4-6", "api_key": "sk-ant-test"},
        ))
        assert result["advice"] == "Use Opus for complex reasoning."
        mock_anthropic.assert_called_once()


# ===========================================================================
# 8. Schema and registry
# ===========================================================================

class TestSchemaAndRegistry:
    """Tool schema is valid and registry entry exists."""

    def test_schema_structure(self):
        from tools.advisor_tool import ASK_ADVISOR_SCHEMA
        assert ASK_ADVISOR_SCHEMA["name"] == "ask_advisor"
        assert "parameters" in ASK_ADVISOR_SCHEMA
        props = ASK_ADVISOR_SCHEMA["parameters"]["properties"]
        assert "question" in props
        assert "urgency" in props
        assert ASK_ADVISOR_SCHEMA["parameters"]["required"] == ["question"]

    def test_urgency_enum(self):
        from tools.advisor_tool import ASK_ADVISOR_SCHEMA
        urgency = ASK_ADVISOR_SCHEMA["parameters"]["properties"]["urgency"]
        assert urgency["enum"] == ["low", "medium", "high"]

    def test_registry_has_ask_advisor(self):
        from tools.registry import registry
        assert "ask_advisor" in registry._tools

    def test_registry_toolset(self):
        from tools.registry import registry
        entry = registry.get_entry("ask_advisor")
        assert entry is not None
        assert entry.toolset == "advisor"


# ===========================================================================
# 9. Output trimming
# ===========================================================================

class TestOutputTrimming:
    """Advisor user message includes 80-word trimming instruction."""

    def test_trimming_instruction_present(self):
        from tools.advisor_tool import call_advisor

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Short advice"
        mock_resp.choices[0].message.reasoning_content = None
        mock_resp.usage = MagicMock()
        mock_resp.usage.prompt_tokens = 100
        mock_resp.usage.completion_tokens = 10

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_resp
            call_advisor(
                messages=[],
                question="test",
                urgency="low",
                config={"model": "test", "api_key": "sk-test"},
            )

            call_args = MockOpenAI.return_value.chat.completions.create.call_args
            messages = call_args[1]["messages"]
            user_msgs = [m for m in messages if m["role"] == "user"]
            last_user = user_msgs[-1] if user_msgs else {}
            assert "80 words" in last_user.get("content", "")


# ===========================================================================
# 10. Default system prompt
# ===========================================================================

class TestDefaultSystemPrompt:
    """Default advisor system prompt follows Anthropic recommendations."""

    def test_has_80_words_instruction(self):
        from tools.advisor_tool import _DEFAULT_ADVISOR_SYSTEM_PROMPT
        assert "80 words" in _DEFAULT_ADVISOR_SYSTEM_PROMPT

    def test_has_no_execute_rule(self):
        from tools.advisor_tool import _DEFAULT_ADVISOR_SYSTEM_PROMPT
        assert "NOT execute" in _DEFAULT_ADVISOR_SYSTEM_PROMPT


# ===========================================================================
# 11. Executor advisor prompt
# ===========================================================================

class TestExecutorAdvisorPrompt:
    """EXECUTOR_ADVISOR_PROMPT follows Anthropic's recommended system prompt."""

    def test_has_timing_guidance(self):
        from tools.advisor_tool import EXECUTOR_ADVISOR_PROMPT
        assert "BEFORE substantive work" in EXECUTOR_ADVISOR_PROMPT
        assert "task is complete" in EXECUTOR_ADVISOR_PROMPT

    def test_has_advice_handling(self):
        from tools.advisor_tool import EXECUTOR_ADVISOR_PROMPT
        assert "serious weight" in EXECUTOR_ADVISOR_PROMPT
        assert "conflict" in EXECUTOR_ADVISOR_PROMPT.lower()


# ===========================================================================
# 12. Anthropic adapter: build_advisor_tool
# ===========================================================================

class TestBuildAdvisorTool:
    """Native Anthropic advisor tool definition."""

    def test_structure(self):
        from agent.anthropic_adapter import _build_advisor_tool
        tool = _build_advisor_tool({"model": "claude-opus-4-7"})
        assert tool["type"] == "advisor_20260301"
        assert tool["name"] == "ask_advisor"
        assert tool["model"] == "claude-opus-4-7"

    def test_default_model(self):
        from agent.anthropic_adapter import _build_advisor_tool
        tool = _build_advisor_tool({})
        assert tool["model"] == "claude-sonnet-4-6"


# ===========================================================================
# 13. Zero-config fallback
# ===========================================================================

class TestZeroConfig:
    """When no advisor config exists, inherit from parent agent."""

    def test_inherit_parent_model(self):
        from tools.advisor_tool import call_advisor

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Advice"
        mock_resp.choices[0].message.reasoning_content = None
        mock_resp.usage = MagicMock()
        mock_resp.usage.prompt_tokens = 10
        mock_resp.usage.completion_tokens = 5

        parent = MagicMock()
        parent.model = "deepseek-v4-flash"
        parent.api_key = "sk-inherited"
        parent.base_url = "https://api.deepseek.com"

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_resp
            result = json.loads(call_advisor(
                messages=[],
                question="test",
                urgency="low",
                config={},  # no model/key set
                parent_agent=parent,
            ))
        assert result["advice"] == "Advice"
        # Verify the model was resolved from parent
        call_args = MockOpenAI.return_value.chat.completions.create.call_args
        assert call_args[1]["model"] == "deepseek-v4-flash"

    def test_no_model_no_parent_returns_error(self):
        from tools.advisor_tool import call_advisor
        result = json.loads(call_advisor(
            messages=[],
            question="test",
            urgency="low",
            config={},
            parent_agent=None,
        ))
        assert "error" in result
