#include "llm/context.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace llm {
namespace {

/// Labels the two caller-supplied blocks so the transcript is readable and so
/// "current state" is never confused with "what happened recently".
constexpr char kRecentHistoryLabel[] = "Recent history:";
constexpr char kCurrentContextLabel[] = "Current context:";

bool IsEmpty(const std::string& text) {
  return text.find_first_not_of(" \t\r\n") == std::string::npos;
}

}  // namespace

Context::Context(ContextConfig config) : config_(config) {}

void Context::SetSystemPrompt(std::string prompt) {
  system_prompt_ = std::move(prompt);
}

void Context::SetTools(std::vector<ToolDefinition> tools) {
  tools_ = std::move(tools);
}

void Context::SetContextUpdate(ContextUpdate update) {
  current_context_ = std::move(update.current_context);
  recent_history_ = std::move(update.recent_history);
  context_revision_ = std::move(update.context_revision);
}

void Context::AppendDecision(DecisionRecord record) {
  decisions_.push_back(std::move(record));
}

void Context::ClearDecisions() {
  decisions_.clear();
}

RenderedContext Context::Render() const {
  RenderedContext rendered;

  // The fixed part: always sent, even if it alone exceeds the budget — a
  // request without the current state or the system prompt would be useless.
  if (!IsEmpty(system_prompt_)) {
    rendered.messages.push_back(Message::System(system_prompt_));
  }
  const std::size_t fixed_messages = rendered.messages.size();
  const bool has_history = !IsEmpty(recent_history_);
  const bool has_context = !IsEmpty(current_context_);
  const std::size_t trailing_messages =
      (has_history ? 1u : 0u) + (has_context ? 1u : 0u);

  // Decisions: newest first, whole records only. The first record that does
  // not fit ends the walk, so the kept history is always a contiguous tail.
  const std::size_t kept_limit =
      std::min(config_.max_decisions, decisions_.size());
  std::vector<const DecisionRecord*> kept;
  kept.reserve(kept_limit);
  std::size_t used = fixed_messages + trailing_messages;
  for (std::size_t i = 0; i < kept_limit; ++i) {
    const DecisionRecord& record = decisions_[decisions_.size() - 1 - i];
    const std::size_t count = record.MessageCount();
    if (used + count > config_.max_messages) {
      break;
    }
    used += count;
    kept.push_back(&record);
  }

  rendered.decisions_included = kept.size();
  rendered.decisions_dropped = decisions_.size() - kept.size();

  for (auto it = kept.rbegin(); it != kept.rend(); ++it) {
    rendered.messages.push_back((*it)->assistant);
    for (const Message& result : (*it)->tool_results) {
      rendered.messages.push_back(result);
    }
  }

  if (has_history) {
    rendered.messages.push_back(Message::User(std::string(kRecentHistoryLabel) +
                                              "\n" + recent_history_));
  }
  if (has_context) {
    rendered.messages.push_back(Message::User(
        std::string(kCurrentContextLabel) + "\n" + current_context_));
  }

  return rendered;
}

ModelRequest Context::BuildRequest(std::string model) const {
  ModelRequest request;
  request.model = std::move(model);
  request.messages = BuildMessages();
  request.tools = tools_;
  request.tool_choice = ToolChoice::Required();
  return request;
}

}  // namespace llm
