#include "llm/context/context.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace llm {
namespace {

/// Labels the caller-supplied blocks so the transcript is readable and so
/// "current state" is never confused with "what happened recently".
constexpr char kRecentHistoryLabel[] = "Recent history:";
constexpr char kCurrentContextLabel[] = "Current context:";
constexpr char kDecisionContextLabel[] = "Decision context:";

bool IsEmpty(const std::string& text) {
  return text.find_first_not_of(" \t\r\n") == std::string::npos;
}

/// Approximate cost of one decision as it appears in the transcript.
std::size_t DecisionTokens(const DecisionRecord& record) {
  std::size_t tokens = EstimateTokens(record.decision_context) +
                       EstimateTokens(record.assistant.content);
  for (const CommandCall& call : record.assistant.command_calls) {
    tokens += EstimateTokens(call.name) + EstimateTokens(call.arguments_json);
  }
  for (const Message& result : record.command_results) {
    tokens += EstimateTokens(result.content);
  }
  return tokens;
}

}  // namespace

Context::Context(ContextConfig config) : config_(config) {}

void Context::SetContextUpdate(ContextUpdate update) {
  current_context_ = std::move(update.current_context);
  recent_history_ = std::move(update.recent_history);
  decision_context_ = std::move(update.decision_context);
  context_revision_ = std::move(update.context_revision);
}

void Context::AppendDecision(DecisionRecord record) {
  decisions_.push_back(std::move(record));
  // Memory retention is always bounded: when no explicit cap is set, the
  // render budget is the cap, so a long-running agent cannot grow forever.
  const std::size_t cap = config_.max_memory_decisions > 0
                              ? config_.max_memory_decisions
                              : config_.max_decisions;
  if (cap > 0) {
    while (decisions_.size() > cap) {
      decisions_.erase(decisions_.begin());
    }
  }
}

void Context::ClearDecisions() {
  decisions_.clear();
}

RenderedContext Context::Render(RenderBudget budget) const {
  RenderedContext rendered;

  // The current inputs are mandatory: they are the question this decision
  // answers. They are always rendered; when they alone exceed the message or
  // token budget, exceeds_budget says so and the caller refuses to send.
  const bool has_history = !IsEmpty(recent_history_);
  const bool has_context = !IsEmpty(current_context_);
  const bool has_decision = !IsEmpty(decision_context_);
  const std::size_t trailing_messages = (has_history ? 1u : 0u) +
                                        (has_context ? 1u : 0u) +
                                        (has_decision ? 1u : 0u);
  const std::size_t trailing_tokens =
      (has_history ? EstimateTokens(kRecentHistoryLabel) +
                         EstimateTokens(recent_history_)
                   : 0u) +
      (has_context ? EstimateTokens(kCurrentContextLabel) +
                         EstimateTokens(current_context_)
                   : 0u) +
      (has_decision ? EstimateTokens(kDecisionContextLabel) +
                          EstimateTokens(decision_context_)
                    : 0u);

  const bool budgeted = budget.max_input_tokens > 0;
  const std::size_t fixed = budget.fixed_tokens + trailing_tokens;
  const std::size_t budget_left =
      !budgeted
          ? 0
          : (budget.max_input_tokens > fixed ? budget.max_input_tokens - fixed
                                             : 0);

  // Decisions: newest first, whole records only. The first record that does not
  // fit ends the walk, so the kept history is always a contiguous tail.
  const std::size_t kept_limit =
      std::min(config_.max_decisions, decisions_.size());
  std::vector<const DecisionRecord*> kept;
  kept.reserve(kept_limit);
  std::size_t used_messages = trailing_messages;
  std::size_t used_tokens = 0;
  for (std::size_t i = 0; i < kept_limit; ++i) {
    const DecisionRecord& record = decisions_[decisions_.size() - 1 - i];
    const std::size_t count = record.MessageCount();
    const std::size_t tokens = DecisionTokens(record);

    if (used_messages + count > config_.max_messages) {
      break;  // the message budget drops this and every older decision
    }
    if (budgeted && used_tokens + tokens > budget_left) {
      break;  // drop this and every older decision
    }
    used_messages += count;
    used_tokens += tokens;
    kept.push_back(&record);
    if (budgeted && used_tokens >= budget_left) {
      break;
    }
  }

  rendered.decisions_included = kept.size();
  rendered.decisions_dropped = decisions_.size() - kept.size();

  for (auto it = kept.rbegin(); it != kept.rend(); ++it) {
    const DecisionRecord& record = **it;
    if (!record.decision_context.empty()) {
      rendered.messages.push_back(Message::User(
          std::string(kDecisionContextLabel) + "\n" + record.decision_context));
    }
    rendered.messages.push_back(record.assistant);
    for (const Message& result : record.command_results) {
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
  // The framing for the decision being asked about goes last, so the model
  // answers this question with all of the current state in view.
  if (has_decision) {
    rendered.messages.push_back(Message::User(
        std::string(kDecisionContextLabel) + "\n" + decision_context_));
  }

  // History is expendable: when it does not fit, it is dropped rather than the
  // request being refused. The mandatory inputs are not, so they also enforce
  // max_messages as a strict ceiling.
  rendered.estimated_tokens = fixed + used_tokens;
  rendered.exceeds_budget = trailing_messages > config_.max_messages ||
                            (budgeted && fixed > budget.max_input_tokens);
  return rendered;
}

}  // namespace llm
