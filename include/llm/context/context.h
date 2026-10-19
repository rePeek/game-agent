#pragma once

/// Message-history management for one decision stream.
///
/// A Context owns everything the model sees, so the application no longer has
/// to reassemble the whole transcript for every decision. The pieces have
/// different lifetimes:
///
/// | piece            | source                | strategy                        |
/// | ---------------- | --------------------- | ------------------------------- |
/// | system prompt    | application           | kept                            |
/// | command defs     | application           | replaced on registration change |
/// | current context  | application, per call | replaces the previous snapshot  |
/// | recent history   | application, per call | replaces the previous window    |
/// | decision context | application, per call | travels with the decision       |
/// | decisions        | confirmed by the SDK  | kept within a budget, oldest first |
///
/// Two rules matter for correctness:
///  * the current context is *state*, not history: it never accumulates;
///  * a decision's user turn, assistant command calls and the command results
///    answering them are stored — and trimmed — as one unit, so the model never
///    sees half a decision.
///
/// The rendering is deterministic: no summarization, no model calls.

#include "llm/core/message.h"
#include "llm/core/request.h"
#include "llm/core/types.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace llm {

/// What the application supplies for one decision round.
struct ContextUpdate {
  /// The situation right now (match state, tactical board, ...). Replaces the
  /// previous snapshot.
  std::string current_context;
  /// A window of recent trends the caller wants the model to see. Replaces the
  /// previous window.
  std::string recent_history;
  /// A short factual summary of the moment this decision is being asked about.
  /// It is snapshotted when the decision starts and stored with the decision
  /// once its results are confirmed, so the transcript shows what each decision
  /// was answering.
  std::string decision_context;
  /// Revision of the game state this update describes. It travels with the
  /// decision so a stale answer can be recognised instead of executed.
  std::string context_revision;
};

/// One confirmed decision: the user turn that framed it, the assistant turn
/// that requested commands and the results the application reported.
struct DecisionRecord {
  std::uint64_t decision_id = 0;
  std::string context_revision;
  /// Framing the application supplied when the decision was taken.
  std::string decision_context;
  Message assistant;                      // command calls (+ any assistant text)
  std::vector<Message> command_results;   // one per command call, in order

  std::size_t MessageCount() const {
    return (decision_context.empty() ? 0u : 1u) + 1 + command_results.size();
  }
};

struct ContextConfig {
  /// Budget for the rendered message list.
  std::size_t max_messages = 64;
  /// How many confirmed decisions to keep at most for rendering.
  std::size_t max_decisions = 16;
  /// Hard cap on the decisions kept in memory. 0 keeps everything that
  /// max_decisions would render. Dropping an old decision here is what makes
  /// long-running agents bounded in memory, not just in request size.
  std::size_t max_memory_decisions = 64;

  /// Approximate input-token ceiling for one request, covering the system
  /// prompt, the command schemas and the rendered messages. 0 disables the
  /// check. The estimate is approximate by design (see EstimateTokens).
  std::size_t max_input_tokens = 0;
  /// Tokens reserved for the model's answer, subtracted from max_input_tokens
  /// before the transcript is trimmed.
  std::size_t reserved_output_tokens = 0;
};

/// What the Context is allowed to spend on its own messages.
struct RenderBudget {
  /// Approximate input token ceiling; 0 disables the check.
  std::size_t max_input_tokens = 0;
  /// Tokens already spent outside the Context (system prompt, command schemas)
  /// and reserved for the answer.
  std::size_t fixed_tokens = 0;
};

/// The rendered transcript plus what the budget did to it.
struct RenderedContext {
  std::vector<Message> messages;
  std::size_t decisions_included = 0;
  std::size_t decisions_dropped = 0;
  /// Approximate tokens of the whole request this render belongs to, including
  /// fixed_tokens. Diagnostic only.
  std::size_t estimated_tokens = 0;
  /// True when even the fixed content and the newest decision do not fit the
  /// budget. The caller must refuse the request rather than send it.
  bool exceeds_budget = false;
};

/// Owns the message history of one decision stream (one coach, one side).
///
/// Thread-compatible but not thread-safe: the Agent serializes access.
class Context {
 public:
  explicit Context(ContextConfig config = {});

  ContextConfig config() const { return config_; }
  void SetConfig(ContextConfig config) { config_ = config; }

  // --- application-owned inputs -------------------------------------------
  /// Records the latest snapshots; all three replace what was there before.
  void SetContextUpdate(ContextUpdate update);
  const std::string& current_context() const { return current_context_; }
  const std::string& recent_history() const { return recent_history_; }
  const std::string& decision_context() const { return decision_context_; }
  /// Revision of the newest update (empty until one is set).
  const std::string& context_revision() const { return context_revision_; }

  // --- confirmed decisions ------------------------------------------------
  /// Appends one complete decision. Contexts store what they are given: the
  /// Agent is what guarantees a record is complete before it lands here. Old
  /// decisions beyond the memory cap are dropped from the front.
  void AppendDecision(DecisionRecord record);
  const std::vector<DecisionRecord>& decisions() const { return decisions_; }
  std::size_t decision_count() const { return decisions_.size(); }
  void ClearDecisions();

  // --- rendering ----------------------------------------------------------
  /// Renders the context: the newest decisions that fit, recent history and the
  /// current snapshot. The owner of the context prepends its system prompt and
  /// adds its commands and model parameters.
  RenderedContext Render(RenderBudget budget = {}) const;
  std::vector<Message> BuildMessages() const { return Render().messages; }

 private:
  ContextConfig config_;
  std::string current_context_;
  std::string recent_history_;
  std::string decision_context_;
  std::string context_revision_;
  std::vector<DecisionRecord> decisions_;  // oldest first
};

}  // namespace llm
