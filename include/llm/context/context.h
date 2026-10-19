#pragma once

/// Message-history management for one decision stream.
///
/// A Context owns everything the model sees, so the application no longer has
/// to reassemble the whole transcript for every decision. The pieces have
/// different lifetimes:
///
/// | piece            | source                | strategy | | ---------------- |
/// --------------------- | ------------------------------ | | system prompt |
/// application           | kept                         | | command definitions |
/// application           | replaced on update             | | current context
/// | application, per call | replaces the previous snapshot | | recent history
/// | application, per call | replaces the previous window   | | decisions |
/// confirmed by the SDK  | kept within a budget, oldest dropped first |
///
/// Two rules matter for correctness:
///  * the current context is *state*, not history: it never accumulates;
///  * an assistant command-call message and the command results answering it are
///    stored — and trimmed — as one unit, so the model never sees half a
///    decision.
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
  /// Revision of the game state this update describes. It travels with the
  /// decision so a stale answer can be recognised instead of executed.
  std::string context_revision;
};

/// One confirmed decision: the assistant turn that requested commands and the
/// results the application reported for them.
struct DecisionRecord {
  std::uint64_t decision_id = 0;
  std::string context_revision;
  Message assistant;                  // command calls (+ any assistant text)
  std::vector<Message> command_results;  // one per command call, in order

  std::size_t MessageCount() const { return 1 + command_results.size(); }
};

struct ContextConfig {
  /// Budget for the rendered message list.
  std::size_t max_messages = 64;
  /// How many confirmed decisions to keep at most.
  std::size_t max_decisions = 16;
};

/// The rendered transcript plus what the budget did to it.
struct RenderedContext {
  std::vector<Message> messages;
  std::size_t decisions_included = 0;
  std::size_t decisions_dropped = 0;
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
  /// Records the latest snapshots; both replace what was there before.
  void SetContextUpdate(ContextUpdate update);
  const std::string& current_context() const { return current_context_; }
  const std::string& recent_history() const { return recent_history_; }
  /// Revision of the newest update (empty until one is set).
  const std::string& context_revision() const { return context_revision_; }

  // --- confirmed decisions ------------------------------------------------
  /// Appends one complete decision. Contexts store what they are given: the
  /// The Agent is what guarantees a record is complete before it lands here.
  void AppendDecision(DecisionRecord record);
  const std::vector<DecisionRecord>& decisions() const { return decisions_; }
  std::size_t decision_count() const { return decisions_.size(); }
  void ClearDecisions();

  // --- rendering ----------------------------------------------------------
  /// Renders the context: the newest decisions that fit, recent history and the
  /// current snapshot. The owner of the context prepends its system prompt and
  /// adds its commands and model parameters.
  RenderedContext Render() const;
  std::vector<Message> BuildMessages() const { return Render().messages; }

 private:
  ContextConfig config_;
  std::string current_context_;
  std::string recent_history_;
  std::string context_revision_;
  std::vector<DecisionRecord> decisions_;  // oldest first
};

}  // namespace llm
