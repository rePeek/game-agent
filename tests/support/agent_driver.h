#pragma once

/// Test-only helpers for driving an owner-thread llm::Agent.
///
/// The Agent never blocks and never stages a decision until the owner polls, so
/// tests start a request and then poll until it is ready.

#include <chrono>
#include <optional>
#include <stdexcept>
#include <thread>

#include <llm/agent.hpp>

namespace llm {
namespace test {

/// Polls until a decision is ready or the timeout expires.
inline std::optional<DecisionResult> PollUntilReady(
    Agent& agent,
    std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    if (std::optional<DecisionResult> result = agent.PollDecision()) {
      return result;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return std::nullopt;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

/// Starts a decision and waits for it, failing the test by throwing when it
/// cannot be started or does not finish.
inline DecisionResult DecideOrThrow(
    Agent& agent,
    std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
  const StartOutcome started = agent.DecideAsync();
  if (!started.ok()) {
    throw std::runtime_error("decision did not start");
  }
  std::optional<DecisionResult> result = PollUntilReady(agent, timeout);
  if (!result.has_value()) {
    throw std::runtime_error("decision did not complete");
  }
  return std::move(*result);
}

/// Waits for a decision that was started with DecideAsync(); throws when it
/// does not finish in time.
inline DecisionResult WaitDecision(
    Agent& agent,
    std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
  std::optional<DecisionResult> result = PollUntilReady(agent, timeout);
  if (!result.has_value()) {
    throw std::runtime_error("decision did not complete");
  }
  return std::move(*result);
}

}  // namespace test
}  // namespace llm
