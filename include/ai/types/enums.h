#pragma once

namespace ai {

enum MessageRole {
  kMessageRoleSystem,
  kMessageRoleUser,
  kMessageRoleAssistant
};

enum FinishReason {
  kFinishReasonStop,
  kFinishReasonLength,
  kFinishReasonContentFilter,
  kFinishReasonToolCalls,
  kFinishReasonError
};
}  // namespace ai