pub mod roles;
pub mod gateway;
pub mod context;
pub mod sandbox;
pub mod workspace;
pub mod tools;
pub mod orchestrator;

pub use roles::{get_role, get_standard_roles, AgentRoleSpec};
pub use gateway::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, FunctionCall, OpenAICompatibleAdapter,
    OutboundFilter, ProviderGateway, ScriptedProviderFixture, TokenUsage, ToolCall, ToolDefinition,
    ToolFunction,
};
pub use context::{Achievement, ActiveJobReceipt, AgentContextManager, Handoff, ScientificFinding};
pub use sandbox::{SandboxDetector, SandboxExecutionResult, SandboxLimits, SandboxRunner, SandboxStatus};
pub use workspace::BranchWorkspaceManager;
pub use tools::{ToolDescriptor, ToolExecutionReceipt, ToolRegistry};
pub use orchestrator::{AgentOrchestrator, OrchestratorConfig, TurnExecutionResult};
