pub mod roles;
pub mod gateway;
pub mod context;
pub mod sandbox;
pub mod workspace;
pub mod tools;
pub mod orchestrator;
pub mod recovery;
pub mod task_contract;

pub use roles::{get_consolidated_default_roles, get_role, get_standard_roles, AgentRoleSpec};
pub use gateway::{
    approve_outbound_view, register_model_pricing, resolve_model_pricing, ChatCompletionRequest,
    ChatCompletionResponse, ChatMessage, FunctionCall, OpenAICompatibleAdapter, OutboundFilter,
    ProviderGateway, ScriptedProviderFixture, TokenUsage, ToolCall, ToolDefinition, ToolFunction,
};
pub use context::{Achievement, ActiveJobReceipt, AgentContextManager, Handoff, ScientificFinding};
pub use sandbox::{SandboxDetector, SandboxExecutionResult, SandboxLimits, SandboxRunner, SandboxStatus};
pub use workspace::BranchWorkspaceManager;
pub use tools::{ToolDescriptor, ToolExecutionReceipt, ToolRegistry};
pub use orchestrator::{AgentOrchestrator, OrchestratorConfig, TurnExecutionResult};
pub use recovery::BranchLock;
pub use task_contract::{
    check_user_approval, get_pending_steering_directives, get_task_assignment, get_task_completion,
    mark_steering_directive_applied, record_steering_directive, record_task_assignment,
    record_task_completion, register_user_approval, SteeringDirectiveRecord, TaskAssignment, TaskCompletionReceipt,
    TaskLimits,
};
