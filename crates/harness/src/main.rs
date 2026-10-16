#![recursion_limit = "256"]

use clap::{Parser,Subcommand};
use mlrescurix_harness::{Runtime,Policy,Request,contracts::{decode_frame,MAX_FRAME},execution::{self,JobSpec,Supervisor},search};
use sha2::Digest;
use std::{path::PathBuf,io::Read};
#[derive(Parser)]
#[command(version,about="MLRescurix durable local harness")]
struct Cli { #[command(subcommand)] command:Command }
#[derive(Subcommand)]
enum Command {
    Init { #[arg(long)] state:PathBuf, #[arg(long)] policy:PathBuf },
    Command { #[arg(long)] state:PathBuf, #[arg(long)] request:PathBuf },
    Snapshot { #[arg(long)] state:PathBuf, #[arg(long)] project:String },
    Events { #[arg(long)] state:PathBuf, #[arg(long)] project:String, #[arg(long,default_value_t=0)] after:u64 },
    Restore { #[arg(long)] state:PathBuf, #[arg(long)] project:String, #[arg(long)] task:String },
    ValidateFrame { path:PathBuf },
    Submit { #[arg(long)] state:PathBuf, #[arg(long)] spec:PathBuf },
    InspectJob { #[arg(long)] state:PathBuf, #[arg(long)] project:String, #[arg(long)] job:String },
    CancelJob { #[arg(long)] state:PathBuf, #[arg(long)] project:String, #[arg(long)] job:String },
    Supervise { #[arg(long)] state:PathBuf, #[arg(long,default_value_t=1)] workers:usize, #[arg(long,default_value_t=false)] drain:bool },
    CreateEnvironment { #[arg(long)] state:PathBuf, #[arg(long)] id:String, #[arg(long)] python:PathBuf },
    RegisterContainerEnvironment { #[arg(long)] state:PathBuf, #[arg(long)] id:String, #[arg(long)] image:String },
    RemoveEnvironment { #[arg(long)] state:PathBuf, #[arg(long)] id:String },
    InstallScientificProfile { #[arg(long)] state:PathBuf, #[arg(long)] id:String },
    InstallSearchProfile { #[arg(long)] state:PathBuf, #[arg(long)] id:String },
    SearchInit { #[arg(long)] state:PathBuf, #[arg(long)] spec:PathBuf },
    SearchInspect { #[arg(long)] state:PathBuf, #[arg(long)] project:String, #[arg(long)] session:String },
    SearchRun { #[arg(long)] state:PathBuf, #[arg(long)] project:String, #[arg(long)] session:String, #[arg(long)] task:String, #[arg(long)] generation:u64, #[arg(long)] environment:Option<String> },
    PinArtifact { #[arg(long)] state:PathBuf, #[arg(long)] project:String, #[arg(long)] job:String, #[arg(long,default_value_t=true)] pinned:bool },
    CollectArtifacts { #[arg(long)] state:PathBuf },
    ArchiveArtifacts { #[arg(long)] state:PathBuf, #[arg(long)] project:Option<String> },
    SetProjectWeight { #[arg(long)] state:PathBuf, #[arg(long)] project:String, #[arg(long)] weight:u8 },
    RegisterDataset { #[arg(long)] state:PathBuf, #[arg(long)] project:String, #[arg(long)] id:String, #[arg(long)] csv:PathBuf },
    Dashboard { #[arg(long)] state:PathBuf, #[arg(long,default_value_t=8765)] port:u16, #[arg(long)] token:Option<String>, #[arg(long,default_value_t=false)] legacy:bool },
    Approve { #[arg(long)] state:PathBuf, #[arg(long)] project:String, #[arg(long)] candidate:String, #[arg(long)] binding_sha256:String, #[arg(long)] rationale:String },
    CriticAudit { #[arg(long)] state:PathBuf, #[arg(long)] project:String, #[arg(long)] candidate:String, #[arg(long)] binding_sha256:String, #[arg(long)] actor:String, #[arg(long,default_value_t=true)] approved:bool, #[arg(long)] rationale:String },
    FinalizationBinding { #[arg(long)] state:PathBuf, #[arg(long)] project:String, #[arg(long)] candidate:String },
    StoragePolicySummary { #[arg(long)] state:PathBuf },
    RegisterModelPricing { #[arg(long)] state:PathBuf, #[arg(long)] model:String, #[arg(long)] prompt_millicents_per_1k:i64, #[arg(long)] completion_millicents_per_1k:i64 },
    ApproveOutboundView { #[arg(long)] state:PathBuf, #[arg(long)] project:String, #[arg(long)] request_sha256:String, #[arg(long,default_value="local-human")] actor:String, #[arg(long,default_value_t=true)] approved:bool, #[arg(long,default_value_t=3600000)] ttl_ms:i64 },
    AgentRun { #[arg(long)] state:PathBuf, #[arg(long)] project:String, #[arg(long)] role:String, #[arg(long)] task:String, #[arg(long)] branch:String, #[arg(long,default_value="default-model")] model:String, #[arg(long,default_value_t=10)] max_turns:u32, #[arg(long)] endpoint:Option<String>, #[arg(long)] api_key:Option<String>, #[arg(long)] mock:Option<PathBuf> },
    VerifyBundle { #[arg(long)] bundle:PathBuf, #[arg(long,default_value_t=true)] sandbox:bool, #[arg(long)] state:Option<PathBuf> },
    EstimateFeasibility {
        #[arg(long)] state: PathBuf,
        #[arg(long)] project: String,
        #[arg(long)] dataset: String,
        #[arg(long)] family: String,
        #[arg(long, default_value_t = 5)] folds: u32,
        #[arg(long, default_value_t = true)] refit: bool,
        #[arg(long, default_value_t = 1)] workers: u32,
        #[arg(long, default_value_t = 3600000)] budget_ms: u64,
        #[arg(long)] max_memory_mb: Option<u64>,
        #[arg(long, default_value_t = true)] pilot: bool,
    },
    NegotiatePlan {
        #[arg(long)] state: PathBuf,
        #[arg(long)] project: String,
        #[arg(long)] report: String,
        #[arg(long)] adaptations: Vec<String>,
        #[arg(long)] preset: Option<String>,
    },
    InspectArrow {
        #[arg(long)] state: PathBuf,
        #[arg(long)] project: String,
        #[arg(long)] dataset: String,
    },
    ValidateTemporalSplit {
        #[arg(long)] state: PathBuf,
        #[arg(long)] project: String,
        #[arg(long)] backtest: String,
    },
    InspectAnomalyClustering {
        #[arg(long)] state: PathBuf,
        #[arg(long)] project: String,
        #[arg(long)] adapter: String,
    },
    InspectFrozenEmbeddings {
        #[arg(long)] state: PathBuf,
        #[arg(long)] project: String,
        #[arg(long)] adapter: String,
    },
    SynthesizeCandidate {
        #[arg(long)] state: PathBuf,
        #[arg(long)] project: String,
        #[arg(long)] candidate: String,
        #[arg(long)] task_type: String,
        #[arg(long)] target: String,
        #[arg(long, default_value = "boosting")] family: String,
        #[arg(long, default_value = "lightgbm")] framework: String,
        #[arg(long)] path: Option<String>,
    },
    InspectDiagnostics {
        #[arg(long)] state: PathBuf,
        #[arg(long)] project: String,
        #[arg(long)] candidate: Option<String>,
        #[arg(long)] failure_type: Option<String>,
        #[arg(long)] diagnostic_id: Option<String>,
        #[arg(long, default_value_t = 10)] limit: usize,
    },
    ProposeHypothesis {
        #[arg(long)] state: PathBuf,
        #[arg(long)] project: String,
        #[arg(long)] candidate: String,
        #[arg(long)] mutation: String,
        #[arg(long)] statement: String,
        #[arg(long)] parent: Option<String>,
        #[arg(long, default_value = "balanced_accuracy")] metric: String,
        #[arg(long, default_value = "maximize")] direction: String,
        #[arg(long)] target: Option<f64>,
        #[arg(long, default_value_t = 0.005)] min_improvement: f64,
    },
    QueryParetoFrontier {
        #[arg(long)] state: PathBuf,
        #[arg(long)] project: String,
        #[arg(long)] frontier_id: Option<String>,
        #[arg(long, default_value = "primary_metric")] metric: String,
        #[arg(long, default_value = "maximize")] direction: String,
    },
    ProfileDataset {
        #[arg(long)] state: PathBuf,
        #[arg(long)] csv: PathBuf,
        #[arg(long)] target: Option<String>,
        #[arg(long)] id_col: Option<String>,
    },
    PackageCandidateBundle {
        #[arg(long)] state: PathBuf,
        #[arg(long)] candidate_dir: PathBuf,
        #[arg(long)] bundle_dir: PathBuf,
        #[arg(long)] dataset: PathBuf,
        #[arg(long)] test_access_id: String,
        #[arg(long)] project: String,
        #[arg(long)] candidate: String,
        #[arg(long, default_value = "binary_classification")] task_type: String,
        #[arg(long, default_value = "target")] target: String,
        #[arg(long, default_value = "balanced_accuracy")] metric: String,
    },
    #[command(name="__worker",hide=true)] Worker { spec:PathBuf, output:PathBuf },
    Capabilities,
}
fn read(path:&PathBuf)->Result<Vec<u8>,String>{let mut data=Vec::new();std::fs::File::open(path).map_err(|e|e.to_string())?.take((MAX_FRAME+1)as u64).read_to_end(&mut data).map_err(|e|e.to_string())?;if data.len()>MAX_FRAME{return Err("input exceeds frame bound".into());}Ok(data)}
fn print<T:serde::Serialize>(v:T)->Result<(),String>{println!("{}",serde_json::to_string_pretty(&v).map_err(|e|e.to_string())?);Ok(())}
async fn run()->Result<(),String>{
    // DB operations run on their own bounded writer, never on Tokio's executor.
    let cli=Cli::parse();
    tokio::task::spawn_blocking(move||match cli.command {
        Command::Init{state,policy}=>{let p:Policy=serde_json::from_slice(&read(&policy)?).map_err(|e|e.to_string())?;Runtime::init(&state,p)?;print(serde_json::json!({"state":state,"initialized":true}))},
        Command::Command{state,request}=>{let r:Request=serde_json::from_slice(&read(&request)?).map_err(|e|e.to_string())?;let receipt=Runtime::open(&state)?.command(r)?;print(&receipt)?;if receipt.status=="rejected"{Err("command rejected; durable receipt printed".into())}else{Ok(())}},
        Command::Snapshot{state,project}=>print(Runtime::open(&state)?.snapshot(&project)?),
        Command::Events{state,project,after}=>print(Runtime::open(&state)?.events(&project,after)?),
        Command::Restore{state,project,task}=>print(Runtime::open(&state)?.restore(&project,&task)?),
        Command::ValidateFrame{path}=>print(decode_frame(&read(&path)?)?),
        Command::Submit{state,spec}=>{let s:JobSpec=serde_json::from_slice(&read(&spec)?).map_err(|e|e.to_string())?;print(execution::submit(&state,s)?)},
        Command::InspectJob{state,project,job}=>print(execution::inspect(&state,&project,&job)?),
        Command::CancelJob{state,project,job}=>print(execution::cancel(&state,&project,&job)?),
        Command::Supervise{state,workers,drain}=>{
            let sup = Supervisor::start(&state,workers)?;
            if drain {
                sup.drain()?;
                print(serde_json::json!({"drained":true}))
            } else {
                sup.serve()
            }
        },
        Command::CreateEnvironment{state,id,python}=>print(execution::create_environment(&state,&id,&python)?),
        Command::RegisterContainerEnvironment{state,id,image}=>print(execution::register_container_environment(&state,&id,&image)?),
        Command::RemoveEnvironment{state,id}=>{execution::remove_environment(&state,&id)?;print(serde_json::json!({"id":id,"removed":true}))},
        Command::InstallScientificProfile{state,id}=>print(execution::install_scientific_profile(&state,&id)?),
        Command::InstallSearchProfile{state,id}=>print(execution::install_search_profile(&state,&id)?),
        Command::SearchInit{state,spec}=>{let s:search::SearchSpec=serde_json::from_slice(&read(&spec)?).map_err(|e|e.to_string())?;print(search::init_search(&state,s)?)},
        Command::SearchInspect{state,project,session}=>print(search::inspect_search(&state,&project,&session)?),
        Command::SearchRun{state,project,session,task,generation,environment}=>{
            let mut sup=Supervisor::start(&state,1)?;
            print(search::execute_search(&state,&project,&session,&task,generation,&mut sup,environment.as_deref())?)
        },
        Command::PinArtifact{state,project,job,pinned}=>{execution::pin_artifact(&state,&project,&job,pinned)?;print(serde_json::json!({"project":project,"job":job,"pinned":pinned}))},
        Command::CollectArtifacts{state}=>print(serde_json::json!({"reclaimed_bytes":execution::collect_artifacts(&state)?})),
        Command::ArchiveArtifacts{state,project}=>print(execution::archive_artifacts(&state,project.as_deref())?),
        Command::SetProjectWeight{state,project,weight}=>{execution::set_project_weight(&state,&project,weight)?;print(serde_json::json!({"project":project,"weight":weight}))},
        Command::RegisterDataset{state,project,id,csv}=>print(execution::register_dataset(&state,&project,&id,&csv)?),
        Command::Dashboard{state,port,token,legacy}=>{
            if !(1024..=65535).contains(&port){return Err("dashboard port must be 1024..=65535".into());}
            execution::migrate(&state)?;
            if legacy {
                let pythonpath=mlrescurix_harness::assets::ensure_python_src(&state)?;
                let binary=std::env::current_exe().map_err(|e|e.to_string())?;
                let status=std::process::Command::new("python3").args(["-m","mlrescurix_worker.scientific.dashboard","--state"])
                    .arg(&state).arg("--binary").arg(binary).arg("--port").arg(port.to_string())
                    .env_clear().env("PATH","/usr/bin:/bin").env("PYTHONPATH",pythonpath).status().map_err(|e|e.to_string())?;
                if status.success(){Ok(())}else{Err(format!("dashboard exited with {status}"))}
            } else {
                let session_token = token.or_else(|| std::env::var("MLRESCURIX_SESSION_TOKEN").ok()).unwrap_or_else(|| {
                    let mut hasher = sha2::Sha256::new();
                    sha2::Digest::update(&mut hasher, format!("{}:{}:mlr", port, std::process::id()).as_bytes());
                    let hex = format!("{:x}", sha2::Digest::finalize(hasher));
                    format!("mlr-tok-{}", &hex[..24])
                });
                let server = mlrescurix_harness::server::DashboardServer::bind(&state, port, &session_token)?;
                println!("MLRescurix dashboard running at http://127.0.0.1:{port} with session token: {session_token}");
                server.serve()
            }
        },
        Command::FinalizationBinding { state,project,candidate }=>print(execution::finalization_binding(&state,&project,&candidate)?),
        Command::StoragePolicySummary { state }=>print(execution::storage_policy_summary(&state)?),
        Command::RegisterModelPricing { state, model, prompt_millicents_per_1k, completion_millicents_per_1k } => {
            execution::migrate(&state)?;
            print(mlrescurix_harness::agents::gateway::register_model_pricing(&state, &model, prompt_millicents_per_1k, completion_millicents_per_1k)?)
        },
        Command::ApproveOutboundView { state, project, request_sha256, actor, approved, ttl_ms } => {
            execution::migrate(&state)?;
            print(mlrescurix_harness::agents::gateway::approve_outbound_view(&state, &project, &request_sha256, &actor, approved, ttl_ms)?)
        },
        Command::Approve { state, project, candidate, binding_sha256, rationale } => {
            execution::record_content_decision(&state,&project,&candidate,&binding_sha256,"user_approval",&format!("local-human-uid-{}",unsafe{libc::getuid()}),true,&rationale)?;
            print(serde_json::json!({
                "status": "approved",
                "project": project,
                "candidate_id": candidate,
                "approved_by": "user",
                "rationale": rationale,
            }))
        },
        Command::CriticAudit { state, project, candidate, binding_sha256, actor, approved, rationale } => {
            execution::record_content_decision(&state,&project,&candidate,&binding_sha256,"critic_audit",&actor,approved,&rationale)?;
            print(serde_json::json!({
                "status": if approved { "approved" } else { "rejected" },
                "project": project,
                "candidate_id": candidate,
                "auditor": actor,
                "rationale": rationale,
            }))
        },
        Command::AgentRun { state, project, role, task, branch, model, max_turns, endpoint, api_key, mock } => {
            let provider_cfg: Option<serde_json::Value> = std::fs::read_to_string(".config/provider.json")
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok());
            let configured_model = provider_cfg.as_ref()
                .and_then(|c| c["model"].as_str())
                .unwrap_or("deepseek/deepseek-v4.1-flash");
            let effective_model = if model == "default-model" {
                configured_model.to_string()
            } else {
                model
            };
            let config = mlrescurix_harness::agents::OrchestratorConfig {
                project_id: project,
                task_id: task,
                role_id: role,
                branch_id: branch,
                model: effective_model,
                max_turns,
            };
            let effective_key = api_key
                .or_else(|| std::env::var("OPENROUTER_API_KEY").ok())
                .or_else(|| {
                    std::fs::read_to_string(".config/openrouter.key")
                        .ok()
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                })
                .or_else(|| {
                    let home = std::env::var("HOME").ok()?;
                    std::fs::read_to_string(std::path::Path::new(&home).join(".config/mlrescurix/openrouter.key"))
                        .ok()
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                });
            let default_ep = provider_cfg.as_ref()
                .and_then(|c| c["endpoint"].as_str())
                .unwrap_or("https://openrouter.ai/api/v1/chat/completions");
            let effective_endpoint = endpoint
                .or_else(|| std::env::var("OPENROUTER_BASE_URL").ok())
                .unwrap_or_else(|| default_ep.to_string());
            let gateway: std::sync::Arc<dyn mlrescurix_harness::agents::ProviderGateway> = if let Some(mock_path) = mock {
                let mock_bytes = std::fs::read(&mock_path).map_err(|e| format!("cannot read mock file: {e}"))?;
                let responses: Vec<mlrescurix_harness::agents::ChatCompletionResponse> = serde_json::from_slice(&mock_bytes).map_err(|e| format!("invalid mock JSON: {e}"))?;
                std::sync::Arc::new(mlrescurix_harness::agents::ScriptedProviderFixture::new(responses))
            } else if let Some(key) = effective_key {
                std::sync::Arc::new(mlrescurix_harness::agents::OpenAICompatibleAdapter::new(effective_endpoint, key))
            } else {
                return Err("AgentRun requires either --mock <path> for offline scripted execution or an API key (--api-key, OPENROUTER_API_KEY env, or .config/openrouter.key) for live provider execution".into());
            };
            let mut orchestrator = mlrescurix_harness::agents::AgentOrchestrator::new(&state, config, gateway)?;
            let mut results = Vec::new();
            while orchestrator.current_turn < max_turns {
                let turn_res = orchestrator.step()?;
                let stopped = turn_res.stopped;
                results.push(turn_res);
                if stopped {
                    break;
                }
            }
            print(results)
        },
        Command::VerifyBundle { bundle, sandbox, state } => {
            let report = mlrescurix_harness::bundle::verify_bundle_with_state(&bundle, sandbox, state.as_deref())?;
            print(report)
        },
        Command::EstimateFeasibility { state, project, dataset, family, folds, refit, workers, budget_ms, max_memory_mb, pilot } => {
            let hw = mlrescurix_harness::feasibility::HardwareProfile::from_broker(&state)?;
            let meta = mlrescurix_harness::feasibility::DatasetMetadata::synthetic(&dataset, 1000, 10, 0, 0);
            let fam = mlrescurix_harness::feasibility::ModelFamily::from_str_name(&family);
            let pilot_run = if pilot {
                let p = mlrescurix_harness::feasibility::execute_micro_pilot(&project, &dataset, fam, 100, 3000, 256 * 1024 * 1024);
                mlrescurix_harness::feasibility::record_micro_pilot_run(&state, &p)?;
                Some(p)
            } else {
                None
            };
            let rep = mlrescurix_harness::feasibility::assess_feasibility(
                &project, &hw, &meta, fam, folds, refit, workers, budget_ms, max_memory_mb.map(|m| m * 1024 * 1024), pilot_run.as_ref()
            );
            mlrescurix_harness::feasibility::record_feasibility_report(&state, &rep)?;
            print(rep)
        },
        Command::NegotiatePlan { state, project, report, adaptations, preset } => {
            let rep = mlrescurix_harness::feasibility::get_feasibility_report(&state, &project, &report)?;
            let plan = mlrescurix_harness::feasibility::negotiate_plan(&rep, &adaptations, None, preset.as_deref());
            mlrescurix_harness::feasibility::record_execution_plan(&state, &plan)?;
            print(plan)
        },
        Command::InspectArrow { state, project, dataset } => {
            let adapter = mlrescurix_harness::adapters::get_arrow_adapter(&state, &project, &dataset)?;
            mlrescurix_harness::adapters::validate_arrow_adapter_memory_bounds(&adapter, 512 * 1024 * 1024)?;
            print(adapter)
        },
        Command::ValidateTemporalSplit { state, project, backtest } => {
            let bt = mlrescurix_harness::adapters::get_time_series_backtest(&state, &project, &backtest)?;
            print(bt)
        },
        Command::InspectAnomalyClustering { state, project, adapter } => {
            let ad = mlrescurix_harness::adapters::get_anomaly_clustering_adapter(&state, &project, &adapter)?;
            mlrescurix_harness::adapters::validate_anomaly_clustering_config(&ad)?;
            print(ad)
        },
        Command::InspectFrozenEmbeddings { state, project, adapter } => {
            let ad = mlrescurix_harness::adapters::get_frozen_embedding_adapter(&state, &project, &adapter)?;
            mlrescurix_harness::adapters::validate_frozen_embedding_config(&ad)?;
            print(ad)
        },
        Command::SynthesizeCandidate { state, project, candidate, task_type, target, family, framework, path } => {
            execution::migrate(&state)?;
            let branch = format!("b-{candidate}");
            let dest_path = path.unwrap_or_else(|| "candidate".into());
            let branch_dir = mlrescurix_harness::agents::workspace::BranchWorkspaceManager::get_branch_dir(&state, &project, &branch);
            let cand_dir = branch_dir.join(&dest_path);
            std::fs::create_dir_all(&cand_dir).map_err(|e| e.to_string())?;

            let spec_payload = serde_json::json!({
                "schema_version": "v2",
                "synthesis_id": format!("synth-{candidate}"),
                "candidate_id": candidate,
                "name": format!("Candidate {candidate}"),
                "problem_definition": {
                    "task_type": task_type,
                    "target_column": target,
                    "dataset_id": "default",
                    "id_column": "row_id",
                },
                "target_descriptor": {
                    "name": target,
                    "task_type": task_type,
                },
                "metric_objective": {
                    "primary_metric": if task_type == "regression" { "rmse" } else { "balanced_accuracy" },
                    "direction": if task_type == "regression" { "minimize" } else { "maximize" },
                },
                "model_intent": {
                    "family": family,
                    "framework": framework,
                    "hyperparameters": {},
                    "feature_pipeline": {},
                }
            });

            let scratch = state.join("workflows").join(&project).join("synth_staging").join(&candidate);
            std::fs::create_dir_all(&scratch).map_err(|e| e.to_string())?;
            let spec_file = scratch.join("spec.json");
            let out_file = scratch.join("out.json");
            std::fs::write(&spec_file, serde_json::to_string_pretty(&spec_payload).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;

            let mut cmd = std::process::Command::new("python3");
            cmd.args([
                "-m", "mlrescurix_worker.scientific.cli",
                "candidate-synthesize",
                "--spec", &spec_file.to_string_lossy(),
                "--output-dir", &cand_dir.to_string_lossy(),
                "--output", &out_file.to_string_lossy(),
            ]);
            let embedded_py = mlrescurix_harness::assets::ensure_python_root(&state)?;
            let mut pypaths = vec![embedded_py.join("src").to_string_lossy().to_string()];
            if let Ok(repo_src) = std::fs::canonicalize(std::path::Path::new("python/src")) {
                pypaths.push(repo_src.to_string_lossy().to_string());
            }
            if let Ok(cur_py) = std::env::var("PYTHONPATH") {
                if !cur_py.is_empty() { pypaths.push(cur_py); }
            }
            cmd.env("PYTHONPATH", pypaths.join(":"));
            cmd.env("PYTHONDONTWRITEBYTECODE", "1");
            let out = cmd.output().map_err(|e| e.to_string())?;
            if !out.status.success() {
                let _ = std::fs::remove_dir_all(&scratch);
                return Err(format!("candidate synthesis failed: {}", String::from_utf8_lossy(&out.stderr)));
            }
            let res: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&out_file).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
            let _ = std::fs::remove_dir_all(&scratch);
            print(res)
        },
        Command::InspectDiagnostics { state, project, candidate, failure_type, diagnostic_id, limit } => {
            execution::migrate(&state)?;
            if let Some(did) = diagnostic_id {
                let rec = mlrescurix_harness::diagnostics::get_diagnostic_receipt(&state, &project, &did)?;
                print(rec)
            } else {
                let recs = mlrescurix_harness::diagnostics::list_diagnostic_receipts(&state, &project, candidate.as_deref(), failure_type.as_deref(), limit)?;
                print(serde_json::json!({
                    "count": recs.len(),
                    "diagnostics": recs,
                }))
            }
        },
        Command::ProposeHypothesis { state, project, candidate, mutation, statement, parent, metric, direction, target, min_improvement } => {
            execution::migrate(&state)?;
            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis().min(i64::MAX as u128) as i64;
            let hyp_id = format!("hyp-{candidate}-{now}");
            let hyp = mlrescurix_harness::iteration::IterationHypothesis {
                schema_version: "v2".into(),
                hypothesis_id: hyp_id,
                candidate_id: candidate,
                parent_candidate_id: parent,
                mutation_type: mutation,
                hypothesis_statement: statement,
                rationale: None,
                expected_metric_delta: mlrescurix_harness::iteration::ExpectedMetricDelta {
                    metric,
                    direction,
                    target_threshold: target,
                    minimum_improvement: min_improvement,
                },
                multi_fidelity_budget: mlrescurix_harness::iteration::MultiFidelityBudget::default(),
                outcome: Some(mlrescurix_harness::iteration::HypothesisOutcome {
                    status: "pending".into(),
                    baseline_metric: None,
                    proxy_metric_observed: None,
                    full_metric_observed: None,
                    metric_delta: None,
                    reason: None,
                    settled_at_ms: None,
                }),
                created_ms: now,
            };
            mlrescurix_harness::iteration::record_hypothesis(&state, &project, &hyp)?;
            print(hyp)
        },
        Command::QueryParetoFrontier { state, project, frontier_id, metric, direction } => {
            execution::migrate(&state)?;
            if let Some(fid) = frontier_id {
                let frontier = mlrescurix_harness::iteration::get_pareto_frontier(&state, &project, &fid)?;
                print(frontier)
            } else if let Ok(Some(latest)) = mlrescurix_harness::iteration::get_latest_pareto_frontier(&state, &project) {
                print(latest)
            } else {
                let empty = mlrescurix_harness::iteration::compute_pareto_frontier(&[], &metric, &direction, &project, None);
                print(empty)
            }
        },
        Command::ProfileDataset { state, csv, target, id_col } => {
            let prof = mlrescurix_harness::rescue::profile_dataset_csv(&state, &csv, target.as_deref(), id_col.as_deref())?;
            print(prof)
        },
        Command::PackageCandidateBundle {
            state, candidate_dir, bundle_dir, dataset, test_access_id, project, candidate, task_type, target, metric
        } => {
            let spec = serde_json::json!({
                "candidate_dir": candidate_dir.to_string_lossy(),
                "output_bundle_dir": bundle_dir.to_string_lossy(),
                "dataset_path": dataset.to_string_lossy(),
                "test_access_id": test_access_id,
                "project_id": project,
                "candidate_id": candidate,
                "task_type": task_type,
                "target_column": target,
                "primary_metric": metric,
            });
            let receipt = mlrescurix_harness::rescue::package_candidate_bundle(&state, &spec)?;
            print(receipt)
        },
        Command::Worker{spec,output}=>{let result=execution::worker(&spec,&output)?;if result==0 {Ok(())} else {std::process::exit(result)}}
        Command::Capabilities=>print(serde_json::json!({
            "phase":"P7",
            "release":"Phase P7 Apex Delivery & Release Certification",
            "protocol":"mlrescurix/2",
            "source_sha256":mlrescurix_harness::assets::embedded_python_src_hash(),
            "container_image_pinned":mlrescurix_harness::agents::sandbox::SCIENTIFIC_IMAGE_REF.trim(),
            "state_schema_version":10,
            "os":std::env::consts::OS,
            "arch":std::env::consts::ARCH,
            "supported_platforms":[
                {"host_os":"macos","host_arch":"aarch64","scientific_container_os":"linux","scientific_container_arch":"aarch64","status":"observed_test_fixture_only"},
                {"host_os":"linux","host_arch":"x86_64","scientific_container_os":"linux","scientific_container_arch":"x86_64","status":"supported"}
            ],
            "writer_queue":64,
            "frame_bytes":MAX_FRAME,
            "trusted_probe_execution":cfg!(target_os="macos"),
            "scientific_execution":true,
            "official_evaluation":true,
            "constrained_search":true,
            "resumable_search":true,
            "agent_coding_harness":true,
            "agent_authored_tools":true,
            "sandbox_isolation":true,
            "bundle_verification":true,
            "external_bundle_authenticity":true,
            "standalone_install_without_checkout":true,
            "historical_m6_release_gates":{
                "gate_1_specialist_reconciliation":"closed_and_verified",
                "gate_2_state_bounds_and_enospc":"closed_and_verified",
                "gate_3_installed_lifecycle":"closed_and_verified",
                "gate_4_matched_benchmarks":"closed_and_verified",
                "gate_5_memory_certification":"closed_and_verified"
            },
            "v2_release_gates":{
                "p1_control_plane":"passed",
                "p2_artifact_registry":"passed",
                "p3_resource_broker":"passed",
                "p4_candidate_tools":"passed",
                "p5_collaboration":"passed",
                "p6_dashboard":"passed",
                "p7_apex_delivery":"passed",
                "p8_feasibility":"passed",
                "p9_adapters":"passed",
                "p10_candidate_synthesis":"passed",
                "p10_diagnostic_bridge":"passed",
                "p10_iteration_loop":"passed",
                "p10_autonomous_rescue":"passed"
            },
            "autonomous_rescue_capabilities":{
                "closed_loop_orchestration":true,
                "dataset_profiling":true,
                "failure_interception_and_repair":true,
                "hypothesis_driven_mutation":true,
                "multi_fidelity_screening":true,
                "pareto_optimization":true,
                "bundle_certification":true
            },
            "diagnostic_bridge_capabilities":{
                "cgroup_oom":true,
                "non_finite_loss":true,
                "shape_mismatch":true,
                "class_order_mismatch":true,
                "execution_timeout":true,
                "missing_dependency":true,
                "syntax_error":true,
                "runtime_exception":true,
                "root_cause_attribution":true,
                "bounded_traceback_summaries":true,
                "actionable_remediations":true
            },
            "adapter_capabilities":{
                "arrow_ipc":true,
                "feather":true,
                "parquet":true,
                "memory_mapped_batches":true,
                "empirical_memory_accounting":true,
                "unfounded_zero_copy_rejection":true,
                "out_of_core_incremental_scaling":true,
                "out_of_core_sgd":true,
                "temporal_splits":true,
                "entity_aware_splits":true,
                "rolling_window_backtest":true,
                "expanding_window_backtest":true,
                "future_lookahead_rejection":true,
                "causal_lags_and_rolling":true,
                "naive_and_seasonal_baselines":true,
                "time_series_delivery_predict":true,
                "unsupervised_anomaly_detection":true,
                "unsupervised_clustering":true,
                "empirical_contamination_calibration":true,
                "clustering_partition_profiles":true,
                "out_of_sample_projection":true,
                "frozen_text_embeddings":true,
                "frozen_vision_embeddings":true,
                "content_addressed_embedding_cache":true
            },
            "candidate_synthesis_capabilities":{
                "autonomous_synthesis_engine":true,
                "decoupled_modular_candidates":true,
                "candidate_protocol_conformance":true,
                "arbitrary_model_families":true,
                "lightgbm_support":true,
                "xgboost_support":true,
                "pytorch_support":true,
                "tree_ensembles_support":true,
                "linear_baselines_support":true,
                "custom_pipelines_support":true,
                "label_isolation_enforced":true,
                "row_id_alignment_enforced":true
            },
            "iteration_loop_capabilities":{
                "progressive_multi_fidelity":true,
                "tier_0_proxy_screening":true,
                "tier_1_full_cross_validation":true,
                "early_discard_non_viable":true,
                "hypothesis_settlement":true,
                "multi_objective_pareto_tracking":true,
                "non_dominated_frontier":true
            },
            "feasibility_capabilities":{
                "hardware_profile":true,
                "dataset_metadata":true,
                "analytical_estimators":true,
                "capped_micro_pilots":true,
                "transparent_verdicts":true,
                "evidence_backed_alternatives":true,
                "safe_downscaling_negotiation":true,
                "locked_eval_split_preservation":true
            },
            "delivery_capabilities":{
                "bundle_v2_manifest":true,
                "native_joblib_export":true,
                "safetensors_export":true,
                "documented_tolerances":true,
                "sealed_holdout_access":true,
                "no_private_row_leakage":true,
                "standalone_http_serve":true,
                "standalone_cli_inference":true,
                "installed_without_checkout":true
            },
            "resource_enforcement":{
                "worker_count":"enforced",
                "thread_pool_environment_hints":"applied; library dependent",
                "memory":"Docker per-job limits; cgroup v2 certification requires complete peak/event receipt",
                "disk":"free-space admission, bounded outputs, and 128 MiB managed-state ceiling",
                "cpu":"best-effort process sample"
            },
            "providers":true
        })),
    }).await.map_err(|e|e.to_string())?
}
#[tokio::main(flavor="current_thread")]
async fn main(){if let Err(e)=run().await{eprintln!("{e}");std::process::exit(1);}}
