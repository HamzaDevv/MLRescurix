use clap::{Parser,Subcommand};
use mlrescurix_harness::{Runtime,Policy,Request,contracts::{decode_frame,MAX_FRAME},execution::{self,JobSpec,Supervisor}};
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
    Supervise { #[arg(long)] state:PathBuf, #[arg(long,default_value_t=1)] workers:usize },
    CreateEnvironment { #[arg(long)] state:PathBuf, #[arg(long)] id:String, #[arg(long)] python:PathBuf },
    RemoveEnvironment { #[arg(long)] state:PathBuf, #[arg(long)] id:String },
    InstallScientificProfile { #[arg(long)] state:PathBuf, #[arg(long)] id:String },
    PinArtifact { #[arg(long)] state:PathBuf, #[arg(long)] project:String, #[arg(long)] job:String, #[arg(long,default_value_t=true)] pinned:bool },
    CollectArtifacts { #[arg(long)] state:PathBuf },
    SetProjectWeight { #[arg(long)] state:PathBuf, #[arg(long)] project:String, #[arg(long)] weight:u8 },
    RegisterDataset { #[arg(long)] state:PathBuf, #[arg(long)] project:String, #[arg(long)] id:String, #[arg(long)] csv:PathBuf },
    Dashboard { #[arg(long)] state:PathBuf, #[arg(long,default_value_t=8765)] port:u16 },
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
        Command::Supervise{state,workers}=>Supervisor::start(&state,workers)?.serve(),
        Command::CreateEnvironment{state,id,python}=>print(execution::create_environment(&state,&id,&python)?),
        Command::RemoveEnvironment{state,id}=>{execution::remove_environment(&state,&id)?;print(serde_json::json!({"id":id,"removed":true}))},
        Command::InstallScientificProfile{state,id}=>print(execution::install_scientific_profile(&state,&id)?),
        Command::PinArtifact{state,project,job,pinned}=>{execution::pin_artifact(&state,&project,&job,pinned)?;print(serde_json::json!({"project":project,"job":job,"pinned":pinned}))},
        Command::CollectArtifacts{state}=>print(serde_json::json!({"reclaimed_bytes":execution::collect_artifacts(&state)?})),
        Command::SetProjectWeight{state,project,weight}=>{execution::set_project_weight(&state,&project,weight)?;print(serde_json::json!({"project":project,"weight":weight}))},
        Command::RegisterDataset{state,project,id,csv}=>print(execution::register_dataset(&state,&project,&id,&csv)?),
        Command::Dashboard{state,port}=>{
            if !(1024..=65535).contains(&port){return Err("dashboard port must be 1024..=65535".into());}
            execution::migrate(&state)?;
            let pythonpath=PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../python/src");
            let binary=std::env::current_exe().map_err(|e|e.to_string())?;
            let status=std::process::Command::new("python3").args(["-m","mlrescurix_worker.scientific.dashboard","--state"])
                .arg(&state).arg("--binary").arg(binary).arg("--port").arg(port.to_string())
                .env_clear().env("PATH","/usr/bin:/bin").env("PYTHONPATH",pythonpath).status().map_err(|e|e.to_string())?;
            if status.success(){Ok(())}else{Err(format!("dashboard exited with {status}"))}
        },
        Command::Worker{spec,output}=>{let result=execution::worker(&spec,&output)?;if result==0 {Ok(())} else {std::process::exit(result)}}
        Command::Capabilities=>print(serde_json::json!({"phase":"M1","protocol":"mlrescurix/1","os":std::env::consts::OS,"arch":std::env::consts::ARCH,"writer_queue":64,"frame_bytes":MAX_FRAME,"trusted_probe_execution":cfg!(target_os="macos"),"scientific_execution":false,"official_evaluation":false,"resource_enforcement":{"worker_count":"enforced","thread_pool_environment_hints":"applied; library dependent","memory":"RSS sampled; quota unsupported","disk":"free-space admission and bounded outputs","cpu":"best-effort process sample"},"providers":false})),
    }).await.map_err(|e|e.to_string())?
}
#[tokio::main(flavor="current_thread")]
async fn main(){if let Err(e)=run().await{eprintln!("{e}");std::process::exit(1);}}
