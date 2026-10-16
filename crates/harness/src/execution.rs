//! M1 trusted local execution kernel. Jobs are typed probes, never shell commands.
use crate::{Result, core::{Policy,Action}};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, io::{Read,Write}, os::fd::AsRawFd, os::unix::{fs::{OpenOptionsExt,PermissionsExt},process::CommandExt}, path::{Path,PathBuf}, process::{Child,ChildStdin,Command,Stdio}, sync::{Arc,atomic::{AtomicBool,Ordering}}, thread, time::{Duration,Instant,SystemTime,UNIX_EPOCH}};

const MAX_FRAME: usize = 65_536;
const POLL_MS: u64 = 100;
const LOG_CAP: u64 = 1_048_576;
pub const OUTPUT_CAP: u64 = 4_194_304;
const DISK_HEADROOM: u64 = 64 * 1024 * 1024;
const OWNER_MARKER: &str = "mlrescurix-owned-v1\n";
static STOP: AtomicBool = AtomicBool::new(false);
#[path = "execution_resources.rs"]
pub mod resources;
extern "C" fn stop_signal(_:libc::c_int){STOP.store(true,Ordering::Relaxed);}

#[derive(Debug,Clone,Serialize,Deserialize,PartialEq)]
#[serde(tag="kind",rename_all="snake_case",deny_unknown_fields)]
pub enum Probe {
    Sleep { millis: u64 },
    Fail { code: u8 },
    Write { name: String, content: String },
    Scientific { operation: String, dataset_id: String, spec: Option<Value>, candidate_id: Option<String>, test_access_id: Option<String> },
}

#[derive(Debug,Clone,Serialize,Deserialize,PartialEq)]
#[serde(deny_unknown_fields)]
pub struct JobSpec {
    pub job_id: String,
    pub project_id: String,
    pub task_id: String,
    pub generation: u64,
    pub timeout_ms: u64,
    pub threads: u32,
    pub max_output_bytes: u64,
    #[serde(default)] pub environment_id: Option<String>,
    #[serde(default, skip_serializing_if = "resources::ResourceRequirements::is_default")]
    pub resources: resources::ResourceRequirements,
    pub probe: Probe,
}

#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct JobView { pub spec: JobSpec, pub status: String, pub reason: Option<String>, pub pid: Option<u32>, pub submitted_ms: i64, pub started_ms: Option<i64>, pub ended_ms: Option<i64>, pub artifacts: Vec<Value>, pub telemetry: Option<ResourceSample>, pub resource_receipt: Option<Value> }
#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct ResourceSample {pub sampled_ms:i64,pub cpu_percent:Option<f64>,pub rss_bytes:Option<u64>,pub peak_rss_bytes:u64}

fn now_ms() -> i64 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis().min(i64::MAX as u128) as i64 }
fn sql<T>(x:rusqlite::Result<T>)->Result<T>{x.map_err(|e|e.to_string())}
fn hash(data:&[u8])->String{format!("{:x}",Sha256::digest(data))}
fn valid_id(s:&str)->Result<()>{if s.is_empty()||s.len()>128||!s.bytes().all(|b|b.is_ascii_alphanumeric()||b"_.:-".contains(&b)){Err("invalid job identity".into())}else{Ok(())}}
pub fn connect(root:&Path)->Result<Connection>{
    if fs::symlink_metadata(root).map_err(|e|e.to_string())?.file_type().is_symlink(){return Err("state root cannot be a symlink".into());}
    if fs::read_to_string(root.join("OWNED")).map_err(|e|e.to_string())?!=OWNER_MARKER{return Err("unowned state".into());}
    for name in ["state.sqlite","state.sqlite-wal","state.sqlite-shm","OWNED"] {if fs::symlink_metadata(root.join(name)).is_ok_and(|m|m.file_type().is_symlink()){return Err("state file symlink rejected".into());}}
    let mut c=sql(Connection::open(root.join("state.sqlite")))?;
    sql(c.busy_timeout(Duration::from_secs(10)))?;
    // Journal mode is persistent after initialization. Reissuing the mode
    // change on every connection takes a database lock and breaks concurrent
    // scheduler/GC access despite WAL already being active.
    sql(c.execute_batch("PRAGMA foreign_keys=ON; PRAGMA synchronous=FULL;"))?;
    loop {
        let version:i64=sql(c.query_row("PRAGMA user_version",[],|r|r.get(0)))?;
        if version==10 { break; }
        if version==6 { sql(c.execute_batch("PRAGMA foreign_keys=OFF;"))?; }
        let tx=sql(c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate))?;
        let confirmed:i64=sql(tx.query_row("PRAGMA user_version",[],|r|r.get(0)))?;
        if confirmed!=version {tx.rollback().map_err(|e|e.to_string())?;sql(c.execute_batch("PRAGMA foreign_keys=ON;"))?;continue;}
        match version {
            1=>sql(tx.execute_batch(include_str!("schema_m1.sql")))?,
            2=>sql(tx.execute_batch(include_str!("schema_m2.sql")))?,
            3=>sql(tx.execute_batch(include_str!("schema_m3.sql")))?,
            4=>sql(tx.execute_batch(include_str!("schema_m4.sql")))?,
            5=>sql(tx.execute_batch(include_str!("schema_m5.sql")))?,
            6=>sql(tx.execute_batch(include_str!("schema_m6.sql")))?,
            7=>sql(tx.execute_batch(include_str!("schema_m6_completion.sql")))?,
            8=>sql(tx.execute_batch(include_str!("schema_m6_recovery.sql")))?,
            9=>sql(tx.execute_batch(include_str!("schema_m7_artifacts.sql")))?,
            _=>return Err(format!("unsupported database migration version {version}")),
        }
        let violation: bool = sql(tx.query_row("SELECT EXISTS(SELECT 1 FROM pragma_foreign_key_check)", [], |r| r.get(0)))?;
        if violation { return Err("migration would leave invalid foreign keys".into()); }
        sql(tx.commit())?;
        sql(c.execute_batch("PRAGMA foreign_keys=ON;"))?;
    }
    sql(c.execute_batch(include_str!("schema_p2_events.sql")))?;
    sql(c.execute_batch(include_str!("schema_p3_leases.sql")))?;
    sql(c.execute_batch(include_str!("schema_p4_candidates.sql")))?;
    sql(c.execute_batch(include_str!("schema_p5_orchestration.sql")))?;
    sql(c.execute_batch(include_str!("schema_p8_feasibility.sql")))?;
    sql(c.execute_batch(include_str!("schema_p9_adapters.sql")))?;
    sql(c.execute_batch(include_str!("schema_p9_anomaly.sql")))?;
    sql(c.execute_batch(include_str!("schema_p9_embeddings.sql")))?;
    sql(c.execute_batch(include_str!("schema_p10_synthesis.sql")))?;
    sql(c.execute_batch(include_str!("schema_p10_diagnostics.sql")))?;
    sql(c.execute_batch(include_str!("schema_p10_iteration.sql")))?;
    sql(c.execute_batch(include_str!("schema_p10_rescue.sql")))?;
    let has_old_recipe_check: bool = sql(c.query_row(
        "SELECT (sql LIKE '%CHECK(recipe IN (''dummy'',''linear'',''tree'')%' OR sql LIKE '%CHECK(recipe IN (''dummy'',''linear'',''tree'',''hist_boosting'')%') AND sql NOT LIKE '%length(recipe)%' FROM sqlite_master WHERE type='table' AND name='experiment_dag'",
        [],
        |r| r.get(0),
    )).unwrap_or(false);
    if has_old_recipe_check {
        sql(c.execute_batch("
            PRAGMA foreign_keys=OFF;
            CREATE TABLE experiment_dag_p4(
              project TEXT NOT NULL REFERENCES projects(id),
              session_id TEXT NOT NULL,
              id TEXT NOT NULL,
              recipe TEXT NOT NULL CHECK(recipe IN ('dummy','linear','tree','hist_boosting') OR length(recipe) <= 128),
              parameters TEXT NOT NULL,
              seed INTEGER NOT NULL,
              fidelity REAL NOT NULL CHECK(fidelity > 0.0 AND fidelity <= 1.0),
              node_type TEXT NOT NULL CHECK(node_type IN ('baseline','screening','full_fidelity','repeat','draft','debug','improve','ablate','candidate')),
              parent_id TEXT,
              created_ms INTEGER NOT NULL,
              PRIMARY KEY(project,id),
              FOREIGN KEY(project,session_id) REFERENCES search_sessions(project,id)
            );
            INSERT INTO experiment_dag_p4 SELECT * FROM experiment_dag;
            DROP TABLE experiment_dag;
            ALTER TABLE experiment_dag_p4 RENAME TO experiment_dag;
            PRAGMA foreign_keys=ON;
        "))?;
    }
    sql(c.execute_batch("CREATE TABLE IF NOT EXISTS tombstones(
        project TEXT NOT NULL REFERENCES projects(id),
        entity_kind TEXT NOT NULL CHECK(entity_kind IN('job','artifact','action')),
        entity_id TEXT NOT NULL,
        parent_id TEXT NOT NULL,
        identity_hash TEXT NOT NULL,
        reclaimed_bytes INTEGER NOT NULL,
        details_json TEXT NOT NULL,
        tombstoned_ms INTEGER NOT NULL,
        PRIMARY KEY(project, entity_kind, entity_id)
    );
    CREATE INDEX IF NOT EXISTS tombstones_project_kind ON tombstones(project, entity_kind, tombstoned_ms);
    CREATE TABLE IF NOT EXISTS action_fences(
        project TEXT NOT NULL REFERENCES projects(id),
        action_id TEXT NOT NULL,
        task_id TEXT NOT NULL,
        generation INTEGER NOT NULL,
        intent_revision INTEGER NOT NULL,
        lease_id TEXT NOT NULL,
        input_hash TEXT NOT NULL,
        status TEXT NOT NULL,
        receipt_json TEXT,
        tombstoned_ms INTEGER NOT NULL,
        PRIMARY KEY(project, action_id)
    );
    CREATE INDEX IF NOT EXISTS action_fences_task ON action_fences(project, task_id);"))?;
    Ok(c)
}
fn ensure_dir(root:&Path,path:&Path)->Result<()>{
    if !path.starts_with(root){return Err("managed directory escapes state root".into());}
    let relative=path.strip_prefix(root).map_err(|e|e.to_string())?;let mut current=root.to_path_buf();
    for part in relative.components(){match part{std::path::Component::Normal(name)=>current.push(name),_=>return Err("invalid managed directory component".into())}
        match fs::symlink_metadata(&current){Ok(m) if m.file_type().is_symlink()||!m.is_dir()=>return Err("managed directory is not a real directory".into()),Ok(_)=>{},Err(e) if e.kind()==std::io::ErrorKind::NotFound=>fs::create_dir(&current).map_err(|e|e.to_string())?,Err(e)=>return Err(e.to_string())}}
    Ok(())
}

pub fn validate_spec(s:&JobSpec)->Result<()> {
    s.resources.validate()?;
    for x in [&s.job_id,&s.project_id,&s.task_id] {valid_id(x)?;}
    if s.job_id.len()>124||s.generation==0||s.timeout_ms==0||s.timeout_ms>86_400_000||s.threads==0||s.threads>256 {return Err("invalid job bounds".into());}
    if s.max_output_bytes>OUTPUT_CAP {return Err("output cap exceeds M1 ceiling".into());}
    match &s.probe {
        Probe::Sleep{millis} if *millis>86_400_000=>return Err("probe duration exceeds ceiling".into()),
        Probe::Fail{code} if *code==0=>return Err("failure probe needs nonzero exit code".into()),
        Probe::Write{name,content}=>{
            if name.is_empty()||name.len()>128||name=="."||name==".."||name.contains('/')||name.contains('\\') {return Err("artifact name must be one safe path component".into());}
            if content.len() as u64>s.max_output_bytes {return Err("artifact exceeds requested output bound".into());}
        }, _=>{}
    }
    if let Probe::Scientific{operation,dataset_id,spec,candidate_id,test_access_id}=&s.probe {
        valid_id(dataset_id)?;
        if !matches!(operation.as_str(),"audit"|"baselines"|"finalize"|"trial"|"pipeline"){return Err("unsupported scientific operation".into());}
        if *operation=="audit" && spec.is_none(){return Err("audit requires an immutable project spec".into());}
        if *operation=="trial" && spec.is_none(){return Err("trial requires an experiment spec".into());}
        if *operation=="pipeline" {
            let value=spec.as_ref().ok_or("pipeline requires a protocol")?;
            valid_id(value["experiment_id"].as_str().ok_or("pipeline requires candidate identity")?)?;
            if value["format"]!="mlrescurix.pipeline.v1" {return Err("unsupported pipeline protocol".into());}
        }
        if *operation=="finalize" {
            let candidate=candidate_id.as_deref().ok_or("finalize requires candidate identity")?;let access=test_access_id.as_deref().ok_or("finalize requires test access identity")?;
            valid_id(candidate)?;valid_id(access)?;
        }
        if s.environment_id.is_none(){return Err("scientific jobs require a managed scientific environment".into());}
    }
    if let Some(e)=&s.environment_id {valid_id(e)?;}
    let encoded=serde_json::to_vec(s).map_err(|e|e.to_string())?;
    if encoded.len()>MAX_FRAME{return Err("job spec exceeds frame bound".into());} Ok(())
}

pub fn migrate(root:&Path)->Result<()> {let _=connect(root)?;Ok(())}

/// Register an already-installed local OCI image; never pull or run host Python.
pub fn register_container_environment(root:&Path,id:&str,image:&str)->Result<Value>{
    valid_id(id)?;
    let digest=image.rsplit_once("@sha256:").map(|(_,d)|d).unwrap_or("");
    if digest.len()!=64||!digest.bytes().all(|b|b.is_ascii_hexdigit()){return Err("scientific container requires a sha256-pinned image".into());}
    let stage=tempfile_for_environment(root,id)?;
    let result=(||{
        fs::write(stage.join("verify.py"),r#"
import sys,importlib.metadata,json,subprocess
from pathlib import Path
assert sys.version_info[:2]==(3,13)
lock=Path('/opt/mlrescurix/requirements.lock').read_text()
versions={}
for line in lock.splitlines():
    if not line or line.startswith('#'): continue
    name,version=line.split(' --hash=sha256:')[0].split('==')
    assert importlib.metadata.version(name)==version
    versions[name]=version
assert {k:versions[k] for k in ['numpy','pandas','scipy','scikit-learn','joblib']}=={'numpy':'2.2.6','pandas':'2.3.0','scipy':'1.16.1','scikit-learn':'1.8.0','joblib':'1.5.1'}
subprocess.run([sys.executable,'-m','pip','check'],check=True,capture_output=True)
print(json.dumps({'python':sys.version,'versions':versions,'lock':lock}))
"#).map_err(|e|e.to_string())?;
        let output=crate::agents::sandbox::SandboxRunner::execute(&stage,"verify.py",&[],&crate::agents::sandbox::SandboxLimits::default(),Some(image))?;
        if output.exit_code!=0 {return Err(format!("container dependency verification failed: {}",output.stderr));}
        let evidence:Value=serde_json::from_str(output.stdout.trim()).map_err(|e|e.to_string())?;
        let identity=hash(format!("{image}|{evidence}").as_bytes());let c=connect(root)?;
        let previous:Option<String>=sql(c.query_row("SELECT identity FROM environments WHERE id=?1",[id],|r|r.get(0)).optional())?;
        if let Some(old)=previous {if old!=identity{return Err("environment id has different managed content".into());}}
        let path=root.join("environments").join(id);ensure_dir(root,&path)?;
        fs::write(path.join("container.json"),json!({"image":image,"identity":identity,"evidence":evidence}).to_string()).map_err(|e|e.to_string())?;
        sql(c.execute("INSERT INTO environments(id,python,identity,path,status) VALUES(?1,?2,?3,?4,'ready') ON CONFLICT(id) DO NOTHING",params![id,format!("docker:{image}"),identity,path.to_string_lossy()]))?;
        Ok(json!({"id":id,"image":image,"identity":identity,"evidence":evidence,"status":"ready"}))
    })();
    fs::remove_dir_all(stage).map_err(|e|e.to_string())?;result
}
fn tempfile_for_environment(root:&Path,id:&str)->Result<PathBuf>{
    let path=root.join("environments").join(format!(".{id}-verify-{}",now_ms()));ensure_dir(root,&path)?;Ok(path)
}

/// Copy an explicitly selected CSV into the owned dataset store before scientific work.
pub fn register_dataset(root:&Path,project:&str,id:&str,source:&Path)->Result<Value>{
    valid_id(project)?;valid_id(id)?;
    let meta=fs::symlink_metadata(source).map_err(|e|e.to_string())?;
    if meta.file_type().is_symlink()||!meta.is_file()||meta.len()==0||meta.len()>32*1024*1024 {
        return Err("dataset must be a regular non-symlink CSV between 1 byte and 32 MiB".into());
    }
    let mut input=fs::File::open(source).map_err(|e|e.to_string())?;
    let mut digest=Sha256::new();let mut bytes=0u64;let mut buffer=[0u8;64*1024];
    loop {let n=input.read(&mut buffer).map_err(|e|e.to_string())?;if n==0{break;}bytes+=n as u64;if bytes>32*1024*1024{return Err("dataset exceeds 32 MiB bound".into());}digest.update(&buffer[..n]);}
    let sha=format!("{:x}",digest.finalize());let mut connection=connect(root)?;let conn=sql(connection.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate))?;
    let project_exists:bool=sql(conn.query_row("SELECT EXISTS(SELECT 1 FROM projects WHERE id=?1)",[project],|r|r.get(0)))?;
    if !project_exists{return Err("unknown project".into());}
    let folder=root.join("datasets").join(project);ensure_dir(root,&folder)?;let path=folder.join(format!("{id}.csv"));
    let existing:Option<(String,String)>=sql(conn.query_row("SELECT path,sha256 FROM scientific_datasets WHERE project=?1 AND id=?2",params![project,id],|r|Ok((r.get(0)?,r.get(1)?))).optional())?;
    if let Some((old_path,old_hash))=existing {if Path::new(&old_path)==path&&old_hash==sha{return Ok(json!({"project_id":project,"dataset_id":id,"sha256":sha,"bytes":bytes,"reused":true}));}return Err("dataset id already exists with different content".into());}
    let (project_count,device_count,device_bytes):(i64,i64,i64)=sql(conn.query_row(
        "SELECT (SELECT count(*) FROM scientific_datasets WHERE project=?1),(SELECT count(*) FROM scientific_datasets),(SELECT COALESCE(sum(bytes),0) FROM scientific_datasets)",
        [project],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))))?;
    if project_count>=10||device_count>=100||device_bytes.saturating_add(bytes as i64)>512*1024*1024{return Err("owned dataset storage quota reached".into());}
    ensure_disk_space(root,bytes.saturating_add(OUTPUT_CAP))?;
    enforce_state_write_admission(root,Some(&conn),bytes,StorageWriteKind::ManagedBlob,Some(project))?;
    if path.exists(){return Err("owned dataset path already exists".into());}
    let temp=folder.join(format!(".{id}.staging"));
    let mut src=fs::File::open(source).map_err(|e|e.to_string())?;let mut dst=fs::OpenOptions::new().write(true).create_new(true).open(&temp).map_err(|e|e.to_string())?;
    std::io::copy(&mut src,&mut dst).map_err(|e|e.to_string())?;dst.sync_all().map_err(|e|e.to_string())?;
    let copied=fs::read(&temp).map_err(|e|e.to_string())?;if hash(&copied)!=sha{let _=fs::remove_file(&temp);return Err("dataset changed while staging".into());}
    fs::set_permissions(&temp,std::fs::Permissions::from_mode(0o444)).map_err(|e|e.to_string())?;fs::rename(&temp,&path).map_err(|e|e.to_string())?;
    sql(conn.execute("INSERT INTO scientific_datasets(project,id,path,bytes,sha256,registered_ms) VALUES(?1,?2,?3,?4,?5,?6)",params![project,id,path.to_string_lossy(),bytes as i64,sha,now_ms()]))?;
    sql(conn.execute("INSERT INTO events(project,kind,payload) VALUES(?1,'dataset.registered',?2)",params![project,json!({"dataset_id":id,"sha256":sha,"bytes":bytes}).to_string()]))?;
    conn.commit().map_err(|e|e.to_string())?;Ok(json!({"project_id":project,"dataset_id":id,"sha256":sha,"bytes":bytes,"reused":false}))
}

pub fn submit(root:&Path,mut s:JobSpec)->Result<JobView>{
    crate::broker::resolve_job_spec_preset(root, &mut s)?;
    validate_spec(&s)?;
    if let Probe::Scientific{operation,candidate_id:Some(_),..}=&s.probe {
        if operation=="finalize"{validate_finalization_job(root,&s)?;}
    }
    crate::broker::check_request_feasibility(root, &s)?;
    let mut c=connect(root)?;
    // Serialize admission with other submitters so the state byte check and
    // output reservation cannot race each other.
    let tx=sql(c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate))?;
    let queue_project:i64=sql(tx.query_row("SELECT count(*) FROM jobs WHERE project=?1 AND status='queued'",[&s.project_id],|r|r.get(0)))?;let queue_device:i64=sql(tx.query_row("SELECT count(*) FROM jobs WHERE status='queued'",[],|r|r.get(0)))?;
    if queue_project>=128||queue_device>=1024{return Err("bounded execution queue is full".into());}
    let previous:Option<String>=sql(tx.query_row("SELECT spec FROM jobs WHERE id=?1 AND project=?2",params![s.job_id,s.project_id],|r|r.get(0)).optional())?;
    let encoded=serde_json::to_string(&s).map_err(|e|e.to_string())?;
    if let Some(old)=previous {if old!=encoded{return Err("job id reused with a different specification".into());}let row=load_job(&tx,&s.project_id,&s.job_id)?;tx.commit().map_err(|e|e.to_string())?;return Ok(row);}
    resources::check_certification_request(&s)?;
    let count:i64=sql(tx.query_row("SELECT count(*) FROM jobs WHERE project=?1",[&s.project_id],|r|r.get(0)))?;
    if count>=4096{return Err("retained project job history ceiling reached".into());}
    enforce_state_admission(root, &tx, s.max_output_bytes.saturating_add(LOG_CAP).saturating_add(64*1024))?;
    let (project_state,project_policy):(String,String)=sql(tx.query_row("SELECT state,policy FROM projects WHERE id=?1",[&s.project_id],|r|Ok((r.get(0)?,r.get(1)?))))?;
    let is_finalize_op=matches!(&s.probe,Probe::Scientific{operation,..} if operation=="finalize");
    if project_state!="created"&&!(is_finalize_op&&project_state=="finalizing") {return Err(format!("project does not admit work in state {project_state}"));}
    let _p:Policy=serde_json::from_str(&project_policy).map_err(|e|e.to_string())?;
    let task_json:Option<String>=sql(tx.query_row("SELECT data FROM tasks WHERE project=?1 AND id=?2",params![s.project_id,s.task_id],|r|r.get(0)).optional())?;
    let task_json=task_json.ok_or("unknown task")?;
    let task:crate::core::Task=serde_json::from_str(&task_json).map_err(|e|e.to_string())?;
    if task.status!="active"||task.generation!=s.generation||task.intent_revision!=sql(tx.query_row("SELECT intent_revision FROM projects WHERE id=?1",[&s.project_id],|r|r.get::<_,u64>(0)))?||task.expires_ms.is_none_or(|end|end<now_ms().saturating_add(s.timeout_ms as i64)){return Err("stale or too-short task lease".into());}
    sql(tx.execute("INSERT INTO jobs(id,project,task,generation,spec,status,submitted_ms) VALUES(?1,?2,?3,?4,?5,'queued',?6)",params![s.job_id,s.project_id,s.task_id,s.generation,encoded,now_ms()]))?;
    if let Probe::Scientific{operation,dataset_id,spec:scientific_spec,candidate_id,test_access_id}=&s.probe {
        let closed:bool=sql(tx.query_row("SELECT EXISTS(SELECT 1 FROM final_test_gates WHERE project=?1)",[&s.project_id],|r|r.get(0)))?;
        if closed{return Err("scientific research is closed by the durable final-test gate".into());}
        if operation=="pipeline"||operation=="finalize" {
            let ready:bool=sql(tx.query_row("SELECT EXISTS(SELECT 1 FROM environments WHERE id=?1 AND status='ready' AND python LIKE 'docker:%')",[s.environment_id.as_deref().ok_or("scientific container environment missing")?],|r|r.get(0)))?;
            if !ready{return Err(format!("{operation} requires a verified ready container environment; host fallback prohibited"));}
            if operation=="pipeline" {
                let workflow=root.join("workflows").join(&s.project_id);
                let count:i64=sql(tx.query_row("SELECT count(*) FROM pipeline_receipts WHERE project=?1",[&s.project_id],|r|r.get(0)))?;
                if count>=16 || (workflow.exists()&&tree_size(&workflow)?>64*1024*1024){return Err("retained pipeline/storage ceiling reached; explicit retirement is required".into());}
            }
        }
        let dataset:Option<(String,i64,String)>=sql(tx.query_row("SELECT path,bytes,sha256 FROM scientific_datasets WHERE project=?1 AND id=?2",params![s.project_id,dataset_id],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional())?;
        let Some((_,_,dataset_hash))=dataset else{return Err("scientific job requires a registered project dataset".into());};
        if *operation=="audit" && scientific_spec.is_some(){
            let out=serde_json::to_string(scientific_spec.as_ref().unwrap()).map_err(|e|e.to_string())?;
            if out.len()>4096{return Err("scientific project spec exceeds bound".into());}
        }
        if *operation=="finalize" {
            let cand=candidate_id.as_deref().unwrap();
            let receipt_dataset:Option<String>=sql(tx.query_row("SELECT dataset_sha256 FROM pipeline_receipts WHERE project=?1 AND candidate=?2",params![s.project_id,cand],|r|r.get(0)).optional())?;
            if receipt_dataset.as_deref()!=Some(dataset_hash.as_str()) {
                return Err("candidate is not eligible under the registered dataset pipeline receipts".into());
            }
        }
        sql(tx.execute("INSERT INTO scientific_jobs(project,job,dataset,operation,spec) VALUES(?1,?2,?3,?4,?5)",params![s.project_id,s.job_id,dataset_id,operation,serde_json::to_string(&s.probe).map_err(|e|e.to_string())?]))?;
        if *operation=="finalize" {
            let candidate=candidate_id.as_deref().unwrap();let access=test_access_id.as_deref().unwrap();
            let gate:Option<(String,String,String)>=sql(tx.query_row("SELECT access_id,candidate_id,status FROM final_test_gates WHERE project=?1 AND dataset_sha256=?2",params![s.project_id,dataset_hash],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional())?;
            if gate.is_some(){return Err("test set already has a durable access reservation".into());}
            sql(tx.execute("INSERT INTO final_test_gates(project,access_id,dataset_sha256,candidate_id,status) VALUES(?1,?2,?3,?4,'reserved')",params![s.project_id,access,dataset_hash,candidate]))?;
            sql(tx.execute("INSERT INTO events(project,kind,payload) VALUES(?1,'final_test.reserved',?2)",params![s.project_id,json!({"job_id":s.job_id,"dataset_sha256":dataset_hash,"candidate_id":candidate,"test_access_id":access}).to_string()]))?;
        }
    }
    sql(tx.execute("INSERT INTO events(project,kind,payload) VALUES(?1,'job.queued',?2)",params![s.project_id,json!({"job_id":s.job_id}).to_string()]))?;
    tx.commit().map_err(|e|e.to_string())?;load_job(&c,&s.project_id,&s.job_id)
}

fn load_job(c:&Connection,p:&str,id:&str)->Result<JobView>{
    let row:(String,String,Option<String>,Option<u32>,i64,Option<i64>,Option<i64>,String)=sql(c.query_row("SELECT spec,status,reason,pid,submitted_ms,started_ms,ended_ms,artifacts FROM jobs WHERE project=?1 AND id=?2",params![p,id],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?,r.get(6)?,r.get(7)?))))?;
    let telemetry:Option<(i64,Option<f64>,Option<u64>,u64)>=sql(c.query_row("SELECT sampled_ms,cpu_percent,rss_bytes,peak_rss_bytes FROM job_telemetry WHERE project=?1 AND job=?2",params![p,id],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?))).optional())?;
    Ok(JobView{spec:serde_json::from_str(&row.0).map_err(|e|e.to_string())?,status:row.1,reason:row.2,pid:row.3,submitted_ms:row.4,started_ms:row.5,ended_ms:row.6,artifacts:serde_json::from_str(&row.7).map_err(|e|e.to_string())?,telemetry:telemetry.map(|(sampled_ms,cpu_percent,rss_bytes,peak_rss_bytes)|ResourceSample{sampled_ms,cpu_percent,rss_bytes,peak_rss_bytes}),resource_receipt:resources::record(c,p,id)?})
}

pub fn inspect(root:&Path,p:&str,id:&str)->Result<JobView>{let c=connect(root)?;load_job(&c,p,id)}
fn save_action(root:&Path,p:&str,a:&Action)->Result<()>{let c=connect(root)?;sql(c.execute("UPDATE actions SET data=?3 WHERE project=?1 AND id=?2",params![p,a.action_id,serde_json::to_string(a).map_err(|e|e.to_string())?]))?;Ok(())}

pub fn cancel(root:&Path,p:&str,id:&str)->Result<JobView>{
    let mut c=connect(root)?;let tx=sql(c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate))?;
    let status:String=sql(tx.query_row("SELECT status FROM jobs WHERE project=?1 AND id=?2",params![p,id],|r|r.get(0)))?;
    if status=="queued" {sql(tx.execute("UPDATE jobs SET status='cancelled',reason='cancelled before admission',ended_ms=?3 WHERE project=?1 AND id=?2",params![p,id,now_ms()]))?;sql(tx.execute("INSERT INTO events(project,kind,payload) VALUES(?1,'job.terminal',?2)",params![p,json!({"job_id":id,"status":"cancelled"}).to_string()]))?;}
    else if status=="running" {sql(tx.execute("UPDATE jobs SET cancel_requested=1,status='terminating' WHERE project=?1 AND id=?2",params![p,id]))?;sql(tx.execute("INSERT INTO events(project,kind,payload) VALUES(?1,'job.cancel_requested',?2)",params![p,json!({"job_id":id}).to_string()]))?;}
    tx.commit().map_err(|e|e.to_string())?;load_job(&c,p,id)
}
pub fn set_project_weight(root:&Path,p:&str,weight:u8)->Result<()>{
    if !(1..=10).contains(&weight){return Err("project scheduling weight must be 1..=10".into());}let c=connect(root)?;
    sql(c.execute("INSERT INTO project_weights(project,weight) VALUES(?1,?2) ON CONFLICT(project) DO UPDATE SET weight=excluded.weight",params![p,weight]))?;
    sql(c.execute("INSERT INTO events(project,kind,payload) VALUES(?1,'scheduler.weight_changed',?2)",params![p,json!({"weight":weight}).to_string()]))?;Ok(())
}

pub fn create_environment(root:&Path,id:&str,python:&Path)->Result<Value>{
    valid_id(id)?;let out=Command::new(python).arg("--version").output().map_err(|e|format!("cannot run selected Python: {e}"))?;
    if !out.status.success(){return Err("selected Python version probe failed".into());}
        let version=output_text(&out);
    if !version.starts_with("Python 3.13."){return Err("M1 managed environment requires Python 3.13".into());}
    let identity=hash(format!("{version}|{}|{}|{}",python.display(),std::env::consts::OS,std::env::consts::ARCH).as_bytes());
    let c=connect(root)?;let exists:Option<(String,String)>=sql(c.query_row("SELECT identity,status FROM environments WHERE id=?1",[id],|r|Ok((r.get(0)?,r.get(1)?))).optional())?;
    if let Some((old,status))=exists {if old==identity&&status=="ready"{return Ok(json!({"id":id,"identity":identity,"status":"ready","reused":true}));}return Err(format!("environment identity already exists with status {status}"));}
    ensure_disk_space(root,0)?;let base=root.join("environments");ensure_dir(root,&base)?;let stage=base.join(format!(".{id}.staging"));let final_path=base.join(id);
    if stage.exists(){safe_remove(root,&stage)?;}if final_path.exists(){return Err("environment path already exists".into());}
    sql(c.execute("INSERT INTO environments(id,python,identity,path,status) VALUES(?1,?2,?3,?4,'building')",params![id,python.to_string_lossy(),identity,final_path.to_string_lossy()]))?;
    let output=Command::new(python).args(["-m","venv","--without-pip"]).arg(&stage).env_clear().env("PATH","/usr/bin:/bin").output();
    let result=match output {Ok(o) if o.status.success()=>{
        let env_python=stage.join("bin/python3");let verify=Command::new(&env_python).arg("--version").output().map_err(|e|e.to_string())?;
        if !verify.status.success()||output_text(&verify)!=version {Err(format!("created interpreter failed version verification (expected {version:?}, got {:?}, stderr {})",output_text(&verify),String::from_utf8_lossy(&verify.stderr)))}
        else {fs::rename(&stage,&final_path).map_err(|e|e.to_string())?;sql(c.execute("UPDATE environments SET status='ready' WHERE id=?1",[id]))?;Ok(json!({"id":id,"identity":identity,"status":"ready","python":final_path.join("bin/python3")}))}
    },Ok(o)=>Err(format!("venv creation failed: {}",String::from_utf8_lossy(&o.stderr))),Err(e)=>Err(e.to_string())};
    if result.is_err(){if stage.exists(){let _=safe_remove(root,&stage);}let _=c.execute("UPDATE environments SET status='failed' WHERE id=?1",[id]);}
    result
}

/// Install the fixed M2 scientific profile only inside an owned ready environment.
pub fn install_scientific_profile(root:&Path,id:&str)->Result<Value>{
    valid_id(id)?;let c=connect(root)?;
    let (path,status):(String,String)=sql(c.query_row("SELECT path,status FROM environments WHERE id=?1",[id],|r|Ok((r.get(0)?,r.get(1)?))))?;
    if status!="ready" {return Err("scientific profile requires a ready owned environment".into());}
    let expected=root.join("environments").join(id);if Path::new(&path)!=expected{return Err("environment path does not match owned identity".into());}
    let python=expected.join("bin/python3");if !python.is_file(){return Err("managed Python executable is absent".into());}
    ensure_disk_space(root,512*1024*1024)?;
    let ensure=Command::new(&python).args(["-m","ensurepip","--upgrade"]).env_clear().env("PATH","/usr/bin:/bin").stdout(Stdio::null()).stderr(Stdio::null()).status().map_err(|e|e.to_string())?;
    if !ensure.success(){return Err("could not bootstrap pip inside owned environment".into());}
    let source=crate::assets::ensure_python_root(root)?.join("scientific-requirements.txt");
    let install=Command::new(&python).args(["-m","pip","install","--no-input","--disable-pip-version-check","--only-binary=:all:","--no-cache-dir","-r"]).arg(&source)
        .env("PIP_NO_INPUT","1").stdout(Stdio::null()).stderr(Stdio::null()).spawn().map_err(|e|e.to_string())?;
    let mut child=install;let deadline=Instant::now()+Duration::from_secs(1200);let status=loop{if Instant::now()>=deadline{let _=child.kill();let _=child.wait();return Err("scientific profile installation exceeded 20-minute limit".into());}if let Some(s)=child.try_wait().map_err(|e|e.to_string())?{break s;}thread::sleep(Duration::from_millis(100));};
    if !status.success(){return Err(format!("scientific profile installation failed with {status}"));}
    let verify=Command::new(&python).args(["-c","import numpy,pandas,scipy,sklearn,joblib; assert (numpy.__version__,pandas.__version__,scipy.__version__,sklearn.__version__,joblib.__version__)==('2.2.6','2.3.0','1.16.1','1.8.0','1.5.1')"])
        .env_clear().env("PATH","/usr/bin:/bin").stdout(Stdio::null()).stderr(Stdio::null()).status().map_err(|e|e.to_string())?;
    if !verify.success(){return Err("scientific profile version verification failed".into());}
    Ok(json!({"id":id,"status":"ready","profile":"m2-cpu-v1","python":python,"installed_bytes":environment_size(&expected)?}))
}

/// Install the fixed M3 search profile only inside an owned ready environment.
pub fn install_search_profile(root:&Path,id:&str)->Result<Value>{
    valid_id(id)?;let c=connect(root)?;
    let (path,status):(String,String)=sql(c.query_row("SELECT path,status FROM environments WHERE id=?1",[id],|r|Ok((r.get(0)?,r.get(1)?))))?;
    if status!="ready" {return Err("search profile requires a ready owned environment".into());}
    let leased:bool=sql(c.query_row("SELECT EXISTS(SELECT 1 FROM environment_leases WHERE environment=?1)",[id],|r|r.get(0)))?;
    if leased {return Err("cannot mutate a leased environment".into());}
    let expected=root.join("environments").join(id);if Path::new(&path)!=expected{return Err("environment path does not match owned identity".into());}
    let python=expected.join("bin/python3");if !python.is_file(){return Err("managed Python executable is absent".into());}
    ensure_disk_space(root,512*1024*1024)?;
    let ensure=Command::new(&python).args(["-m","ensurepip","--upgrade"]).env_clear().env("PATH","/usr/bin:/bin").stdout(Stdio::null()).stderr(Stdio::null()).status().map_err(|e|e.to_string())?;
    if !ensure.success(){return Err("could not bootstrap pip inside owned environment".into());}
    let source=crate::assets::ensure_python_root(root)?.join("search-requirements.txt");
    let install=Command::new(&python).args(["-m","pip","install","--no-input","--disable-pip-version-check","--only-binary=:all:","--no-cache-dir","-r"]).arg(&source)
        .env("PIP_NO_INPUT","1").stdout(Stdio::null()).stderr(Stdio::null()).spawn().map_err(|e|e.to_string())?;
    let mut child=install;let deadline=Instant::now()+Duration::from_secs(1200);let status=loop{if Instant::now()>=deadline{let _=child.kill();let _=child.wait();return Err("search profile installation exceeded 20-minute limit".into());}if let Some(s)=child.try_wait().map_err(|e|e.to_string())?{break s;}thread::sleep(Duration::from_millis(100));};
    if !status.success(){return Err(format!("search profile installation failed with {status}"));}
    let verify=Command::new(&python).args(["-c","import optuna; assert optuna.__version__=='5.0.0'"])
        .env_clear().env("PATH","/usr/bin:/bin").stdout(Stdio::null()).stderr(Stdio::null()).status().map_err(|e|e.to_string())?;
    if !verify.success(){return Err("search profile version verification failed".into());}
    Ok(json!({"id":id,"status":"ready","profile":"m3-search-v1","python":python,"installed_bytes":environment_size(&expected)?}))
}

fn environment_size(path:&Path)->Result<u64>{
    let mut total=0u64;
    for entry in fs::read_dir(path).map_err(|e|e.to_string())? {
        let entry=entry.map_err(|e|e.to_string())?;let meta=fs::symlink_metadata(entry.path()).map_err(|e|e.to_string())?;
        if meta.file_type().is_symlink(){continue;}
        if meta.is_file(){total=total.saturating_add(meta.len());}
        else if meta.is_dir(){total=total.saturating_add(environment_size(&entry.path())?);}
    }
    Ok(total)
}
fn output_text(out:&std::process::Output)->String{let stdout=String::from_utf8_lossy(&out.stdout).trim().to_string();if stdout.is_empty(){String::from_utf8_lossy(&out.stderr).trim().to_string()}else{stdout}}

pub fn remove_environment(root:&Path,id:&str)->Result<()> {
    valid_id(id)?;let c=connect(root)?;let tx=sql(rusqlite::Transaction::new_unchecked(&c,rusqlite::TransactionBehavior::Immediate))?;
    let (path,status):(String,String)=sql(tx.query_row("SELECT path,status FROM environments WHERE id=?1",[id],|r|Ok((r.get(0)?,r.get(1)?))))?;
    let leases:i64=sql(tx.query_row("SELECT count(*) FROM environment_leases WHERE environment=?1",[id],|r|r.get(0)))?;if leases>0{return Err("environment is leased by active or queued work".into());}
    if status=="building"{return Err("cannot remove an environment while it is building".into());}
    sql(tx.execute("UPDATE environments SET status='deleting' WHERE id=?1",[id]))?;tx.commit().map_err(|e|e.to_string())?;
    let expected=root.join("environments").join(id);if Path::new(&path)!=expected{return Err("managed environment path does not match its identity".into());}
    if expected.exists(){safe_remove(root,&expected)?;}
    sql(c.execute("DELETE FROM environments WHERE id=?1",[id]))?;Ok(())
}

pub fn pin_artifact(root:&Path,p:&str,job:&str,pinned:bool)->Result<()> {
    valid_id(p)?; valid_id(job)?;
    let c=connect(root)?;
    let tx=sql(rusqlite::Transaction::new_unchecked(&c,rusqlite::TransactionBehavior::Immediate))?;
    enforce_state_write_admission(root,Some(&tx),16*1024,StorageWriteKind::Checkpoint,Some(p))?;
    let changed=sql(tx.execute("UPDATE managed_artifacts SET pinned=?3 WHERE project=?1 AND job=?2",params![p,job,pinned as i64]))?;
    if changed==0{return Err("job has no managed artifact to pin".into());}
    sql(tx.execute(
        "UPDATE artifacts SET pinned=?3, updated_ms=?4 WHERE project=?1 AND artifact_id IN (SELECT artifact_id FROM artifact_versions WHERE project=?1 AND source_job_id=?2)",
        params![p,job,pinned as i64,now_ms()],
    ))?;
    tx.commit().map_err(|e|e.to_string())?;
    Ok(())
}

pub const MAX_RETAINED_EVENTS_PER_PROJECT: i64 = 64;
pub const MAX_RETAINED_COMMANDS_PER_PROJECT: i64 = 64;
pub const MAX_ACTIVE_COMMAND_RAW_BODIES: usize = 24;
pub const MAX_RETAINED_CHECKPOINTS_PER_TASK: i64 = 4;
pub const MAX_RETAINED_TERMINAL_JOBS_PER_PROJECT: i64 = 24;
pub const MAX_RETAINED_SETTLED_ACTIONS_PER_PROJECT: usize = 24;
pub const MAX_RETAINED_ACTION_FENCES_PER_PROJECT: i64 = 64;
pub const MAX_RETAINED_TOMBSTONES_PER_PROJECT: i64 = 128;
pub const MAX_STATE_TOTAL_BYTES: u64 = 128 * 1024 * 1024;

pub fn record_tombstone_conn(
    tx: &rusqlite::Transaction<'_>,
    project: &str,
    entity_kind: &str,
    entity_id: &str,
    parent_id: &str,
    identity_hash: &str,
    reclaimed_bytes: u64,
    details: Value,
) -> Result<()> {
    sql(tx.execute(
        "INSERT INTO tombstones(project, entity_kind, entity_id, parent_id, identity_hash, reclaimed_bytes, details_json, tombstoned_ms)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
         ON CONFLICT(project, entity_kind, entity_id) DO UPDATE SET
            parent_id=excluded.parent_id,
            identity_hash=excluded.identity_hash,
            reclaimed_bytes=excluded.reclaimed_bytes,
            details_json=excluded.details_json,
            tombstoned_ms=excluded.tombstoned_ms",
        params![
            project,
            entity_kind,
            entity_id,
            parent_id,
            identity_hash,
            reclaimed_bytes as i64,
            details.to_string(),
            now_ms(),
        ],
    ))?;
    Ok(())
}

pub fn record_tombstone(
    root: &Path,
    project: &str,
    entity_kind: &str,
    entity_id: &str,
    parent_id: &str,
    identity_hash: &str,
    reclaimed_bytes: u64,
    details: Value,
) -> Result<()> {
    let c = connect(root)?;
    let tx = sql(rusqlite::Transaction::new_unchecked(&c,rusqlite::TransactionBehavior::Immediate))?;
    record_tombstone_conn(&tx, project, entity_kind, entity_id, parent_id, identity_hash, reclaimed_bytes, details)?;
    tx.commit().map_err(|e| e.to_string())?;
    Ok(())
}

pub fn archive_artifacts(root: &Path, project: Option<&str>) -> Result<Value> {
    let reclaimed_bytes = collect_artifacts(root)?;
    let c = connect(root)?;
    let (tombstones_count, reclaimed_sum): (i64, i64) = if let Some(p) = project {
        sql(c.query_row(
            "SELECT count(*), COALESCE(sum(reclaimed_bytes), 0) FROM tombstones WHERE project=?1",
            [p],
            |r| Ok((r.get(0)?, r.get(1)?)),
        ))?
    } else {
        sql(c.query_row(
            "SELECT count(*), COALESCE(sum(reclaimed_bytes), 0) FROM tombstones",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        ))?
    };
    Ok(json!({
        "status": "archived",
        "project": project,
        "reclaimed_bytes": reclaimed_bytes,
        "tombstones_count": tombstones_count,
        "tombstone_reclaimed_bytes": reclaimed_sum,
    }))
}
pub const EMERGENCY_RECOVERY_HEADROOM_BYTES: u64 = 4 * 1024 * 1024;
pub const SQLITE_WAL_RESERVE_BYTES: u64 = 2 * 1024 * 1024;
pub const DEFAULT_MAX_ARTIFACT_BUDGET_BYTES: u64 = 32 * 1024 * 1024;
pub const DEFAULT_MAX_ENVIRONMENT_BUDGET_BYTES: u64 = 32 * 1024 * 1024;
pub const DEFAULT_MAX_EXPORT_BUDGET_BYTES: u64 = 32 * 1024 * 1024;
pub const DEFAULT_MAX_WORKFLOW_BUDGET_BYTES: u64 = 64 * 1024 * 1024;
pub const DEFAULT_MAX_CHECKPOINT_BUDGET_BYTES: u64 = 8 * 1024 * 1024;
pub const DEFAULT_MAX_PROVIDER_RECEIPT_BYTES: u64 = 8 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct StorageBudgetConfig {
    pub max_state_total_bytes: u64,
    pub disk_headroom_bytes: u64,
    pub emergency_recovery_headroom_bytes: u64,
    pub sqlite_wal_reserve_bytes: u64,
    pub max_artifact_budget_bytes: u64,
    pub max_environment_budget_bytes: u64,
    pub max_export_budget_bytes: u64,
    pub max_workflow_budget_bytes: u64,
    pub max_checkpoint_budget_bytes: u64,
    pub max_provider_receipt_bytes: u64,
}

impl Default for StorageBudgetConfig {
    fn default() -> Self {
        Self {
            max_state_total_bytes: MAX_STATE_TOTAL_BYTES,
            disk_headroom_bytes: DISK_HEADROOM,
            emergency_recovery_headroom_bytes: EMERGENCY_RECOVERY_HEADROOM_BYTES,
            sqlite_wal_reserve_bytes: SQLITE_WAL_RESERVE_BYTES,
            max_artifact_budget_bytes: DEFAULT_MAX_ARTIFACT_BUDGET_BYTES,
            max_environment_budget_bytes: DEFAULT_MAX_ENVIRONMENT_BUDGET_BYTES,
            max_export_budget_bytes: DEFAULT_MAX_EXPORT_BUDGET_BYTES,
            max_workflow_budget_bytes: DEFAULT_MAX_WORKFLOW_BUDGET_BYTES,
            max_checkpoint_budget_bytes: DEFAULT_MAX_CHECKPOINT_BUDGET_BYTES,
            max_provider_receipt_bytes: DEFAULT_MAX_PROVIDER_RECEIPT_BYTES,
        }
    }
}

impl StorageBudgetConfig {
    pub fn load(root: &Path) -> Self {
        let mut cfg = Self::default();
        let config_file = root.join("storage_budgets.json");
        if config_file.is_file() {
            if let Ok(bytes) = fs::read(&config_file) {
                if let Ok(custom) = serde_json::from_slice::<StorageBudgetConfig>(&bytes) {
                    cfg = custom;
                }
            }
        }
        if let Ok(val) = std::env::var("MLRESCURIX_MAX_STATE_TOTAL_BYTES") {
            if let Ok(n) = val.parse::<u64>() { cfg.max_state_total_bytes = n; }
        }
        if let Ok(val) = std::env::var("MLRESCURIX_DISK_HEADROOM_BYTES") {
            if let Ok(n) = val.parse::<u64>() { cfg.disk_headroom_bytes = n; }
        }
        if let Ok(val) = std::env::var("MLRESCURIX_ARTIFACT_BUDGET_BYTES") {
            if let Ok(n) = val.parse::<u64>() { cfg.max_artifact_budget_bytes = n; }
        }
        if let Ok(val) = std::env::var("MLRESCURIX_ENVIRONMENT_BUDGET_BYTES") {
            if let Ok(n) = val.parse::<u64>() { cfg.max_environment_budget_bytes = n; }
        }
        if let Ok(val) = std::env::var("MLRESCURIX_EXPORT_BUDGET_BYTES") {
            if let Ok(n) = val.parse::<u64>() { cfg.max_export_budget_bytes = n; }
        }
        if let Ok(val) = std::env::var("MLRESCURIX_WORKFLOW_BUDGET_BYTES") {
            if let Ok(n) = val.parse::<u64>() { cfg.max_workflow_budget_bytes = n; }
        }
        if let Ok(val) = std::env::var("MLRESCURIX_CHECKPOINT_BUDGET_BYTES") {
            if let Ok(n) = val.parse::<u64>() { cfg.max_checkpoint_budget_bytes = n; }
        }
        if let Ok(val) = std::env::var("MLRESCURIX_PROVIDER_RECEIPT_BYTES") {
            if let Ok(n) = val.parse::<u64>() { cfg.max_provider_receipt_bytes = n; }
        }
        cfg.max_state_total_bytes=cfg.max_state_total_bytes.min(MAX_STATE_TOTAL_BYTES);
        cfg
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageWriteKind {
    JobAdmission,
    Checkpoint,
    ProviderReceipt,
    Log,
    ManagedBlob,
    RuntimeAsset,
    BundleStaging,
    EmergencyRecovery,
}

pub fn root_from_conn(c: &rusqlite::Connection) -> Result<PathBuf> {
    let file: String = c.query_row("PRAGMA database_list", [], |r| r.get(2))
        .map_err(|e| format!("cannot determine database file: {e}"))?;
    let path = Path::new(&file);
    let root = path.parent().ok_or("database file has no parent directory")?;
    Ok(root.to_path_buf())
}

pub fn get_active_job_reservations(root: &Path, conn: Option<&Connection>) -> Result<u64> {
    if let Some(c) = conn {
        get_active_job_reservations_from_conn(c)
    } else {
        let db_path = root.join("state.sqlite");
        if !db_path.exists() {
            return Ok(0);
        }
        let c = rusqlite::Connection::open(&db_path).map_err(|e| e.to_string())?;
        get_active_job_reservations_from_conn(&c)
    }
}

fn get_active_job_reservations_from_conn(c: &Connection) -> Result<u64> {
    let has_jobs: bool = c.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='jobs')", [], |r| r.get(0)).unwrap_or(false);
    if !has_jobs {
        return Ok(0);
    }
    let mut stmt = sql(c.prepare("SELECT spec FROM jobs WHERE status IN('queued','running','terminating','recovering')"))?;
    let specs = sql(stmt.query_map([], |row| row.get::<_, String>(0)))?
        .collect::<rusqlite::Result<Vec<_>>>().map_err(|e| e.to_string())?;
    let mut reserved = 0u64;
    for encoded in specs {
        let spec: JobSpec = serde_json::from_str(&encoded).map_err(|e| e.to_string())?;
        reserved = reserved.checked_add(spec.max_output_bytes).and_then(|n| n.checked_add(LOG_CAP + 64*1024))
            .ok_or("state reservation size overflow")?;
    }
    Ok(reserved)
}

fn get_artifact_usage(c: &Connection, project: Option<&str>) -> Result<u64> {
    let has_ma: bool = c.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='managed_artifacts')", [], |r| r.get(0)).unwrap_or(false);
    let has_sd: bool = c.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='scientific_datasets')", [], |r| r.get(0)).unwrap_or(false);
    if !has_ma && !has_sd {
        return Ok(0);
    }
    let ma_sum: i64 = if has_ma {
        if let Some(p) = project {
            sql(c.query_row("SELECT COALESCE(sum(bytes), 0) FROM managed_artifacts WHERE project=?1", [p], |r| r.get(0)))?
        } else {
            sql(c.query_row("SELECT COALESCE(sum(bytes), 0) FROM managed_artifacts", [], |r| r.get(0)))?
        }
    } else { 0 };
    let sd_sum: i64 = if has_sd {
        if let Some(p) = project {
            sql(c.query_row("SELECT COALESCE(sum(bytes), 0) FROM scientific_datasets WHERE project=?1", [p], |r| r.get(0)))?
        } else {
            sql(c.query_row("SELECT COALESCE(sum(bytes), 0) FROM scientific_datasets", [], |r| r.get(0)))?
        }
    } else { 0 };
    Ok((ma_sum as u64).saturating_add(sd_sum as u64))
}

fn get_checkpoint_usage(c: &Connection, project: Option<&str>) -> Result<u64> {
    let has_cp: bool = c.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='checkpoints')", [], |r| r.get(0)).unwrap_or(false);
    let has_acp: bool = c.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='agent_checkpoints')", [], |r| r.get(0)).unwrap_or(false);
    let cp_sum: i64 = if has_cp {
        if let Some(p) = project {
            sql(c.query_row("SELECT COALESCE(sum(length(data)), 0) FROM checkpoints WHERE project=?1", [p], |r| r.get(0)))?
        } else {
            sql(c.query_row("SELECT COALESCE(sum(length(data)), 0) FROM checkpoints", [], |r| r.get(0)))?
        }
    } else { 0 };
    let acp_sum: i64 = if has_acp {
        if let Some(p) = project {
            sql(c.query_row("SELECT COALESCE(sum(length(state_snapshot)), 0) FROM agent_checkpoints WHERE project=?1", [p], |r| r.get(0)))?
        } else {
            sql(c.query_row("SELECT COALESCE(sum(length(state_snapshot)), 0) FROM agent_checkpoints", [], |r| r.get(0)))?
        }
    } else { 0 };
    Ok((cp_sum as u64).saturating_add(acp_sum as u64))
}

fn get_provider_receipt_usage(c: &Connection, project: Option<&str>) -> Result<u64> {
    let has_ptr: bool = c.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='provider_turn_receipts')", [], |r| r.get(0)).unwrap_or(false);
    if !has_ptr {
        return Ok(0);
    }
    let sum: i64 = if let Some(p) = project {
        sql(c.query_row("SELECT COALESCE(sum(length(response_json)), 0) FROM provider_turn_receipts WHERE response_json IS NOT NULL AND project=?1", [p], |r| r.get(0)))?
    } else {
        sql(c.query_row("SELECT COALESCE(sum(length(response_json)), 0) FROM provider_turn_receipts WHERE response_json IS NOT NULL", [], |r| r.get(0)))?
    };
    Ok(sum as u64)
}

fn get_environment_usage(root: &Path) -> Result<u64> {
    let mut total = 0u64;
    let env_dir = root.join("environments");
    if env_dir.is_dir() {
        total = total.saturating_add(state_size(&env_dir)?);
    }
    let assets_dir = root.join("runtime-assets");
    if assets_dir.is_dir() {
        total = total.saturating_add(state_size(&assets_dir)?);
    }
    Ok(total)
}

fn get_export_usage(root: &Path, project: Option<&str>) -> Result<u64> {
    let mut total = 0u64;
    if let Some(p) = project {
        let b = root.join("workflows").join(p).join("bundle");
        if b.is_dir() {
            total = total.saturating_add(state_size(&b)?);
        }
    } else {
        let wf = root.join("workflows");
        if wf.is_dir() {
            for entry in fs::read_dir(&wf).map_err(|e| e.to_string())? {
                let entry = entry.map_err(|e| e.to_string())?;
                let b = entry.path().join("bundle");
                if b.is_dir() {
                    total = total.saturating_add(state_size(&b)?);
                }
            }
        }
    }
    Ok(total)
}

pub fn enforce_state_write_admission(
    root: &Path,
    conn: Option<&rusqlite::Connection>,
    requested_bytes: u64,
    kind: StorageWriteKind,
    project: Option<&str>,
) -> Result<()> {
    let config = StorageBudgetConfig::load(root);
    ensure_disk_space_with_config(root, requested_bytes, &config)?;

    let current = total_state_bytes(root)?;

    if kind == StorageWriteKind::EmergencyRecovery {
        if current.checked_add(requested_bytes).is_none_or(|n| n > config.max_state_total_bytes) {
            return Err(format!("emergency recovery hard ceiling reached: {current} bytes used, {requested_bytes} bytes requested, {} byte limit", config.max_state_total_bytes));
        }
        return Ok(());
    }

    let active_reservations = get_active_job_reservations(root, conn)?;
    let required_headroom = config.emergency_recovery_headroom_bytes.saturating_add(config.sqlite_wal_reserve_bytes);

    let total_committed = current
        .checked_add(active_reservations)
        .and_then(|n| n.checked_add(requested_bytes))
        .and_then(|n| n.checked_add(required_headroom));

    if total_committed.is_none_or(|n| n > config.max_state_total_bytes) {
        return Err(format!(
            "total state admission ceiling reached: {current} bytes used, {active_reservations} bytes reserved, {requested_bytes} bytes requested, {required_headroom} bytes emergency/wal headroom, {} byte limit",
            config.max_state_total_bytes
        ));
    }

    let fallback_conn = if conn.is_none() && root.join("state.sqlite").exists() {
        connect(root).ok()
    } else {
        None
    };
    let active_conn = conn.or(fallback_conn.as_ref());

    match kind {
        StorageWriteKind::ManagedBlob => {
            let used = match active_conn {
                Some(c) => get_artifact_usage(c, project)?,
                None => 0,
            };
            if used.saturating_add(requested_bytes) > config.max_artifact_budget_bytes {
                return Err(format!(
                    "artifact storage budget reached: {used} bytes used, {requested_bytes} bytes requested, {} byte limit",
                    config.max_artifact_budget_bytes
                ));
            }
        }
        StorageWriteKind::RuntimeAsset => {
            let used = get_environment_usage(root)?;
            if used.saturating_add(requested_bytes) > config.max_environment_budget_bytes {
                return Err(format!(
                    "environment storage budget reached: {used} bytes used, {requested_bytes} bytes requested, {} byte limit",
                    config.max_environment_budget_bytes
                ));
            }
        }
        StorageWriteKind::BundleStaging => {
            let used = get_export_usage(root, project)?;
            if used.saturating_add(requested_bytes) > config.max_export_budget_bytes {
                return Err(format!(
                    "export storage budget reached: {used} bytes used, {requested_bytes} bytes requested, {} byte limit",
                    config.max_export_budget_bytes
                ));
            }
        }
        StorageWriteKind::Checkpoint => {
            let used = match active_conn {
                Some(c) => get_checkpoint_usage(c, project)?,
                None => 0,
            };
            if used.saturating_add(requested_bytes) > config.max_checkpoint_budget_bytes {
                return Err(format!(
                    "checkpoint storage budget reached: {used} bytes used, {requested_bytes} bytes requested, {} byte limit",
                    config.max_checkpoint_budget_bytes
                ));
            }
        }
        StorageWriteKind::ProviderReceipt => {
            let used = match active_conn {
                Some(c) => get_provider_receipt_usage(c, project)?,
                None => 0,
            };
            if used.saturating_add(requested_bytes) > config.max_provider_receipt_bytes {
                return Err(format!(
                    "provider receipt storage budget reached: {used} bytes used, {requested_bytes} bytes requested, {} byte limit",
                    config.max_provider_receipt_bytes
                ));
            }
        }
        _ => {}
    }

    Ok(())
}

pub fn enforce_state_write_admission_conn(
    conn: &rusqlite::Connection,
    requested_bytes: u64,
    kind: StorageWriteKind,
    project: Option<&str>,
) -> Result<()> {
    let root = root_from_conn(conn)?;
    enforce_state_write_admission(&root, Some(conn), requested_bytes, kind, project)
}

/// Hold the SQLite writer fence through a managed filesystem effect and its
/// metadata update, so concurrent branches cannot both consume the same room.
pub fn with_state_write<T>(root:&Path,bytes:u64,kind:StorageWriteKind,project:Option<&str>,effect:impl FnOnce(&Connection)->Result<T>)->Result<T>{
    let mut c=connect(root)?;
    let tx=sql(c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate))?;
    enforce_state_write_admission(root,Some(&tx),bytes.saturating_add(16*1024),kind,project)?;
    let result=effect(&tx)?;
    tx.commit().map_err(|e|e.to_string())?;
    Ok(result)
}

pub fn enforce_state_admission(root: &Path, tx: &rusqlite::Transaction<'_>, requested: u64) -> Result<()> {
    enforce_state_write_admission(root, Some(tx), requested, StorageWriteKind::JobAdmission, None)
}

struct GcLock(fs::File);
impl GcLock {
    fn acquire(root: &Path) -> Result<Self> {
        let file = fs::OpenOptions::new().read(true).write(true).create(true)
            .custom_flags(libc::O_NOFOLLOW).open(root.join("artifact-gc.lock"))
            .map_err(|e| e.to_string())?;
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } != 0 {
            return Err("could not acquire artifact GC lock".into());
        }
        Ok(Self(file))
    }
}
impl Drop for GcLock {
    fn drop(&mut self) { unsafe { libc::flock(self.0.as_raw_fd(), libc::LOCK_UN); } }
}

pub fn total_state_bytes(root: &Path) -> Result<u64> {
    if !root.exists() {
        return Ok(0);
    }
    state_size(root)
}

/// Count entries inside the owned state without following symlinks. Runtime
/// installations may legitimately contain links; their directory entries are
/// charged by link length while the target is never traversed.
fn state_size(path: &Path) -> Result<u64> {
    let mut total = 0u64;
    for entry in fs::read_dir(path).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let metadata = match fs::symlink_metadata(entry.path()) {Ok(m)=>m,Err(e) if e.kind()==std::io::ErrorKind::NotFound=>continue,Err(e)=>return Err(e.to_string())};
        if metadata.file_type().is_symlink() || metadata.is_file() {
            total = total.saturating_add(metadata.len());
        } else if metadata.is_dir() {
            total = total.saturating_add(state_size(&entry.path())?);
        }
    }
    Ok(total)
}

pub fn storage_policy_summary(root: &Path) -> Result<Value> {
    let c = connect(root)?;
    let config = StorageBudgetConfig::load(root);
    let events_count: i64 = sql(c.query_row("SELECT count(*) FROM events", [], |r| r.get(0)))?;
    let commands_count: i64 = sql(c.query_row("SELECT count(*) FROM commands", [], |r| r.get(0)))?;
    let jobs_count: i64 = sql(c.query_row("SELECT count(*) FROM jobs", [], |r| r.get(0)))?;
    let actions_count: i64 = sql(c.query_row("SELECT count(*) FROM actions", [], |r| r.get(0)))?;
    let charges_count: i64 = sql(c.query_row("SELECT count(*) FROM charges", [], |r| r.get(0)))?;
    let held_reservations: i64 = sql(c.query_row("SELECT count(*) FROM reservations WHERE held=1", [], |r| r.get(0)))?;
    let tombstones_count: i64 = c.query_row("SELECT count(*) FROM tombstones", [], |r| r.get(0)).unwrap_or(0);
    let tombstones_bytes: i64 = c.query_row("SELECT COALESCE(sum(reclaimed_bytes), 0) FROM tombstones", [], |r| r.get(0)).unwrap_or(0);
    let action_fences_count: i64 = c.query_row("SELECT count(*) FROM action_fences", [], |r| r.get(0)).unwrap_or(0);
    let db_bytes = fs::metadata(root.join("state.sqlite")).map(|m| m.len()).unwrap_or(0)
        + fs::metadata(root.join("state.sqlite-wal")).map(|m| m.len()).unwrap_or(0)
        + fs::metadata(root.join("state.sqlite-shm")).map(|m| m.len()).unwrap_or(0);
    let total_bytes = total_state_bytes(root)?;
    let active_reservations = get_active_job_reservations(root, Some(&c))?;
    let artifact_bytes = get_artifact_usage(&c, None)?;
    let checkpoint_bytes = get_checkpoint_usage(&c, None)?;
    let provider_receipt_bytes = get_provider_receipt_usage(&c, None)?;
    let environment_bytes = get_environment_usage(root)?;
    let export_bytes = get_export_usage(root, None)?;
    Ok(json!({
        "policy": {
            "max_retained_events_per_project": MAX_RETAINED_EVENTS_PER_PROJECT,
            "max_retained_commands_per_project": Value::Null,
            "command_receipts": "durable retry fences; older command bodies compacted to SHA-256 hashes beyond active threshold",
            "max_retained_checkpoints_per_task": MAX_RETAINED_CHECKPOINTS_PER_TASK,
            "max_retained_terminal_jobs_per_project": MAX_RETAINED_TERMINAL_JOBS_PER_PROJECT,
            "max_retained_settled_charge_rows_per_project": MAX_RETAINED_SETTLED_ACTIONS_PER_PROJECT,
            "max_retained_action_fences_per_project": MAX_RETAINED_ACTION_FENCES_PER_PROJECT,
            "max_retained_tombstones_per_project": MAX_RETAINED_TOMBSTONES_PER_PROJECT,
            "settled_action_identities": "durable retry fences compacted into action_fences with tombstone accounting",
            "max_state_total_bytes": config.max_state_total_bytes,
            "total_state_ceiling_enforced_for_job_admission": true,
            "state_reservation": "current state bytes plus queued/running/recovering output and log reservations",
            "disk_headroom_bytes": config.disk_headroom_bytes,
            "emergency_recovery_headroom_bytes": config.emergency_recovery_headroom_bytes,
            "sqlite_wal_reserve_bytes": config.sqlite_wal_reserve_bytes,
            "max_artifact_budget_bytes": config.max_artifact_budget_bytes,
            "max_environment_budget_bytes": config.max_environment_budget_bytes,
            "max_export_budget_bytes": config.max_export_budget_bytes,
            "max_workflow_budget_bytes": config.max_workflow_budget_bytes,
            "max_checkpoint_budget_bytes": config.max_checkpoint_budget_bytes,
            "max_provider_receipt_bytes": config.max_provider_receipt_bytes,
        },
        "observed": {
            "total_state_bytes": total_bytes,
            "database_bytes": db_bytes,
            "events_rows": events_count,
            "commands_rows": commands_count,
            "jobs_rows": jobs_count,
            "actions_rows": actions_count,
            "charges_rows": charges_count,
            "tombstones_rows": tombstones_count,
            "tombstone_reclaimed_bytes": tombstones_bytes,
            "action_fences_rows": action_fences_count,
            "held_reservations": held_reservations,
            "active_reservations_bytes": active_reservations,
            "artifact_bytes": artifact_bytes,
            "checkpoint_bytes": checkpoint_bytes,
            "provider_receipt_bytes": provider_receipt_bytes,
            "environment_bytes": environment_bytes,
            "export_bytes": export_bytes,
        }
    }))
}

pub fn collect_artifacts(root:&Path)->Result<u64>{
    let db_before = fs::metadata(root.join("state.sqlite")).map(|m| m.len()).unwrap_or(0)
        + fs::metadata(root.join("state.sqlite-wal")).map(|m| m.len()).unwrap_or(0);
    let _gc_lock = GcLock::acquire(root)?;
    let mut c=connect(root)?;
    // Serialize collectors across processes, but leave SQLite writers free
    // while filesystem paths are reclaimed so scheduler cancellation and
    // settlement can proceed concurrently.
    let mut q=sql(c.prepare(
        "SELECT j.project, j.id FROM jobs j
         WHERE j.status IN('succeeded','failed','timed_out','cancelled','rejected','interrupted')
           AND NOT EXISTS(SELECT 1 FROM managed_artifacts a WHERE a.project=j.project AND a.job=j.id AND a.pinned=1)
           AND NOT EXISTS(SELECT 1 FROM environment_leases l WHERE l.project=j.project AND l.job=j.id)"
    ))?;
    let rows=sql(q.query_map([],|r|Ok((r.get::<_,String>(0)?,r.get::<_,String>(1)?))))?.collect::<rusqlite::Result<Vec<_>>>().map_err(|e|e.to_string())?;
    drop(q);
    let mut reclaimed=0u64;
    for(p,j)in &rows {
        let dir=root.join("artifacts").join(&p).join(&j);
        if dir.exists(){let n=tree_size(&dir)?;safe_remove(root,&dir)?;reclaimed=reclaimed.saturating_add(n);}
        let log=root.join("logs").join(&p).join(format!("{j}.log"));
        if log.exists(){let n=fs::metadata(&log).map_err(|e|e.to_string())?.len();safe_remove(root,&log)?;reclaimed=reclaimed.saturating_add(n);}
        let stage=root.join("staging").join(format!("{p}_{j}"));
        if stage.exists(){let n=tree_size(&stage)?;safe_remove(root,&stage)?;reclaimed=reclaimed.saturating_add(n);}
    }

    // 1. Sweep staging directories: artifacts/<p>/.<j>.staging
    let art_dir = root.join("artifacts");
    if art_dir.is_dir() {
        if let Ok(entries) = fs::read_dir(&art_dir) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    let project_name = entry.file_name().to_string_lossy().to_string();
                    if let Ok(sub_entries) = fs::read_dir(entry.path()) {
                        for sub in sub_entries.flatten() {
                            let name = sub.file_name().to_string_lossy().to_string();
                            if name.starts_with('.') && name.ends_with(".staging") {
                                let job_id = name.trim_start_matches('.').trim_end_matches(".staging");
                                let is_active: bool = c.query_row(
                                    "SELECT EXISTS(SELECT 1 FROM jobs WHERE project=?1 AND id=?2 AND status IN('queued','running','terminating','recovering'))",
                                    params![project_name, job_id],
                                    |r| r.get(0),
                                ).unwrap_or(false);
                                if !is_active {
                                    if let Ok(n) = tree_size(&sub.path()) {
                                        if safe_remove(root, &sub.path()).is_ok() {
                                            reclaimed = reclaimed.saturating_add(n);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 2. Sweep staging/ directory for leaked bundle_verification_* or orphan stage dirs
    let staging_dir = root.join("staging");
    if staging_dir.is_dir() {
        if let Ok(entries) = fs::read_dir(&staging_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("bundle_verification_") {
                    if let Ok(n) = tree_size(&entry.path()) {
                        if safe_remove(root, &entry.path()).is_ok() {
                            reclaimed = reclaimed.saturating_add(n);
                        }
                    }
                }
            }
        }
    }

    // 3. Sweep workflows/<p>/.bundle_stage or other leaked temporary staging
    let wf_dir = root.join("workflows");
    if wf_dir.is_dir() {
        if let Ok(entries) = fs::read_dir(&wf_dir) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    let project_name = entry.file_name().to_string_lossy().to_string();
                    let is_active: bool = c.query_row(
                        "SELECT EXISTS(SELECT 1 FROM jobs WHERE project=?1 AND status IN('queued','running','terminating','recovering'))",
                        params![project_name],
                        |r| r.get(0),
                    ).unwrap_or(false);
                    if !is_active {
                        let bundle_stage = entry.path().join(".bundle_stage");
                        if bundle_stage.exists() {
                            if let Ok(n) = tree_size(&bundle_stage) {
                                if safe_remove(root, &bundle_stage).is_ok() {
                                    reclaimed = reclaimed.saturating_add(n);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 4. Sweep cas/.staging directory for orphaned temporary staging files
    let cas = crate::cas::ContentAddressableStorage::new(root);
    if let Ok(n) = cas.sweep_staging(std::time::Duration::from_secs(0)) {
        reclaimed = reclaimed.saturating_add(n);
    }

    let tx = sql(c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate))?;

    // Record tombstones for collected artifacts before deleting from managed_artifacts
    for (p, j) in &rows {
        let mut art_stmt = sql(tx.prepare(
            "SELECT name, bytes, sha256 FROM managed_artifacts WHERE project=?1 AND job=?2 AND pinned=0"
        ))?;
        let art_rows: Vec<(String, i64, String)> = sql(art_stmt.query_map(params![p, j], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        }))?.collect::<rusqlite::Result<Vec<_>>>().map_err(|e| e.to_string())?;
        drop(art_stmt);

        for (name, bytes, sha256) in art_rows {
            record_tombstone_conn(
                &tx,
                p,
                "artifact",
                &format!("{j}:{name}"),
                j,
                &sha256,
                bytes as u64,
                json!({
                    "job": j,
                    "name": name,
                    "sha256": sha256,
                    "bytes": bytes,
                }),
            )?;
        }
        sql(tx.execute("DELETE FROM managed_artifacts WHERE project=?1 AND job=?2 AND pinned=0", params![p, j]))?;
    }

    // Enforce explicit SQLite audit/metadata retention policy per project
    sql(tx.execute(
        "DELETE FROM events WHERE cursor NOT IN (
            SELECT cursor FROM events e2 WHERE e2.project = events.project ORDER BY e2.cursor DESC LIMIT ?1
        )",
        [MAX_RETAINED_EVENTS_PER_PROJECT],
    ))?;
    sql(tx.execute(
        "DELETE FROM checkpoints WHERE rowid NOT IN (
            SELECT rowid FROM checkpoints c2 WHERE c2.project = checkpoints.project AND c2.task = checkpoints.task ORDER BY c2.sequence DESC LIMIT ?1
        )",
        [MAX_RETAINED_CHECKPOINTS_PER_TASK],
    ))?;
    sql(tx.execute(
        "DELETE FROM agent_checkpoints WHERE rowid NOT IN (
            SELECT rowid FROM agent_checkpoints a2 WHERE a2.project = agent_checkpoints.project AND a2.task_id = agent_checkpoints.task_id ORDER BY a2.created_ms DESC, a2.rowid DESC LIMIT ?1
        )",
        [MAX_RETAINED_CHECKPOINTS_PER_TASK],
    ))?;

    // Prune older unpinned, non-scientific terminal jobs and their telemetry beyond MAX_RETAINED_TERMINAL_JOBS_PER_PROJECT
    let mut prune_stmt = sql(tx.prepare(
        "SELECT j.project, j.id, j.task, j.status, j.spec, j.submitted_ms FROM jobs j
         WHERE j.status IN ('succeeded','failed','timed_out','cancelled','rejected','interrupted')
           AND NOT EXISTS (SELECT 1 FROM managed_artifacts a WHERE a.project=j.project AND a.job=j.id AND a.pinned=1)
           AND NOT EXISTS (SELECT 1 FROM environment_leases l WHERE l.project=j.project AND l.job=j.id)
           AND NOT EXISTS (SELECT 1 FROM scientific_jobs s WHERE s.project=j.project AND s.job=j.id)
           AND NOT EXISTS (SELECT 1 FROM pipeline_receipts pr WHERE pr.project=j.project AND pr.job=j.id)
           AND j.rowid NOT IN (
               SELECT j2.rowid FROM jobs j2 WHERE j2.project=j.project ORDER BY j2.submitted_ms DESC, j2.rowid DESC LIMIT ?1
           )"
    ))?;
    let jobs_to_prune: Vec<(String, String, String, String, String, i64)> = sql(prune_stmt.query_map([MAX_RETAINED_TERMINAL_JOBS_PER_PROJECT], |r| {
        Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?))
    }))?.collect::<rusqlite::Result<Vec<_>>>().map_err(|e| e.to_string())?;
    drop(prune_stmt);

    for (p, j_id, task_id, status, spec, submitted_ms) in &jobs_to_prune {
        let spec_hash = hash(spec.as_bytes());
        record_tombstone_conn(
            &tx,
            p,
            "job",
            j_id,
            task_id,
            &spec_hash,
            0,
            json!({
                "job_id": j_id,
                "task_id": task_id,
                "status": status,
                "submitted_ms": submitted_ms,
                "spec_hash": spec_hash,
            }),
        )?;
    }

    sql(tx.execute(
        "DELETE FROM job_telemetry WHERE (project, job) IN (
            SELECT j.project, j.id FROM jobs j
            WHERE j.status IN ('succeeded','failed','timed_out','cancelled','rejected','interrupted')
              AND NOT EXISTS (SELECT 1 FROM managed_artifacts a WHERE a.project=j.project AND a.job=j.id AND a.pinned=1)
              AND NOT EXISTS (SELECT 1 FROM environment_leases l WHERE l.project=j.project AND l.job=j.id)
              AND NOT EXISTS (SELECT 1 FROM scientific_jobs s WHERE s.project=j.project AND s.job=j.id)
              AND NOT EXISTS (SELECT 1 FROM pipeline_receipts pr WHERE pr.project=j.project AND pr.job=j.id)
              AND j.rowid NOT IN (
                  SELECT j2.rowid FROM jobs j2 WHERE j2.project=j.project ORDER BY j2.submitted_ms DESC, j2.rowid DESC LIMIT ?1
              )
        )",
        [MAX_RETAINED_TERMINAL_JOBS_PER_PROJECT],
    ))?;
    sql(tx.execute(
        "DELETE FROM jobs WHERE rowid IN (
            SELECT j.rowid FROM jobs j
            WHERE j.status IN ('succeeded','failed','timed_out','cancelled','rejected','interrupted')
              AND NOT EXISTS (SELECT 1 FROM managed_artifacts a WHERE a.project=j.project AND a.job=j.id AND a.pinned=1)
              AND NOT EXISTS (SELECT 1 FROM environment_leases l WHERE l.project=j.project AND l.job=j.id)
              AND NOT EXISTS (SELECT 1 FROM scientific_jobs s WHERE s.project=j.project AND s.job=j.id)
              AND NOT EXISTS (SELECT 1 FROM pipeline_receipts pr WHERE pr.project=j.project AND pr.job=j.id)
              AND j.rowid NOT IN (
                  SELECT j2.rowid FROM jobs j2 WHERE j2.project=j.project ORDER BY j2.submitted_ms DESC, j2.rowid DESC LIMIT ?1
              )
        )",
        [MAX_RETAINED_TERMINAL_JOBS_PER_PROJECT],
    ))?;

    let mut proj_stmt = sql(tx.prepare("SELECT id FROM projects"))?;
    let project_ids: Vec<String> = sql(proj_stmt.query_map([], |r| r.get(0)))?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|e| e.to_string())?;
    drop(proj_stmt);

    for pid in &project_ids {
        // Compact older command request bodies to hash:SHA256 and bound total retained commands
        let mut cmd_stmt = sql(tx.prepare(
            "SELECT rowid, request FROM commands WHERE project=?1 ORDER BY rowid DESC"
        ))?;
        let cmd_rows: Vec<(i64, String)> = sql(cmd_stmt.query_map([pid], |r| Ok((r.get(0)?, r.get(1)?))))?
            .collect::<rusqlite::Result<Vec<_>>>().map_err(|e| e.to_string())?;
        drop(cmd_stmt);

        if cmd_rows.len() > MAX_ACTIVE_COMMAND_RAW_BODIES {
            for (rowid, req) in &cmd_rows[MAX_ACTIVE_COMMAND_RAW_BODIES..] {
                if !req.starts_with("hash:") {
                    let h = format!("hash:{}", hash(req.as_bytes()));
                    sql(tx.execute("UPDATE commands SET request=?1 WHERE rowid=?2", params![h, rowid]))?;
                }
            }
        }

        // Roll up excess settled charges/actions per project
        let mut ch_stmt = sql(tx.prepare(
            "SELECT c.action, c.amount, c.proof FROM charges c
             WHERE c.project=?1 AND NOT EXISTS (SELECT 1 FROM reservations r WHERE r.project=c.project AND r.action=c.action AND r.held=1)
             ORDER BY c.rowid DESC"
        ))?;
        let settled: Vec<(String, String, String)> = sql(ch_stmt.query_map([pid], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?))))?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| e.to_string())?;
        drop(ch_stmt);

        if settled.len() > MAX_RETAINED_SETTLED_ACTIONS_PER_PROJECT {
            let keep_idx = MAX_RETAINED_SETTLED_ACTIONS_PER_PROJECT - 1;
            let (anchor_action, anchor_amount_str, anchor_proof) = &settled[keep_idx];
            let mut rolled_amount: BTreeMap<String, i64> = serde_json::from_str(anchor_amount_str).map_err(|e|format!("invalid charge rollup: {e}"))?;
            let to_compact = &settled[MAX_RETAINED_SETTLED_ACTIONS_PER_PROJECT..];
            for (_, amt_str, _) in to_compact {
                let map=serde_json::from_str::<BTreeMap<String, i64>>(amt_str).map_err(|e|format!("invalid charge rollup: {e}"))?;
                for (k, v) in map {
                    let sum=rolled_amount.get(&k).copied().unwrap_or(0).checked_add(v).ok_or("charge rollup overflow")?;
                    if v<0 || sum<0 {return Err("negative charge rollup rejected".into());}
                    rolled_amount.insert(k,sum);
                }
            }
            let updated_proof = if anchor_proof.contains("rollup") {
                anchor_proof.clone()
            } else {
                format!("{anchor_proof} (+compacted_rollup)")
            };
            sql(tx.execute(
                "UPDATE charges SET amount=?3, proof=?4 WHERE project=?1 AND action=?2",
                params![pid, anchor_action, serde_json::to_string(&rolled_amount).map_err(|e| e.to_string())?, updated_proof],
            ))?;
            for (old_act, _, _) in to_compact {
                sql(tx.execute("DELETE FROM charges WHERE project=?1 AND action=?2", params![pid, old_act]))?;
                sql(tx.execute("DELETE FROM reservations WHERE project=?1 AND action=?2 AND held=0", params![pid, old_act]))?;
            }
        }

        // Compact settled actions beyond MAX_RETAINED_SETTLED_ACTIONS_PER_PROJECT into action_fences
        let mut act_stmt = sql(tx.prepare(
            "SELECT a.id, a.task, a.data FROM actions a
             WHERE a.project=?1
               AND json_extract(a.data, '$.status') IN ('succeeded', 'failed', 'cancelled')
               AND NOT EXISTS (SELECT 1 FROM reservations r WHERE r.project=a.project AND r.action=a.id AND r.held=1)
             ORDER BY a.rowid DESC"
        ))?;
        let settled_acts: Vec<(String, String, String)> = sql(act_stmt.query_map([pid], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?))))?
            .collect::<rusqlite::Result<Vec<_>>>().map_err(|e| e.to_string())?;
        drop(act_stmt);

        if settled_acts.len() > MAX_RETAINED_SETTLED_ACTIONS_PER_PROJECT {
            let to_compact_actions = &settled_acts[MAX_RETAINED_SETTLED_ACTIONS_PER_PROJECT..];
            for (act_id, task_id, act_data) in to_compact_actions {
                if let Ok(act) = serde_json::from_str::<Action>(act_data) {
                    let input_hash = if act.input.starts_with("hash:") {
                        act.input.clone()
                    } else {
                        format!("hash:{}", hash(act.input.as_bytes()))
                    };
                    let receipt_json = act.receipt.as_ref().map(|r| r.to_string());
                    sql(tx.execute(
                        "INSERT OR REPLACE INTO action_fences(project, action_id, task_id, generation, intent_revision, lease_id, input_hash, status, receipt_json, tombstoned_ms)
                         VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                        params![
                            pid,
                            act_id,
                            task_id,
                            act.generation,
                            act.intent_revision,
                            act.lease_id,
                            input_hash,
                            act.status,
                            receipt_json,
                            now_ms(),
                        ],
                    ))?;
                    record_tombstone_conn(
                        &tx,
                        pid,
                        "action",
                        act_id,
                        task_id,
                        &input_hash,
                        0,
                        json!({
                            "action_id": act_id,
                            "task_id": task_id,
                            "status": act.status,
                        }),
                    )?;
                    let mut compact_act = act.clone();
                    compact_act.input = input_hash;
                    compact_act.response = None;
                    compact_act.budget = BTreeMap::new();
                    compact_act.capacity = BTreeMap::new();
                    compact_act.uncertainty = None;
                    sql(tx.execute(
                        "UPDATE actions SET data=?1 WHERE project=?2 AND id=?3",
                        params![serde_json::to_string(&compact_act).map_err(|e| e.to_string())?, pid, act_id],
                    ))?;
                }
            }
        }

        // Bound action_fences per project
        sql(tx.execute(
            "DELETE FROM action_fences WHERE project=?1 AND rowid NOT IN (
                SELECT rowid FROM action_fences WHERE project=?1 ORDER BY tombstoned_ms DESC, rowid DESC LIMIT ?2
            )",
            params![pid, MAX_RETAINED_ACTION_FENCES_PER_PROJECT],
        ))?;

        // Bound tombstones per project
        sql(tx.execute(
            "DELETE FROM tombstones WHERE project=?1 AND rowid NOT IN (
                SELECT rowid FROM tombstones WHERE project=?1 ORDER BY tombstoned_ms DESC, rowid DESC LIMIT ?2
            )",
            params![pid, MAX_RETAINED_TOMBSTONES_PER_PROJECT],
        ))?;
    }

    sql(tx.execute("DELETE FROM scheduler WHERE key LIKE 'container.%' AND json_extract(value,'$.exit_confirmed')=1 AND NOT EXISTS(SELECT 1 FROM jobs j WHERE j.project=json_extract(scheduler.value,'$.project') AND j.id=json_extract(scheduler.value,'$.job'))",[]))?;
    tx.commit().map_err(|e|e.to_string())?;
    let _ = c.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);");
    let vacuum_space=fs::metadata(root.join("state.sqlite")).map(|m|m.len().saturating_mul(2)).unwrap_or(0);
    if enforce_state_write_admission(root,Some(&c),vacuum_space,StorageWriteKind::EmergencyRecovery,None).is_ok(){
        let _=c.execute_batch("VACUUM;");
    }else{tracing::debug!("VACUUM deferred: managed-state/disk headroom unavailable");}
    let db_after = fs::metadata(root.join("state.sqlite")).map(|m| m.len()).unwrap_or(0)
        + fs::metadata(root.join("state.sqlite-wal")).map(|m| m.len()).unwrap_or(0);
    reclaimed = reclaimed.saturating_add(db_before.saturating_sub(db_after));
    Ok(reclaimed)
}
fn tree_size(path:&Path)->Result<u64>{let mut total=0u64;for e in fs::read_dir(path).map_err(|e|e.to_string())?{let e=e.map_err(|e|e.to_string())?;let m=fs::symlink_metadata(e.path()).map_err(|e|e.to_string())?;if m.file_type().is_symlink(){return Err("managed artifact symlink rejected".into());}if m.is_file(){total=total.saturating_add(m.len());}else if m.is_dir(){total=total.saturating_add(tree_size(&e.path())?);}}Ok(total)}
fn sample_process(root:&Path,project:&str,job:&str,pid:u32)->Result<Option<ResourceSample>>{
    let output=match Command::new("/bin/ps").args(["-o","%cpu=","-o","rss=","-p",&pid.to_string()]).output(){Ok(o) if o.status.success()=>o,_=>return Ok(None)};
    let fields=String::from_utf8_lossy(&output.stdout).split_whitespace().map(str::to_owned).collect::<Vec<_>>();if fields.len()<2{return Ok(None);}
    let cpu=fields[0].parse::<f64>().ok().filter(|x|x.is_finite()&&*x>=0.0);let rss=fields[1].parse::<u64>().ok().map(|x|x.saturating_mul(1024));if cpu.is_none()&&rss.is_none(){return Ok(None);}
    let c=connect(root)?;let old:Option<u64>=sql(c.query_row("SELECT peak_rss_bytes FROM job_telemetry WHERE project=?1 AND job=?2",params![project,job],|r|r.get(0)).optional())?;
    let peak=old.unwrap_or(0).max(rss.unwrap_or(0));let sampled_ms=now_ms();
    enforce_state_write_admission(root,Some(&c),16*1024,StorageWriteKind::EmergencyRecovery,Some(project))?;
    sql(c.execute("INSERT INTO job_telemetry(project,job,sampled_ms,cpu_percent,rss_bytes,peak_rss_bytes) VALUES(?1,?2,?3,?4,?5,?6) ON CONFLICT(project,job) DO UPDATE SET sampled_ms=excluded.sampled_ms,cpu_percent=excluded.cpu_percent,rss_bytes=excluded.rss_bytes,peak_rss_bytes=MAX(job_telemetry.peak_rss_bytes,excluded.peak_rss_bytes)",params![project,job,sampled_ms,cpu,rss,peak]))?;
    Ok(Some(ResourceSample{sampled_ms,cpu_percent:cpu,rss_bytes:rss,peak_rss_bytes:peak}))
}

struct WorkerControl { _stdin:ChildStdin, logs:Vec<thread::JoinHandle<()>> }
impl WorkerControl {fn close_logs(self) {for handle in self.logs {let _=handle.join();}}}
fn bounded_log<R:Read+Send+'static>(mut reader:R,sink:Arc<std::sync::Mutex<(fs::File,u64)>>)->thread::JoinHandle<()> {
    thread::spawn(move||{let mut buffer=[0u8;8192];loop {let n=match reader.read(&mut buffer){Ok(0)|Err(_)=>break,Ok(n)=>n};
        if let Ok(mut sink)=sink.lock(){let allowed=n.min(LOG_CAP.saturating_sub(sink.1) as usize);if allowed>0 {let _=sink.0.write_all(&buffer[..allowed]);sink.1+=allowed as u64;}}
    }})
}
pub struct Supervisor { root:PathBuf, lock:fs::File, executable:PathBuf, max_workers:usize, children:Vec<(JobView,Child,WorkerControl,Instant,Instant)> }
fn default_worker_executable()->Result<PathBuf>{
    if let Some(p)=std::env::var_os("CARGO_BIN_EXE_mlrescurix-harness"){let pb=PathBuf::from(p);if pb.is_file(){return Ok(pb);}}
    let exe=std::env::current_exe().map_err(|e|e.to_string())?;
    if exe.file_name().and_then(|s|s.to_str()).is_some_and(|s|s.starts_with("mlrescurix-harness")){return Ok(exe);}
    if let Some(candidate)=exe.parent().and_then(|p|p.parent()).map(|p|p.join("mlrescurix-harness")){if candidate.is_file(){return Ok(candidate);}}
    Ok(exe)
}
impl Supervisor {
    pub fn start(root:&Path,max_workers:usize)->Result<Self>{
        Self::start_with_executable(root,max_workers,&default_worker_executable()?)
    }
    pub fn start_with_executable(root:&Path,max_workers:usize,executable:&Path)->Result<Self>{
        if !cfg!(target_os="macos"){return Err("M1 trusted runner is validated only on macOS".into());}
        if !(1..=64).contains(&max_workers){return Err("worker count must be 1..=64".into());}
        if fs::symlink_metadata(root.join("SUPERVISOR.lock")).is_ok_and(|m|m.file_type().is_symlink()){return Err("supervisor lock symlink rejected".into());}
        migrate(root)?;
        let lock=fs::OpenOptions::new().read(true).write(true).create(true).open(root.join("SUPERVISOR.lock")).map_err(|e|e.to_string())?;
        use std::os::fd::AsRawFd;
        if unsafe{libc::flock(lock.as_raw_fd(),libc::LOCK_EX|libc::LOCK_NB)}!=0{return Err("another supervisor owns this state".into());}
        let c=connect(root)?;
        reconcile_final_tests(root,&c)?;
        // Fence old workers before daemon discovery. The lifecycle lock prevents
        // a create racing this transition from escaping the survivor sweep.
        sql(c.execute("UPDATE jobs SET status='recovering',reason='supervisor restart; waiting for bounded worker deadline',recovered_deadline_ms=started_ms+CAST(json_extract(spec,'$.timeout_ms') AS INTEGER)+500 WHERE status IN('running','terminating')",[]))?;
        let _ = c.execute("UPDATE resource_leases SET status='recovering', updated_ms=?1 WHERE status IN ('reserved','active','heartbeated')", [now_ms()]);
        resources::recover(root)?;
        Ok(Self{root:root.to_path_buf(),lock,executable:executable.to_path_buf(),max_workers,children:vec![]})
    }
    pub fn serve(mut self)->Result<()> {
        let _exclusive_lock=&self.lock;
        STOP.store(false,Ordering::Relaxed);
        unsafe{libc::signal(libc::SIGINT,stop_signal as *const () as libc::sighandler_t);libc::signal(libc::SIGTERM,stop_signal as *const () as libc::sighandler_t);}
        loop {if STOP.load(Ordering::Relaxed){self.shutdown()?;return Ok(());}self.tick()?;thread::sleep(Duration::from_millis(POLL_MS));}
    }
    pub fn drain(mut self)->Result<()> {
        let _exclusive_lock=&self.lock;
        for _ in 0..1000 {
            self.tick()?;
            let c=connect(&self.root)?;
            let active_count:i64=sql(c.query_row("SELECT count(*) FROM jobs WHERE status IN('queued','running','terminating','recovering')", [], |r|r.get(0)))?;
            if active_count==0 {return Ok(());}
            thread::sleep(Duration::from_millis(POLL_MS));
        }
        Err("drain timed out waiting for active jobs to complete".into())
    }
    fn shutdown(&mut self)->Result<()>{
        while let Some((view,mut child,_control,started,_last_sample))=self.children.pop(){terminate(&mut child)?;_control.close_logs();self.finish(&view,"cancelled",Some("supervisor shutdown".into()),vec![],started.elapsed().as_millis().max(1) as u64)?;let stage=self.root.join("artifacts").join(&view.spec.project_id).join(format!(".{}.staging",view.spec.job_id));if stage.exists(){safe_remove(&self.root,&stage)?;}}
        Ok(())
    }
    pub fn tick(&mut self)->Result<()> {self.reconcile_queued()?;self.recover_expired()?;self.admit()?;self.reap()?;Ok(())}
    fn reconcile_queued(&self)->Result<()>{
        let c=connect(&self.root)?;let mut q=sql(c.prepare("SELECT j.project,j.id,j.generation,p.state,p.intent_revision,t.data FROM jobs j JOIN projects p ON p.id=j.project JOIN tasks t ON t.project=j.project AND t.id=j.task WHERE j.status='queued'"))?;
        let rows=sql(q.query_map([],|r|Ok((r.get::<_,String>(0)?,r.get::<_,String>(1)?,r.get::<_,u64>(2)?,r.get::<_,String>(3)?,r.get::<_,u64>(4)?,r.get::<_,String>(5)?))))?.collect::<rusqlite::Result<Vec<_>>>().map_err(|e|e.to_string())?;drop(q);
        for(p,id,generation,state,intent,task_data)in rows{let task:crate::core::Task=serde_json::from_str(&task_data).map_err(|e|e.to_string())?;let terminal=if state=="cancelled"{Some(("cancelled","project cancelled before admission"))}else if task.status!="active"||task.generation!=generation||task.intent_revision!=intent{Some(("rejected","task intent changed before admission"))}else{None};if let Some((status,reason))=terminal{let tx=sql(rusqlite::Transaction::new_unchecked(&c,rusqlite::TransactionBehavior::Immediate))?;sql(tx.execute("UPDATE jobs SET status=?3,reason=?4,ended_ms=?5 WHERE project=?1 AND id=?2 AND status='queued'",params![p,id,status,reason,now_ms()]))?;sql(tx.execute("INSERT INTO events(project,kind,payload) VALUES(?1,'job.terminal',?2)",params![p,json!({"job_id":id,"status":status,"reason":reason}).to_string()]))?;tx.commit().map_err(|e|e.to_string())?;}}
        Ok(())
    }
    fn recover_expired(&self)->Result<()> {
        let c=connect(&self.root)?;let mut q=sql(c.prepare("SELECT project,id,spec,stage FROM jobs WHERE status='recovering' AND recovered_deadline_ms<=?1"))?;
        let rows=sql(q.query_map([now_ms()],|r|Ok((r.get::<_,String>(0)?,r.get::<_,String>(1)?,r.get::<_,String>(2)?,r.get::<_,Option<String>>(3)?))))?.collect::<rusqlite::Result<Vec<_>>>().map_err(|e|e.to_string())?;drop(q);
        for (p,id,spec,stage) in rows {
            let s:JobSpec=serde_json::from_str(&spec).map_err(|e|e.to_string())?;resources::confirm_job_exit(&self.root,&s)?;let task_json:String=sql(c.query_row("SELECT data FROM tasks WHERE project=?1 AND id=?2",params![p,s.task_id],|r|r.get(0)))?;let task:crate::core::Task=serde_json::from_str(&task_json).map_err(|e|e.to_string())?;let (project_state,intent):(String,u64)=sql(c.query_row("SELECT state,intent_revision FROM projects WHERE id=?1",[&p],|r|Ok((r.get(0)?,r.get(1)?))))?;
            let is_finalize=matches!(&s.probe,Probe::Scientific{operation,..} if operation=="finalize");
            let eligible=(matches!(project_state.as_str(),"created"|"paused")||(is_finalize&&project_state=="finalizing"))&&task.status=="active"&&task.generation==s.generation&&task.intent_revision==intent;
            let final_dir=self.root.join("artifacts").join(&p).join(&id);let stage_path=stage.clone().map(PathBuf::from);
            let mut recovered=if eligible&&final_dir.is_dir(){manifest_dir(&final_dir,s.max_output_bytes).ok()}else{None};
            if recovered.is_none()&&eligible&&!final_dir.exists(){if let Some(path)=&stage_path{if path.is_dir(){if let Ok(artifacts)=manifest_dir(path,s.max_output_bytes){let spec_file=path.join("job.json");if spec_file.exists(){let _=fs::remove_file(spec_file);}if !artifacts.is_empty(){ensure_dir(&self.root,final_dir.parent().unwrap())?;if fs::rename(path,&final_dir).is_ok(){recovered=Some(artifacts);}}}}}}
            if recovered.is_none(){if final_dir.exists(){let _=safe_remove(&self.root,&final_dir);}if let Some(path)=stage_path{if path.exists(){let _=safe_remove(&self.root,&path);}}}
            let view=JobView{spec:s.clone(),status:"recovering".into(),reason:None,pid:None,submitted_ms:0,started_ms:None,ended_ms:None,artifacts:vec![],telemetry:None,resource_receipt:None};
            if let Some(artifacts)=recovered{self.finish(&view,"succeeded",Some("recovered published output after supervisor restart".into()),artifacts,s.timeout_ms)?;}
            else{let tx=sql(rusqlite::Transaction::new_unchecked(&c,rusqlite::TransactionBehavior::Immediate))?;sql(tx.execute("UPDATE jobs SET status='interrupted',reason='worker deadline elapsed while supervisor was stopped',ended_ms=?3,pid=NULL,stage=NULL WHERE project=?1 AND id=?2 AND status='recovering'",params![p,id,now_ms()]))?;sql(tx.execute("DELETE FROM environment_leases WHERE project=?1 AND job=?2",params![p,id]))?;settle_job_action(&tx,&p,&id,s.timeout_ms,"interrupted")?;let _=crate::broker::release_lease(&tx,&format!("lease-{p}-{id}"),Some("recovering job interrupted on deadline"));sql(tx.execute("INSERT INTO events(project,kind,payload) VALUES(?1,'job.terminal',?2)",params![p,json!({"job_id":id,"status":"interrupted"}).to_string()]))?;tx.commit().map_err(|e|e.to_string())?;}
        }Ok(())
    }
    fn admit(&mut self)->Result<()> {
        let mut skipped_projects = std::collections::HashSet::new();
        while self.children.len()<self.max_workers {
            let c=connect(&self.root)?;let tx=sql(rusqlite::Transaction::new_unchecked(&c,rusqlite::TransactionBehavior::Immediate))?;
            let cursor:Option<String>=sql(tx.query_row("SELECT value FROM scheduler WHERE key='last_project'",[],|r|r.get(0)).optional())?;
            let mut eligible="j.status='queued' AND (p.state='created' OR (p.state='finalizing' AND EXISTS(SELECT 1 FROM scientific_jobs sj WHERE sj.project=j.project AND sj.job=j.id AND sj.operation='finalize'))) AND NOT EXISTS(SELECT 1 FROM questions q WHERE q.project=p.id AND q.answer IS NULL) AND NOT EXISTS(SELECT 1 FROM actions a WHERE a.project=j.project AND a.task=j.task AND json_extract(a.data,'$.status') NOT IN('succeeded','failed','cancelled'))".to_string();
            if !skipped_projects.is_empty() {
                let quoted: Vec<String> = skipped_projects.iter().map(|p: &String| format!("'{}'", p.replace('\'', "''"))).collect();
                eligible.push_str(&format!(" AND j.project NOT IN ({})", quoted.join(",")));
            }
            let rank="((SELECT count(*) FROM jobs prior WHERE prior.project=j.project AND prior.status IN('running','succeeded','failed','timed_out','cancelled','rejected','interrupted'))*1.0/COALESCE((SELECT weight FROM project_weights w WHERE w.project=j.project),1) - (?2-j.submitted_ms)/60000.0)";
            let now=now_ms();let row:Option<(String,String,String)>=sql(tx.query_row(
                &format!("SELECT j.id,j.project,j.spec FROM jobs j JOIN projects p ON p.id=j.project WHERE {eligible} AND j.project>?1 ORDER BY {rank},j.project,j.submitted_ms,j.id LIMIT 1"),rusqlite::params![cursor.as_deref().unwrap_or(""),now],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional())?;
            let row=match row {Some(x)=>Some(x),None=>sql(tx.query_row(&format!("SELECT j.id,j.project,j.spec FROM jobs j JOIN projects p ON p.id=j.project WHERE {eligible} ORDER BY {rank},j.project,j.submitted_ms,j.id LIMIT 1"),rusqlite::params![cursor.as_deref().unwrap_or(""),now],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional())?};
            let Some((id,p,spec))=row else {tx.commit().map_err(|e|e.to_string())?;break};
            let s:JobSpec=serde_json::from_str(&spec).map_err(|e|e.to_string())?;

            let is_finalize = matches!(&s.probe, Probe::Scientific { operation, .. } if operation == "finalize");
            if !is_finalize && crate::broker::is_under_pressure(&self.root).unwrap_or(false) {
                tracing::warn!("system under pressure; deferring non-finalization job admission");
                tx.commit().map_err(|e| e.to_string())?;
                break;
            }

            if let Err(reason)=validate_finalization_job(&self.root,&s).and_then(|_|resources::check_certification_request(&s)).and_then(|_|crate::broker::check_request_feasibility(&self.root,&s)).and_then(|_|admission(&tx,&s,self.max_workers)) {
                if reason.starts_with("project capacity unavailable:") || reason=="project worker capacity unavailable" {
                    skipped_projects.insert(p.clone());
                    tx.execute("INSERT INTO scheduler(key,value) VALUES('last_project',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",[&p]).map_err(|e|e.to_string())?;
                    tx.commit().map_err(|e|e.to_string())?;
                    continue;
                }
                if reason.starts_with("capacity unavailable:") || reason=="device worker capacity unavailable" || reason=="worker slot unavailable" {
                    tx.execute("INSERT INTO scheduler(key,value) VALUES('last_project',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",[&p]).map_err(|e|e.to_string())?;
                    tx.commit().map_err(|e|e.to_string())?;
                    break;
                }
                sql(tx.execute("UPDATE jobs SET status='rejected',reason=?3,ended_ms=?4 WHERE project=?1 AND id=?2",params![p,id,reason,now_ms()]))?;
                tx.commit().map_err(|e|e.to_string())?;continue;
            }
            if let Some(environment)=&s.environment_id {
                let status:Option<String>=sql(tx.query_row("SELECT status FROM environments WHERE id=?1",[environment],|r|r.get(0)).optional())?;
                if status.as_deref()!=Some("ready"){sql(tx.execute("UPDATE jobs SET status='rejected',reason='requested environment is not ready',ended_ms=?3 WHERE project=?1 AND id=?2",params![p,id,now_ms()]))?;tx.commit().map_err(|e|e.to_string())?;continue;}
            }
            let task:String=sql(tx.query_row("SELECT data FROM tasks WHERE project=?1 AND id=?2",params![p,s.task_id],|r|r.get(0)))?;
            let t:crate::core::Task=serde_json::from_str(&task).map_err(|e|e.to_string())?;
            let intent:u64=sql(tx.query_row("SELECT intent_revision FROM projects WHERE id=?1",[&p],|r|r.get(0)))?;
            if t.status!="active"||t.generation!=s.generation||t.intent_revision!=intent||t.expires_ms.is_none_or(|end|end<now_ms().saturating_add(s.timeout_ms as i64)) {
                sql(tx.execute("UPDATE jobs SET status='rejected',reason='stale task fence',ended_ms=?3 WHERE project=?1 AND id=?2",params![p,id,now_ms()]))?;tx.commit().map_err(|e|e.to_string())?;continue;
            }
            let action_id=format!("job.{}",s.job_id);let action_exists:bool=sql(tx.query_row("SELECT EXISTS(SELECT 1 FROM actions WHERE project=?1 AND id=?2 UNION SELECT 1 FROM action_fences WHERE project=?1 AND action_id=?2)",params![p,action_id],|r|r.get(0)))?;
            if action_exists{sql(tx.execute("UPDATE jobs SET status='rejected',reason='job action identity already exists',ended_ms=?3 WHERE project=?1 AND id=?2",params![p,id,now_ms()]))?;tx.commit().map_err(|e|e.to_string())?;continue;}
            if let Err(e) = ensure_disk_space(&self.root, s.max_output_bytes.saturating_add(LOG_CAP).saturating_add(64*1024)) {
                tracing::warn!(reason = %e, "disk space constrained; deferring job admission");
                tx.commit().map_err(|e| e.to_string())?;
                break;
            }
            let stage=self.root.join("artifacts").join(&p).join(format!(".{}.staging",id));
            ensure_dir(&self.root,stage.parent().unwrap())?;if stage.exists(){safe_remove(&self.root,&stage)?;}ensure_dir(&self.root,&stage)?;
            let log=self.root.join("logs").join(&p).join(format!("{id}.log"));ensure_dir(&self.root,log.parent().unwrap())?;
            let out=fs::File::create(&log).map_err(|e|e.to_string())?;let sink=Arc::new(std::sync::Mutex::new((out,0u64)));
            let spec_path=stage.join("job.json");fs::write(&spec_path,&spec).map_err(|e|e.to_string())?;
            let mut cmd=Command::new(&self.executable);cmd.arg("__worker").arg(&spec_path).arg(&stage).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).env_clear().env("PATH","/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin").env("MLRRESCURIX_STATE",&self.root);
            for key in ["HOME","DOCKER_HOST","DOCKER_CONTEXT","DOCKER_CONFIG"] {if let Some(value)=std::env::var_os(key){cmd.env(key,value);}}
            crate::broker::apply_thread_caps_to_command(&mut cmd, s.threads);
            if let Probe::Scientific{dataset_id,..}=&s.probe {
                let env_id=s.environment_id.as_deref().ok_or("scientific environment missing")?;
                let python=self.root.join("environments").join(env_id).join("bin/python3");
                let environment_status:String=sql(tx.query_row("SELECT status FROM environments WHERE id=?1",[env_id],|r|r.get(0)))?;
                let container:Option<(String,String)>=sql(tx.query_row("SELECT python,identity FROM environments WHERE id=?1 AND python LIKE 'docker:%'",[env_id],|r|Ok((r.get(0)?,r.get(1)?))).optional())?;
                if environment_status!="ready"||(!python.is_file()&&container.is_none()){return Err("managed scientific interpreter is unavailable".into());}
                let (data_path,sha):(String,String)=sql(tx.query_row("SELECT path,sha256 FROM scientific_datasets WHERE project=?1 AND id=?2",params![p,dataset_id],|r|Ok((r.get(0)?,r.get(1)?))))?;
                let expected=self.root.join("datasets").join(&p).join(format!("{dataset_id}.csv"));
                if Path::new(&data_path)!=expected||hash(&fs::read(&expected).map_err(|e|e.to_string())?)!=sha{return Err("registered dataset snapshot failed integrity check".into());}
                let workflow=self.root.join("workflows").join(&p);ensure_dir(&self.root,&workflow)?;
                let pythonpath=crate::assets::ensure_python_src(&self.root)?;
                let runtime_source=if container.is_some() {
                    let source_hash=crate::assets::embedded_python_src_hash();
                    let base=self.root.join("scientific-sources");ensure_dir(&self.root,&base)?;
                    let snapshot=base.join(&source_hash);
                    if !snapshot.exists() {
                        if fs::read_dir(&base).map_err(|e|e.to_string())?.count()>=8||tree_size(&pythonpath)?>1024*1024{return Err("bounded scientific source snapshot storage ceiling reached".into());}
                        crate::agents::workspace::BranchWorkspaceManager::copy_dir_recursive(&pythonpath,&snapshot)?;
                    }
                    if crate::agents::workspace::BranchWorkspaceManager::compute_dir_hash(&snapshot)?!=source_hash{return Err("scientific source snapshot failed content verification".into());}
                    snapshot
                } else {pythonpath};
                cmd.env("MLRRESCURIX_PYTHON",python).env("MLRRESCURIX_DATASET",expected)
                    .env("MLRRESCURIX_WORKFLOW",workflow).env("MLRRESCURIX_PYTHONPATH",runtime_source);
                if let Some((image,identity))=container {
                    cmd.env("MLRESCURIX_SCIENTIFIC_IMAGE",image.trim_start_matches("docker:")).env("MLRESCURIX_ENVIRONMENT_IDENTITY",identity);
                } else if matches!(&s.probe,Probe::Scientific{operation,..} if operation=="pipeline"||operation=="finalize") {
                    return Err("full pipelines and finalization require a verified container environment; host fallback prohibited".into());
                }
            }
            cmd.process_group(0);
            cmd.stdin(Stdio::piped());
            let (budget,capacity)=reservation(&tx,&p,&s)?;
            let lease_id=t.lease_id.clone().ok_or("task has no execution lease")?;let deadline=now_ms().saturating_add(s.timeout_ms as i64);
            let mut action=Action{action_id:action_id.clone(),task_id:s.task_id.clone(),generation:s.generation,intent_revision:t.intent_revision,lease_id,input:spec.clone(),budget:budget.clone(),capacity:capacity.clone(),deadline_ms:deadline,status:"dispatched".into(),attempt_id:Some(format!("{}.1",s.job_id)),external_request_id:Some(s.job_id.clone()),response:None,receipt:None,uncertainty:None};
            sql(tx.execute("INSERT INTO actions(project,id,task,data) VALUES(?1,?2,?3,?4)",params![p,action_id,s.task_id,serde_json::to_string(&action).map_err(|e|e.to_string())?]))?;
            sql(tx.execute("INSERT INTO reservations(project,action,budget,capacity,held) VALUES(?1,?2,?3,?4,1)",params![p,action_id,serde_json::to_string(&budget).map_err(|e|e.to_string())?,serde_json::to_string(&capacity).map_err(|e|e.to_string())?]))?;
            let r_lease = crate::broker::ResourceLease {
                lease_id: format!("lease-{p}-{id}"),
                project: p.clone(),
                task: s.task_id.clone(),
                job: id.clone(),
                generation: s.generation,
                status: crate::broker::LeaseStatus::Reserved,
                backend: if s.environment_id.is_some() { "container".into() } else { "process".into() },
                device: "local".into(),
                cpu_millicores: s.threads * 1000,
                library_threads: s.threads,
                memory_bytes: s.resources.memory_bytes,
                accelerator_type: None,
                accelerator_units: 0,
                scratch_bytes: 64 * 1024 * 1024,
                output_bytes: s.max_output_bytes,
                pid_cap: 64,
                deadline_ms: deadline,
                heartbeat_ms: now,
                environment_hash: s.environment_id.clone(),
                created_ms: now,
                updated_ms: now,
            };
            crate::broker::reserve_lease(&tx, &r_lease, Some("job admitted"))?;
            let mut child=match cmd.spawn(){Ok(c)=>c,Err(e)=>{safe_remove(&self.root,&stage)?;let _=crate::broker::release_lease(&tx,&format!("lease-{p}-{id}"),Some("spawn failed"));sql(tx.execute("UPDATE jobs SET status='failed',reason=?3,ended_ms=?4 WHERE project=?1 AND id=?2",params![p,id,format!("worker launch failed: {e}"),now_ms()]))?;tx.commit().map_err(|e|e.to_string())?;continue;}};
            let log_threads=vec![bounded_log(child.stdout.take().ok_or("stdout unavailable")?,sink.clone()),bounded_log(child.stderr.take().ok_or("stderr unavailable")?,sink)];
            sql(tx.execute("UPDATE jobs SET status='running',action=?3,pid=?4,started_ms=?5,stage=?6,log_path=?7 WHERE project=?1 AND id=?2",params![p,id,action_id,child.id(),now_ms(),stage.to_string_lossy(),log.to_string_lossy()]))?;
            sql(tx.execute("INSERT INTO scheduler(key,value) VALUES('last_project',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",[&p]))?;
            sql(tx.execute("INSERT INTO events(project,kind,payload) VALUES(?1,'job.started',?2)",params![p,json!({"job_id":id,"pid":child.id()}).to_string()]))?;
            if let Some(environment)=&s.environment_id {sql(tx.execute("INSERT INTO environment_leases(environment,project,job) VALUES(?1,?2,?3)",params![environment,p,id]))?;}
            tx.execute("INSERT INTO events(project,kind,payload) VALUES(?1,'action.dispatched',?2)",params![p,json!({"action_id":action_id,"job_id":id}).to_string()]).map_err(|e|e.to_string())?;
            tx.commit().map_err(|e|e.to_string())?;
            let _ = crate::broker::transition_lease(&c, &format!("lease-{p}-{id}"), crate::broker::LeaseStatus::Active, Some("worker started"));
            let mut control=child.stdin.take().ok_or("worker start gate unavailable")?;
            if let Err(e)=control.write_all(&[1]) {action.status="outcome_unknown".into();action.uncertainty=Some(format!("worker start gate response lost: {e}"));save_action(&self.root,&p,&action)?;let _=terminate(&mut child);}
            let control=WorkerControl{_stdin:control,logs:log_threads};
            let started=Instant::now();self.children.push((JobView{spec:s,status:"running".into(),reason:None,pid:Some(child.id()),submitted_ms:0,started_ms:Some(now_ms()),ended_ms:None,artifacts:vec![],telemetry:None,resource_receipt:None},child,control,started,started-Duration::from_secs(1)));
        }Ok(())
    }
    fn reap(&mut self)->Result<()> {
        let mut index=0;
        let c = connect(&self.root)?;
        while index<self.children.len(){
            let (mut view,mut child,control,started,mut last_sample)=self.children.swap_remove(index);let elapsed=started.elapsed();
            let _ = crate::broker::heartbeat_lease(&c, &format!("lease-{}-{}", view.spec.project_id, view.spec.job_id));
            if last_sample.elapsed()>=Duration::from_secs(1){if let Some(sample)=sample_process(&self.root,&view.spec.project_id,&view.spec.job_id,child.id())?{view.telemetry=Some(sample);}last_sample=Instant::now();}
            let cancel:bool={let (requested,state,intent,task):(bool,String,u64,String)=sql(c.query_row("SELECT j.cancel_requested,p.state,p.intent_revision,t.data FROM jobs j JOIN projects p ON p.id=j.project JOIN tasks t ON t.project=j.project AND t.id=j.task WHERE j.project=?1 AND j.id=?2",params![view.spec.project_id,view.spec.job_id],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?))))?;let t:crate::core::Task=serde_json::from_str(&task).map_err(|e|e.to_string())?;requested||state=="cancelled"||t.generation!=view.spec.generation||t.intent_revision!=intent||t.status!="active"||t.expires_ms.is_none_or(|end|end<=now_ms())};
            let timed_out=elapsed>Duration::from_millis(view.spec.timeout_ms);
            if (cancel||timed_out)&&child.try_wait().map_err(|e|e.to_string())?.is_none(){terminate(&mut child)?;}
            if child.try_wait().map_err(|e|e.to_string())?.is_some(){
                let status=child.wait().map_err(|e|e.to_string())?;control.close_logs();
                let _ = crate::broker::transition_lease(&c, &format!("lease-{}-{}", view.spec.project_id, view.spec.job_id), crate::broker::LeaseStatus::Reaped, Some("process exited"));
                let reason=if cancel{Some("cancelled".to_string())}else if timed_out||status.code()==Some(124){Some("deadline exceeded".to_string())}else if !status.success(){Some(format!("worker exited with {status}"))}else{None};
                let (artifacts,final_reason)=if reason.is_none(){match self.publish(&view){Ok(a)=>(a,None),Err(e)=>{let stage=self.root.join("artifacts").join(&view.spec.project_id).join(format!(".{}.staging",view.spec.job_id));if stage.exists(){let _=safe_remove(&self.root,&stage);} (vec![],Some(format!("artifact validation: {e}")))}}}else{(vec![],reason)};
                let ok=final_reason.is_none();if !ok{let stage=self.root.join("artifacts").join(&view.spec.project_id).join(format!(".{}.staging",view.spec.job_id));if stage.exists(){safe_remove(&self.root,&stage)?;}}
                self.finish(&view,if ok{"succeeded"}else if cancel{"cancelled"}else if timed_out||status.code()==Some(124){"timed_out"}else{"failed"},final_reason,artifacts,elapsed.as_millis().max(1) as u64)?;
            }else{self.children.push((view,child,control,started,last_sample));index+=1;}
        }Ok(())
    }
    pub fn publish(&self,v:&JobView)->Result<Vec<Value>>{
        if v.spec.resources.require_memory_certification {
            let receipt=resources::inspect(&self.root,&v.spec.project_id,&v.spec.job_id)?.ok_or("required memory certification receipt missing")?;
            if receipt["measurement_complete"]!=true {return Err("required memory certification receipt incomplete".into());}
        }
        if matches!(&v.spec.probe,Probe::Scientific{operation,..} if operation=="pipeline") {verify_pipeline_receipt(&self.root,&v.spec)?;}
        let c=connect(&self.root)?;let task:String=sql(c.query_row("SELECT data FROM tasks WHERE project=?1 AND id=?2",params![v.spec.project_id,v.spec.task_id],|r|r.get(0)))?;let t:crate::core::Task=serde_json::from_str(&task).map_err(|e|e.to_string())?;let (state,intent):(String,u64)=sql(c.query_row("SELECT state,intent_revision FROM projects WHERE id=?1",[&v.spec.project_id],|r|Ok((r.get(0)?,r.get(1)?))))?;
        let is_finalize=matches!(&v.spec.probe,Probe::Scientific{operation,..} if operation=="finalize");
        if !(matches!(state.as_str(),"created"|"paused")||(is_finalize&&state=="finalizing"))||t.status!="active"||t.generation!=v.spec.generation||t.intent_revision!=intent{return Err("stale worker result fenced before publication".into());}

        let stage=self.root.join("artifacts").join(&v.spec.project_id).join(format!(".{}.staging",v.spec.job_id));
        let final_dir=self.root.join("artifacts").join(&v.spec.project_id).join(&v.spec.job_id);

        if final_dir.is_dir() {
            if let Ok(existing_artifacts) = manifest_dir(&final_dir, v.spec.max_output_bytes) {
                if stage.exists() {
                    let _ = safe_remove(&self.root, &stage);
                }
                return Ok(existing_artifacts);
            }
            return Err("artifact destination already exists and is invalid".into());
        }

        if !stage.exists() {
            return Err("worker artifact staging directory missing".into());
        }

        let mut artifacts=vec![];let mut total=0u64;
        for entry in fs::read_dir(&stage).map_err(|e|e.to_string())? {let entry=entry.map_err(|e|e.to_string())?;let meta=fs::symlink_metadata(entry.path()).map_err(|e|e.to_string())?;if meta.file_type().is_symlink()||!meta.is_file(){return Err("worker emitted unsupported artifact type".into());}let name=entry.file_name().to_string_lossy().into_owned();if name=="job.json"{continue;}let bytes=fs::read(entry.path()).map_err(|e|e.to_string())?;total=total.checked_add(bytes.len() as u64).ok_or("artifact size overflow")?;if total>v.spec.max_output_bytes{return Err("worker exceeded artifact output limit".into());}artifacts.push(json!({"name":name,"bytes":bytes.len(),"sha256":hash(&bytes)}));}

        if artifacts.is_empty(){safe_remove(&self.root,&stage)?;return Ok(artifacts);}
        let spec_path=stage.join("job.json");if spec_path.exists(){fs::remove_file(spec_path).map_err(|e|e.to_string())?;}
        fs::rename(&stage,&final_dir).map_err(|e|e.to_string())?;Ok(artifacts)
    }
    fn finish(&self,v:&JobView,status:&str,reason:Option<String>,artifacts:Vec<Value>,elapsed_ms:u64)->Result<()>{
        resources::confirm_job_exit(&self.root,&v.spec)?;
        let c=connect(&self.root)?;let tx=sql(rusqlite::Transaction::new_unchecked(&c,rusqlite::TransactionBehavior::Immediate))?;
        enforce_state_write_admission(&self.root,Some(&tx),32*1024,StorageWriteKind::EmergencyRecovery,Some(&v.spec.project_id))?;
        if let Probe::Scientific{operation,dataset_id:_,candidate_id,test_access_id,spec}=&v.spec.probe {
            let workflow=self.root.join("workflows").join(&v.spec.project_id);
            let result_path=if operation=="trial"||operation=="pipeline"{
                let exp_id=spec.as_ref().and_then(|s|s.get("experiment_id")).and_then(|v|v.as_str()).unwrap_or("");
                workflow.join("trials").join(exp_id).join("trial.json")
            } else {
                let result_name=match operation.as_str(){"audit"=>"dataset_manifest.json","baselines"=>"results.json","finalize"=>"final.json",_=>""};
                workflow.join(result_name)
            };
            let result_hash=if status=="succeeded"&&result_path.is_file(){Some(hash(&fs::read(&result_path).map_err(|e|e.to_string())?))}else{None};
            sql(tx.execute("UPDATE scientific_jobs SET result_sha256=?3 WHERE project=?1 AND job=?2",params![v.spec.project_id,v.spec.job_id,result_hash]))?;
            if operation=="pipeline"&&status=="succeeded" {
                let receipt=verify_pipeline_receipt(&self.root,&v.spec)?;
                sql(tx.execute("INSERT INTO pipeline_receipts(project,job,candidate,dataset_sha256,source_sha256,environment_identity,protocol_sha256,model_sha256,predictions_sha256,oof_sha256,receipt_json) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",params![v.spec.project_id,v.spec.job_id,receipt["experiment_id"].as_str().unwrap(),receipt["dataset_sha256"].as_str().unwrap(),receipt["source_sha256"].as_str().unwrap(),receipt["environment_identity"].as_str().unwrap(),hash(serde_json::to_string(&receipt["protocol"]).unwrap().as_bytes()),receipt["model_sha256"].as_str().unwrap(),receipt["predictions_sha256"].as_str().unwrap(),receipt["oof_sha256"].as_str().unwrap(),receipt.to_string()]))?;
            }
            if operation=="finalize"&&status=="succeeded" {
                let access=test_access_id.as_deref().ok_or("final test key missing")?;let candidate=candidate_id.as_deref().ok_or("candidate missing")?;
                let gate=serde_json::from_str::<Value>(&fs::read_to_string(workflow.join("test_access.json")).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
                let final_result=serde_json::from_str::<Value>(&fs::read_to_string(&result_path).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
                if gate["test_access_id"]!=access||gate["candidate_id"]!=candidate||gate["status"]!="consumed"||final_result["test_access_id"]!=access{return Err("final assessment identity failed verification".into());}
                verify_finalized_pipeline_bundle(&self.root,&v.spec.project_id,candidate,&gate)?;
                let bundle_manifest_path=workflow.join("bundle/manifest.json");
                if bundle_manifest_path.is_file() {
                    let bm_bytes=fs::read(&bundle_manifest_path).map_err(|e|e.to_string())?;
                    let bm_hash = hash(&bm_bytes);
                    sql(tx.execute("INSERT OR REPLACE INTO managed_artifacts(project,job,name,bytes,sha256,pinned) VALUES(?1,?2,'bundle/manifest.json',?3,?4,1)",params![v.spec.project_id,v.spec.job_id,bm_bytes.len() as u64,&bm_hash]))?;
                    let cas = crate::cas::ContentAddressableStorage::new(&self.root);
                    let _ = cas.store_blob(&bm_bytes, "application/json", Some(&v.spec.project_id));
                    if bm_hash.len() == 64 {
                        let now = now_ms();
                        let _ = tx.execute(
                            "INSERT OR IGNORE INTO blobs(sha256, bytes, media_type, created_ms) VALUES(?1, ?2, 'application/json', ?3)",
                            params![bm_hash, bm_bytes.len() as u64, now],
                        );
                        let _ = tx.execute(
                            "INSERT OR IGNORE INTO artifacts(project, artifact_id, kind, producer_authority, current_version_id, pinned, visibility, created_ms, updated_ms)
                             VALUES(?1, 'bundle/manifest.json', 'apex_bundle', 'evaluator:finalizer', NULL, 1, 'project', ?2, ?2)",
                            params![v.spec.project_id, now],
                        );
                        let version_id = format!("{}_bundle_manifest", v.spec.job_id);
                        let index_digest = format!("[bundle/manifest.json] final delivery manifest ({} B, sha:{})", bm_bytes.len(), &bm_hash[..8.min(bm_hash.len())]);
                        let _ = tx.execute(
                            "INSERT OR REPLACE INTO artifact_versions(project, version_id, artifact_id, blob_sha256, index_digest, summary_text, schema_version, metadata_json, source_job_id, created_ms)
                             VALUES(?1, ?2, 'bundle/manifest.json', ?3, ?4, NULL, 1, '{}', ?5, ?6)",
                            params![v.spec.project_id, version_id, bm_hash, index_digest, v.spec.job_id, now],
                        );
                        let _ = tx.execute(
                            "UPDATE artifacts SET current_version_id=?2, pinned=1, updated_ms=?3 WHERE project=?1 AND artifact_id='bundle/manifest.json'",
                            params![v.spec.project_id, version_id, now],
                        );
                    }
                }
                let test_hash=gate["dataset_sha256"].as_str().ok_or("test gate lacks dataset hash")?;
                sql(tx.execute("UPDATE final_test_gates SET frozen_model_sha256=?4,status='consumed',result_sha256=?5 WHERE project=?1 AND access_id=?2 AND dataset_sha256=?3 AND status='reserved'",params![v.spec.project_id,access,test_hash,gate["frozen_model_sha256"].as_str().unwrap_or(""),hash(&fs::read(&result_path).map_err(|e|e.to_string())?)]))?;
                sql(tx.execute("INSERT INTO events(project,kind,payload) VALUES(?1,'final_test.consumed',?2)",params![v.spec.project_id,json!({"job_id":v.spec.job_id,"candidate_id":candidate,"test_access_id":access,"result_sha256":result_hash}).to_string()]))?;
            } else if operation=="finalize" && status!="succeeded" && !workflow.join("test_access.json").exists() {
                if let Some(access)=test_access_id.as_deref(){
                    sql(tx.execute("DELETE FROM final_test_gates WHERE project=?1 AND access_id=?2 AND status='reserved'",params![v.spec.project_id,access]))?;
                    sql(tx.execute("INSERT INTO events(project,kind,payload) VALUES(?1,'final_test.released',?2)",params![v.spec.project_id,json!({"job_id":v.spec.job_id,"test_access_id":access,"reason":"failed before test access"}).to_string()]))?;
                }
            }
        }
        sql(tx.execute("UPDATE jobs SET status=?3,reason=?4,ended_ms=?5,pid=NULL,artifacts=?6 WHERE project=?1 AND id=?2",params![v.spec.project_id,v.spec.job_id,status,reason,now_ms(),serde_json::to_string(&artifacts).map_err(|e|e.to_string())?]))?;
        settle_job_action(&tx,&v.spec.project_id,&v.spec.job_id,elapsed_ms,status)?;
        sql(tx.execute("DELETE FROM environment_leases WHERE project=?1 AND job=?2",params![v.spec.project_id,v.spec.job_id]))?;
        let _ = crate::broker::release_lease(&tx, &format!("lease-{}-{}", v.spec.project_id, v.spec.job_id), Some("job finished"));
        let cas = crate::cas::ContentAddressableStorage::new(&self.root);
        for a in &artifacts {
            let name = a["name"].as_str().unwrap_or("");
            let bytes = a["bytes"].as_u64().unwrap_or(0);
            let sha = a["sha256"].as_str().unwrap_or("");
            sql(tx.execute("INSERT INTO managed_artifacts(project,job,name,bytes,sha256) VALUES(?1,?2,?3,?4,?5)",params![v.spec.project_id,v.spec.job_id,name,bytes,sha]))?;

            let file_path = self.root.join("artifacts").join(&v.spec.project_id).join(&v.spec.job_id).join(name);
            if file_path.is_file() {
                let _ = cas.store_file(&file_path, "application/octet-stream", Some(&v.spec.project_id));
            }

            if sha.len() == 64 {
                let now = now_ms();
                let _ = tx.execute(
                    "INSERT OR IGNORE INTO blobs(sha256, bytes, media_type, created_ms) VALUES(?1, ?2, 'application/octet-stream', ?3)",
                    params![sha, bytes, now],
                );
                let _ = tx.execute(
                    "INSERT OR IGNORE INTO artifacts(project, artifact_id, kind, producer_authority, current_version_id, pinned, visibility, created_ms, updated_ms)
                     VALUES(?1, ?2, 'code', 'agent:worker', NULL, 0, 'project', ?3, ?3)",
                    params![v.spec.project_id, name, now],
                );
                let version_id = format!("{}_{}", v.spec.job_id, name.replace('/', "_").replace('.', "_"));
                let index_digest = format!("[{name}] output of job {} ({} B, sha:{})", v.spec.job_id, bytes, &sha[..8.min(sha.len())]);
                let _ = tx.execute(
                    "INSERT OR REPLACE INTO artifact_versions(project, version_id, artifact_id, blob_sha256, index_digest, summary_text, schema_version, metadata_json, source_job_id, created_ms)
                     VALUES(?1, ?2, ?3, ?4, ?5, NULL, 1, '{}', ?6, ?7)",
                    params![v.spec.project_id, version_id, name, sha, index_digest, v.spec.job_id, now],
                );
                let _ = tx.execute(
                    "UPDATE artifacts SET current_version_id=?2, updated_ms=?3 WHERE project=?1 AND artifact_id=?4",
                    params![v.spec.project_id, version_id, now, name],
                );
            }
        }
        if let Some(log)=sql(tx.query_row("SELECT log_path FROM jobs WHERE project=?1 AND id=?2",params![v.spec.project_id,v.spec.job_id],|r|r.get::<_,Option<String>>(0)))? {if let Ok(m)=fs::metadata(&log){if m.len()>LOG_CAP {let f=fs::OpenOptions::new().write(true).open(log).map_err(|e|e.to_string())?;f.set_len(LOG_CAP).map_err(|e|e.to_string())?;}}}
        sql(tx.execute("INSERT INTO events(project,kind,payload) VALUES(?1,'job.terminal',?2)",params![v.spec.project_id,json!({"job_id":v.spec.job_id,"status":status,"reason":reason,"artifacts":artifacts}).to_string()]))?;
        tx.commit().map_err(|e|e.to_string())?;Ok(())
    }
}

fn verify_finalized_pipeline_bundle(root:&Path,project:&str,candidate:&str,gate:&Value)->Result<()>{
    let workflow=root.join("workflows").join(project);
    let bundle_dir=workflow.join("bundle");
    crate::bundle::verify_bundle(&bundle_dir,false)?;
    let lineage_path=workflow.join("lineage.json");
    if lineage_path.is_file() {
        let lineage:Value=serde_json::from_slice(&fs::read(&lineage_path).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        if lineage["format"]=="mlrescurix.pipeline-lineage.v1" {
            if lineage["candidate_id"]!=candidate||lineage["frozen_model_sha256"]!=gate["frozen_model_sha256"]||lineage["test_row_intersection"]!=0 {
                return Err("finalized pipeline lineage binding mismatch".into());
            }
            let manifest:Value=serde_json::from_slice(&fs::read(workflow.join("dataset_manifest.json")).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            let row_set=|v:&Value|->Result<std::collections::HashSet<u64>>{
                let values=v.as_array().ok_or("row ids must be an array")?;let mut out=std::collections::HashSet::new();
                for x in values {if !out.insert(x.as_u64().ok_or("invalid row id")?){return Err("duplicate row id".into());}}Ok(out)
            };
            let train=row_set(&manifest["splits"]["train"])?;
            let val=row_set(&manifest["splits"]["validation"])?;
            let test=row_set(&manifest["splits"]["test"])?;
            let trainval:std::collections::HashSet<u64>=train.union(&val).copied().collect();
            let refit_rows=row_set(&lineage["final_refit_trainval_row_ids"])?;
            if refit_rows!=trainval||!refit_rows.is_disjoint(&test) {
                return Err("finalized pipeline trainval rows do not match audited train+validation or overlap test".into());
            }
            for fold in lineage["final_refit_folds"].as_array().ok_or("missing final_refit_folds")? {
                let fit=row_set(&fold["train_row_ids"])?;
                let held=row_set(&fold["held_row_ids"])?;
                if fit.is_empty()||held.is_empty()||!fit.is_subset(&trainval)||!held.is_subset(&trainval)||!fit.is_disjoint(&held)||!fit.is_disjoint(&test)||!held.is_disjoint(&test) {
                    return Err("finalized pipeline OOF fold provenance invalid or overlaps sealed test".into());
                }
                if manifest["split_policy"]=="chronological"&&fit.iter().max()>=held.iter().min() {
                    return Err("finalized pipeline chronological OOF provenance invalid".into());
                }
                if manifest["split_policy"]=="grouped" {
                    let groups=manifest["group_assignments"].as_array().ok_or("audited group assignments missing")?;
                    let group_set=|rows:&std::collections::HashSet<u64>|->Result<std::collections::HashSet<String>>{
                        rows.iter().map(|row|groups.get(*row as usize).and_then(Value::as_str).map(str::to_string).ok_or("row lacks an audited group".into())).collect()
                    };
                    if !group_set(&fit)?.is_disjoint(&group_set(&held)?) {
                        return Err("finalized pipeline group OOF provenance overlaps".into());
                    }
                }
            }
        }
    }
    Ok(())
}

/// Recompute all managed byte bindings, including every OOF fit, before commit.
fn verify_pipeline_receipt(root:&Path,s:&JobSpec)->Result<Value>{
    let Probe::Scientific{dataset_id,spec:Some(protocol),..}=&s.probe else{return Err("pipeline lacks protocol".into());};
    let id=protocol["experiment_id"].as_str().ok_or("missing pipeline id")?;
    let workflow=root.join("workflows").join(&s.project_id);let directory=workflow.join("trials").join(id);
    let receipt:Value=serde_json::from_slice(&fs::read(directory.join("trial.json")).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let c=connect(root)?;
    let dataset:String=sql(c.query_row("SELECT sha256 FROM scientific_datasets WHERE project=?1 AND id=?2",params![s.project_id,dataset_id],|r|r.get(0)))?;
    let environment:String=sql(c.query_row("SELECT identity FROM environments WHERE id=?1",[s.environment_id.as_deref().ok_or("missing environment")?],|r|r.get(0)))?;
    let source_hash=crate::assets::embedded_python_src_hash();
    let manifest_bytes=fs::read(workflow.join("dataset_manifest.json")).map_err(|e|e.to_string())?;
    let split_hash=hash(&manifest_bytes);
    let audited:bool=sql(c.query_row("SELECT EXISTS(SELECT 1 FROM scientific_jobs s JOIN jobs j ON s.project=j.project AND s.job=j.id WHERE s.project=?1 AND s.dataset=?2 AND s.operation='audit' AND s.result_sha256=?3 AND j.status='succeeded')",params![s.project_id,dataset_id,split_hash],|r|r.get(0)))?;
    if !audited||receipt["experiment_id"]!=id||receipt["protocol"]!=*protocol||receipt["dataset_sha256"]!=dataset||receipt["split_sha256"]!=split_hash||receipt["environment_identity"]!=environment||receipt["source_sha256"]!=source_hash{return Err("pipeline receipt source/environment/protocol/dataset binding failed".into());}
    for (file,key) in [("candidate.joblib","model_sha256"),("predictions.json","predictions_sha256"),("oof.json","oof_sha256")] {
        if hash(&fs::read(directory.join(file)).map_err(|e|e.to_string())?)!=receipt[key].as_str().unwrap_or(""){return Err(format!("pipeline managed bytes changed: {file}"));}
    }
    let manifest:Value=serde_json::from_slice(&manifest_bytes).map_err(|e|e.to_string())?;
    let row_set=|v:&Value|->Result<std::collections::HashSet<u64>>{
        let values=v.as_array().ok_or("OOF row ids must be an array")?;let mut out=std::collections::HashSet::new();
        for x in values {if !out.insert(x.as_u64().ok_or("invalid OOF row id")?){return Err("duplicate OOF row id".into());}}Ok(out)
    };
    let train=row_set(&manifest["splits"]["train"])?;
    let oof:Value=serde_json::from_slice(&fs::read(directory.join("oof.json")).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    if oof["dataset_sha256"]!=dataset||oof["split_sha256"]!=split_hash{return Err("OOF dataset/split binding failed".into());}
    let covered=row_set(&oof["covered_row_ids"])?;let excluded=row_set(&oof["excluded_warmup_row_ids"])?;
    if covered.is_empty()||!covered.is_disjoint(&excluded)||covered.union(&excluded).copied().collect::<std::collections::HashSet<_>>()!=train{return Err("OOF coverage does not match training partition".into());}
    if manifest["split_policy"]!="chronological"&&!excluded.is_empty(){return Err("non-time OOF must cover every training row".into());}
    let parents=protocol["parents"].as_array().ok_or("missing OOF parent protocols")?.len();
    let mut seen:Vec<std::collections::HashSet<u64>>=(0..parents).map(|_|std::collections::HashSet::new()).collect();
    for fold in oof["folds"].as_array().ok_or("missing OOF fold evidence")? {
        let index=fold["parent"].as_u64().ok_or("missing fold parent")? as usize;
        if index>=parents{return Err("OOF parent outside managed protocol".into());}
        let fit=row_set(&fold["train_row_ids"])?;let held=row_set(&fold["held_row_ids"])?;
        if fit.is_empty()||held.is_empty()||!fit.is_subset(&train)||!held.is_subset(&train)||!fit.is_disjoint(&held)||!seen[index].is_disjoint(&held){return Err("OOF fold train/held provenance invalid".into());}
        if manifest["split_policy"]=="chronological"&&fit.iter().max()>=held.iter().min(){return Err("OOF chronological provenance invalid".into());}
        if manifest["split_policy"]=="grouped" {
            let groups=manifest["group_assignments"].as_array().ok_or("audited group assignments missing")?;
            let group_set=|rows:&std::collections::HashSet<u64>|->Result<std::collections::HashSet<String>>{rows.iter().map(|row|groups.get(*row as usize).and_then(Value::as_str).map(str::to_string).ok_or("OOF row lacks an audited group".into())).collect()};
            if !group_set(&fit)?.is_disjoint(&group_set(&held)?){return Err("OOF group provenance overlaps".into());}
        }
        seen[index].extend(held);
        let file=fold["model_file"].as_str().ok_or("missing fold model")?;valid_id(file)?;
        if hash(&fs::read(directory.join(file)).map_err(|e|e.to_string())?)!=fold["model_sha256"].as_str().unwrap_or(""){return Err("OOF fit model hash failed".into());}
    }
    if seen.iter().any(|rows|*rows!=covered){return Err("OOF parents do not have identical held-row coverage".into());}
    for (file,expected) in oof["artifacts"].as_object().ok_or("missing managed parent artifact hashes")? {
        valid_id(file)?;if hash(&fs::read(directory.join(file)).map_err(|e|e.to_string())?)!=expected.as_str().unwrap_or(""){return Err("parent artifact bytes failed verification".into());}
    }
    Ok(receipt)
}

pub fn finalization_binding(root:&Path,project:&str,candidate:&str)->Result<Value>{
    valid_id(project)?;valid_id(candidate)?;let c=connect(root)?;
    let job:String=sql(c.query_row("SELECT job FROM pipeline_receipts WHERE project=?1 AND candidate=?2",params![project,candidate],|r|r.get(0))).map_err(|_|"candidate lacks a verified full-pipeline receipt; metadata approval cannot authorize finalization".to_string())?;
    let view=load_job(&c,project,&job)?;
    if view.status!="succeeded"{return Err("candidate pipeline job is not successful".into());}
    let receipt=verify_pipeline_receipt(root,&view.spec)?;
    let stored:String=sql(c.query_row("SELECT receipt_json FROM pipeline_receipts WHERE project=?1 AND candidate=?2",params![project,candidate],|r|r.get(0)))?;
    let recorded:Value=serde_json::from_str(&stored).map_err(|e|e.to_string())?;
    if receipt!=recorded{return Err("pipeline receipt differs from authoritative committed receipt".into());}
    let (revision,intent,spec):(i64,i64,i64)=sql(c.query_row("SELECT revision,intent_revision,spec_revision FROM projects WHERE id=?1",[project],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))))?;
    let binding=json!({"format":"mlrescurix.finalization-binding.v1","project":project,"candidate":candidate,"job":job,"producer_task":view.spec.task_id,"intent_revision":intent,"spec_revision":spec,"pipeline_receipt":receipt});
    Ok(json!({"binding_sha256":hash(binding.to_string().as_bytes()),"binding":binding,"observed_project_revision":revision,"status":"reviewable_development_candidate","remaining":"refit, fresh target feasibility, sealed test and delivery required"}))
}

/// Only the local human CLI and the independent critic role call this API.
pub fn record_content_decision(root:&Path,project:&str,candidate:&str,expected:&str,kind:&str,actor:&str,approved:bool,rationale:&str)->Result<Value>{
    if !matches!(kind,"user_approval"|"critic_audit")||actor.is_empty()||rationale.is_empty(){return Err("invalid content decision".into());}
    let review=finalization_binding(root,project,candidate)?;
    if review["binding_sha256"]!=expected{return Err("approval/audit content binding changed; review current managed bytes".into());}
    if kind=="critic_audit"&&review["binding"]["producer_task"]==actor{return Err("candidate producer cannot independently audit its own pipeline".into());}
    let c=connect(root)?;let tx=sql(rusqlite::Transaction::new_unchecked(&c,rusqlite::TransactionBehavior::Immediate))?;
    enforce_state_write_admission(root,Some(&tx),review.to_string().len() as u64*4+32*1024,StorageWriteKind::Checkpoint,Some(project))?;
    let now=now_ms();
    sql(tx.execute("INSERT INTO content_decisions(project,candidate,binding_sha256,kind,actor,approved,rationale,binding_json,created_ms) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9)",params![project,candidate,expected,kind,actor,approved as i64,rationale,review["binding"].to_string(),now]))?;
    if kind=="user_approval" {
        sql(tx.execute("INSERT INTO approvals(project,candidate_id,approved,user_identity,rationale,created_ms) VALUES(?1,?2,?3,?4,?5,?6)",params![project,candidate,approved as i64,actor,rationale,now]))?;
    } else if kind=="critic_audit" {
        sql(tx.execute("INSERT INTO critic_audits(project,candidate_id,auditor_role,status,rationale,source_hash,protocol,constraints_verified,created_ms) VALUES(?1,?2,'critic',?3,?4,?5,'mlrescurix.pipeline.v1',?6,?7)",params![project,candidate,if approved{"approved"}else{"rejected"},rationale,expected,approved as i64,now]))?;
    }
    sql(tx.execute("INSERT INTO events(project,kind,payload) VALUES(?1,'candidate.content_decision',?2)",params![project,json!({"candidate":candidate,"binding_sha256":expected,"kind":kind,"actor":actor,"approved":approved}).to_string()]))?;
    tx.commit().map_err(|e|e.to_string())?;Ok(review)
}

pub fn require_finalization_decisions(root:&Path,project:&str,candidate:&str)->Result<Value>{
    let review=finalization_binding(root,project,candidate)?;let c=connect(root)?;
    for kind in ["user_approval","critic_audit"] {
        let decision:Option<(bool,String)>=sql(c.query_row("SELECT approved,binding_sha256 FROM content_decisions WHERE project=?1 AND candidate=?2 AND kind=?3 ORDER BY created_ms DESC LIMIT 1",params![project,candidate,kind],|r|Ok((r.get(0)?,r.get(1)?))).optional())?;
        if !decision.is_some_and(|(approved,binding)|approved&&review["binding_sha256"]==binding){return Err(format!("finalization requires current content-bound {kind}"));}
    }
    Ok(review)
}

fn validate_finalization_job(root:&Path,s:&JobSpec)->Result<()> {
    if let Probe::Scientific{operation,dataset_id,candidate_id,..}=&s.probe {
        if operation=="finalize" {
            let candidate=candidate_id.as_deref().ok_or("missing finalization candidate")?;
            let review=require_finalization_decisions(root,&s.project_id,candidate)?;
            let c=connect(root)?;
            let producer=load_job(&c,&s.project_id,review["binding"]["job"].as_str().ok_or("missing producer job")?)?;
            if producer.spec.environment_id!=s.environment_id {
                return Err("finalization environment differs from approved producer environment".into());
            }
            if !matches!(&producer.spec.probe,Probe::Scientific{dataset_id:producer_dataset,..} if producer_dataset==dataset_id) {
                return Err("finalization dataset differs from approved producer dataset".into());
            }
        }
    }
    Ok(())
}

fn reconcile_final_tests(root:&Path,c:&Connection)->Result<()> {
    let mut q=sql(c.prepare("SELECT project,access_id,dataset_sha256,candidate_id FROM final_test_gates WHERE status='reserved'"))?;
    let rows=sql(q.query_map([],|r|Ok((r.get::<_,String>(0)?,r.get::<_,String>(1)?,r.get::<_,String>(2)?,r.get::<_,String>(3)?))))?
        .collect::<rusqlite::Result<Vec<_>>>().map_err(|e|e.to_string())?;drop(q);
    for (project,access,dataset,candidate) in rows {
        let workflow=root.join("workflows").join(&project);
        let gate_path=workflow.join("test_access.json");let result_path=workflow.join("final.json");
        if !gate_path.is_file()||gate_path.symlink_metadata().map_err(|e|e.to_string())?.file_type().is_symlink(){continue;}
        let gate:Value=serde_json::from_slice(&fs::read(&gate_path).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        let result:Option<Value>=if result_path.is_file(){Some(serde_json::from_slice(&fs::read(&result_path).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?)}else{None};
        if gate["status"]=="consumed"&&gate["test_access_id"]==access&&gate["dataset_sha256"]==dataset&&gate["candidate_id"]==candidate {
            let frozen=gate["frozen_model_sha256"].as_str().ok_or("final gate has no frozen artifact hash")?;
            let hash_result=if result.as_ref().is_some_and(|r|r["test_access_id"]==access){Some(hash(&fs::read(&result_path).map_err(|e|e.to_string())?))}else{None};
            let tx=sql(rusqlite::Transaction::new_unchecked(&c,rusqlite::TransactionBehavior::Immediate))?;
            sql(tx.execute("UPDATE final_test_gates SET status='consumed',frozen_model_sha256=?4,result_sha256=?5 WHERE project=?1 AND access_id=?2 AND dataset_sha256=?3 AND status='reserved'",params![project,access,dataset,frozen,hash_result]))?;
            sql(tx.execute("INSERT INTO events(project,kind,payload) VALUES(?1,'final_test.reconciled',?2)",params![project,json!({"test_access_id":access,"candidate_id":candidate,"result_sha256":hash_result}).to_string()]))?;
            tx.commit().map_err(|e|e.to_string())?;
        }
    }
    Ok(())
}

fn admission(c:&Connection,s:&JobSpec,worker_ceiling:usize)->Result<()>{
    let root = root_from_conn(c)?;
    crate::broker::check_capacity_admission(&root, c, &s.project_id, s, worker_ceiling)?;
    let _=reservation(c,&s.project_id,s)?;
    Ok(())
}

fn reservation(c:&Connection,p:&str,s:&JobSpec)->Result<(BTreeMap<String,i64>,BTreeMap<String,i64>)>{
    let (device,project):(String,String)=sql(c.query_row("SELECT d.policy,p.policy FROM device d JOIN projects p ON p.id=?1 WHERE d.id=1",[p],|r|Ok((r.get(0)?,r.get(1)?))))?;
    let d:Policy=serde_json::from_str(&device).map_err(|e|e.to_string())?;let pr:Policy=serde_json::from_str(&project).map_err(|e|e.to_string())?;
    if d.budget.keys().ne(pr.budget.keys()){return Err("M1 requires matching device/project budget dimensions".into());}
    if !d.budget.contains_key("wall_ms"){return Err("M1 execution policy must declare wall_ms budget".into());}
    let budget:BTreeMap<String,i64>=d.budget.keys().map(|k|(k.clone(),if k=="wall_ms"{s.timeout_ms as i64}else{0})).collect();
    let mut capacity:BTreeMap<String,i64>=d.capacity.keys().map(|k|(k.clone(),match k.as_str(){"workers"=>1,"cpu_threads"=>s.threads as i64,"memory_bytes"=>s.resources.memory_bytes as i64,_=>-1})).collect();
    if capacity.values().any(|v|*v<0)||d.capacity.keys().ne(pr.capacity.keys()){return Err("M1 capacity dimensions must be workers and optional cpu_threads/memory_bytes".into());}
    capacity.insert("memory_bytes".to_string(), s.resources.memory_bytes as i64);
    for(scope,policy) in [(None,&d),(Some(p),&pr)] {
        let mut held:BTreeMap<String,i64>=policy.budget.keys().map(|k|(k.clone(),0)).collect();let mut spent=held.clone();
        let mut reservations=sql(c.prepare("SELECT budget FROM reservations WHERE held=1 AND (?1 IS NULL OR project=?1)"))?;
        for row in sql(reservations.query_map([scope],|r|r.get::<_,String>(0)))?{let values:BTreeMap<String,i64>=serde_json::from_str(&sql(row)?).map_err(|e|e.to_string())?;for(k,v)in values{if let Some(x)=held.get_mut(&k){*x=x.checked_add(v).ok_or("budget overflow")?;}}}
        let mut charges=sql(c.prepare("SELECT amount FROM charges WHERE (?1 IS NULL OR project=?1)"))?;
        for row in sql(charges.query_map([scope],|r|r.get::<_,String>(0)))?{let values:BTreeMap<String,i64>=serde_json::from_str(&sql(row)?).map_err(|e|e.to_string())?;for(k,v)in values{if let Some(x)=spent.get_mut(&k){*x=x.checked_add(v).ok_or("budget overflow")?;}}}
        for(k,limit)in &policy.budget {let used=held[k].checked_add(spent[k]).and_then(|x|x.checked_add(budget[k])).ok_or("budget overflow")?;if used>*limit{return Err(format!("budget exhausted: {k}"));}}
    }
    Ok((budget,capacity))
}

fn settle_job_action(c:&Connection,p:&str,job:&str,elapsed_ms:u64,job_status:&str)->Result<()>{
    let action_id=format!("job.{job}");let row:Option<String>=sql(c.query_row("SELECT data FROM actions WHERE project=?1 AND id=?2",params![p,action_id],|r|r.get(0)).optional())?;let Some(row)=row else{return Ok(())};
    let mut a:Action=serde_json::from_str(&row).map_err(|e|e.to_string())?;
    if sql(c.query_row("SELECT EXISTS(SELECT 1 FROM charges WHERE project=?1 AND action=?2)",params![p,action_id],|r|r.get::<_,bool>(0)))?{return Ok(());}
    let actual:BTreeMap<String,i64>=a.budget.keys().map(|k|(k.clone(),if k=="wall_ms"{elapsed_ms.min(i64::MAX as u64) as i64}else{0})).collect();
    sql(c.execute("INSERT INTO charges(project,action,amount,proof) VALUES(?1,?2,?3,?4)",params![p,action_id,serde_json::to_string(&actual).map_err(|e|e.to_string())?,format!("supervisor observed bounded local job outcome {job_status}")]))?;
    sql(c.execute("UPDATE reservations SET held=0 WHERE project=?1 AND action=?2",params![p,action_id]))?;
    a.status=if job_status=="succeeded"{"succeeded".into()}else{"failed".into()};a.receipt=Some(json!({"action_id":action_id,"status":a.status,"charged":actual,"job_status":job_status}));
    sql(c.execute("UPDATE actions SET data=?3 WHERE project=?1 AND id=?2",params![p,action_id,serde_json::to_string(&a).map_err(|e|e.to_string())?]))?;
    sql(c.execute("INSERT INTO events(project,kind,payload) VALUES(?1,'action.committed',?2)",params![p,serde_json::to_string(&a.receipt).map_err(|e|e.to_string())?]))?;Ok(())
}

fn terminate(child:&mut Child)->Result<()>{
    let pid=child.id() as i32;
    unsafe{libc::kill(-pid,libc::SIGTERM);}
    let until=Instant::now()+Duration::from_millis(300);
    while Instant::now()<until {if child.try_wait().map_err(|e|e.to_string())?.is_some(){return Ok(());}thread::sleep(Duration::from_millis(10));}
    unsafe{libc::kill(-pid,libc::SIGKILL);}
    let _=child.wait();Ok(())
}
fn safe_remove(root:&Path,path:&Path)->Result<()>{
    if !path.starts_with(root){return Err("managed path escapes state root".into());}
    let canon_root=fs::canonicalize(root).map_err(|e|e.to_string())?;let parent=fs::canonicalize(path.parent().ok_or("managed path has no parent")?).map_err(|e|e.to_string())?;if !parent.starts_with(&canon_root){return Err("managed path parent escapes state root".into());}
    let meta=fs::symlink_metadata(path).map_err(|e|e.to_string())?;if meta.file_type().is_symlink(){return Err("managed path symlink rejected".into());}
    if meta.is_dir(){fs::remove_dir_all(path).map_err(|e|e.to_string())}else{fs::remove_file(path).map_err(|e|e.to_string())}
}
fn manifest_dir(dir:&Path,max_bytes:u64)->Result<Vec<Value>>{let mut total=0u64;let mut artifacts=vec![];for entry in fs::read_dir(dir).map_err(|e|e.to_string())?{let e=entry.map_err(|e|e.to_string())?;let meta=fs::symlink_metadata(e.path()).map_err(|e|e.to_string())?;if meta.file_type().is_symlink()||!meta.is_file(){return Err("invalid recovered artifact".into());}let name=e.file_name().to_string_lossy().into_owned();if name=="job.json"{continue;}let data=fs::read(e.path()).map_err(|e|e.to_string())?;total=total.saturating_add(data.len() as u64);if total>max_bytes{return Err("recovered artifacts exceed output bound".into());}artifacts.push(json!({"name":name,"bytes":data.len(),"sha256":hash(&data)}));}if artifacts.is_empty(){return Err("no output artifact".into());}Ok(artifacts)}
pub fn ensure_disk_space(root:&Path,output:u64)->Result<()>{
    let config = StorageBudgetConfig::load(root);
    ensure_disk_space_with_config(root, output, &config)
}
pub fn ensure_disk_space_with_config(root:&Path,output:u64,config:&StorageBudgetConfig)->Result<()>{
    if let Ok(sim) = std::env::var("MLRESCURIX_SIMULATED_FREE_DISK") {
        if let Ok(free) = sim.parse::<u64>() {
            if !disk_space_allows(free, output, config.disk_headroom_bytes) {
                return Err("insufficient free disk for bounded job outputs and storage headroom".into());
            }
            return Ok(());
        }
    }
    let mut check_path = root.to_path_buf();
    while !check_path.exists() {
        if let Some(parent) = check_path.parent() {
            check_path = parent.to_path_buf();
        } else {
            break;
        }
    }
    let p=std::ffi::CString::new(check_path.as_os_str().as_encoded_bytes()).map_err(|e|e.to_string())?;
    let mut stats=unsafe{std::mem::zeroed::<libc::statvfs>()};
    if unsafe{libc::statvfs(p.as_ptr(),&mut stats)}!=0{return Err("cannot inspect available state storage".into());}
    let free=(stats.f_bavail as u64).saturating_mul(stats.f_frsize as u64);
    if !disk_space_allows(free,output,config.disk_headroom_bytes){return Err("insufficient free disk for bounded job outputs and storage headroom".into());}Ok(())
}
fn disk_space_allows(free:u64,output:u64,headroom:u64)->bool{free>=output.saturating_add(headroom)}

#[cfg(test)] mod tests {use super::*;#[test]fn low_disk_admission_boundary(){assert!(disk_space_allows(100,40,60));assert!(!disk_space_allows(99,40,60));assert!(disk_space_allows(u64::MAX,10,u64::MAX));}}

/// Entrypoint for the private probe process. The control plane never accepts an executable path.
pub fn worker(spec_path:&Path,output:&Path)->Result<i32>{
    let bytes=fs::read(spec_path).map_err(|e|e.to_string())?;let s:JobSpec=serde_json::from_slice(&bytes).map_err(|e|e.to_string())?;validate_spec(&s)?;
    let mut control=std::io::stdin();let mut gate=[0u8;1];if control.read_exact(&mut gate).is_err()||gate[0]!=1{return Ok(125);}
    let worker_spec=s.clone();
    let alive=Arc::new(AtomicBool::new(true));let watcher=alive.clone();thread::spawn(move||{let mut byte=[0u8;1];loop{match control.read(&mut byte){Ok(0)|Err(_)=>{watcher.store(false,Ordering::Release);break},Ok(_)=>{}}}});
    match s.probe {
        Probe::Sleep{millis}=>{let deadline=Instant::now()+Duration::from_millis(millis.min(s.timeout_ms));while Instant::now()<deadline{if !alive.load(Ordering::Acquire){return Ok(125)}thread::sleep(Duration::from_millis(10));}if millis>s.timeout_ms{return Ok(124)}},
        Probe::Fail{code}=>return Ok(code as i32),
        Probe::Write{name,content}=>{if !alive.load(Ordering::Acquire){return Ok(125)}let p=output.join(name);let mut f=fs::OpenOptions::new().write(true).create_new(true).open(p).map_err(|e|e.to_string())?;f.write_all(content.as_bytes()).map_err(|e|e.to_string())?;f.sync_all().map_err(|e|e.to_string())?;},
        Probe::Scientific{operation,dataset_id:_,spec,candidate_id,test_access_id}=>{
            if !alive.load(Ordering::Acquire){return Ok(125)}
            if let Ok(image)=std::env::var("MLRESCURIX_SCIENTIFIC_IMAGE") {
                return container_scientific_worker(&worker_spec,&operation,spec.as_ref(),candidate_id.as_deref(),test_access_id.as_deref(),output,&image,s.timeout_ms,s.threads,alive);
            }
            return Err("scientific host execution prohibited; verified Docker environment required".into());
        }
    }
    Ok(0)
}

fn container_scientific_worker(job_spec:&JobSpec,operation:&str,spec:Option<&Value>,candidate:Option<&str>,access:Option<&str>,output:&Path,image:&str,timeout:u64,threads:u32,alive:Arc<AtomicBool>)->Result<i32>{
    let root=PathBuf::from(std::env::var("MLRRESCURIX_STATE").map_err(|e|e.to_string())?);
    let job=&job_spec.job_id;let project=&job_spec.project_id;
    let resource_dir=root.join("resource-receipts").join(hash(format!("{project}/{job}").as_bytes()));
    enforce_state_write_admission(&root,None,64*1024,StorageWriteKind::EmergencyRecovery,Some(project))?;
    ensure_dir(&root,&resource_dir)?;
    let workflow=PathBuf::from(std::env::var("MLRRESCURIX_WORKFLOW").map_err(|e|e.to_string())?);
    let dataset=std::env::var("MLRRESCURIX_DATASET").map_err(|e|e.to_string())?;
    let source=std::env::var("MLRRESCURIX_PYTHONPATH").map_err(|e|e.to_string())?;
    let source_hash=crate::agents::workspace::BranchWorkspaceManager::compute_dir_hash(Path::new(&source))?;
    let req=output.join("project-spec.json");
    if let Some(value)=spec {fs::write(&req,serde_json::to_vec(value).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;}
    let name=format!("mlr-science-{}",&hash(format!("{}/{project}/{job}/{}",root.display(),job_spec.generation).as_bytes())[..24]);
    let mut cmd=Command::new("docker");
    cmd.args(["create","--pull=never","--init","--name",&name,"--network","none","--read-only","--user"])
        .arg(format!("{}:{}",unsafe{libc::getuid()},unsafe{libc::getgid()}))
        .args(["--cap-drop","ALL","--security-opt","no-new-privileges:true","--memory"]).arg(job_spec.resources.memory_bytes.to_string()).arg("--memory-swap").arg(job_spec.resources.memory_bytes.to_string()).args(["--cpus"])
        .arg(threads.to_string()).args(["--pids-limit","64","--tmpfs","/tmp:rw,noexec,nosuid,size=64m","-v"])
        .arg(format!("{}:/workflow:rw",workflow.display())).arg("-v").arg(format!("{dataset}:/data.csv:ro"))
        .args(["--env","PYTHONPATH=/source","--env","PYTHONDONTWRITEBYTECODE=1"])
        .args(["--env", &format!("MLRESCURIX_NUM_THREADS={threads}")])
        .args(["--env", &format!("OMP_NUM_THREADS={threads}")])
        .args(["--env", &format!("OPENBLAS_NUM_THREADS={threads}")])
        .args(["--env", &format!("MKL_NUM_THREADS={threads}")])
        .args(["--env", &format!("VECLIB_MAXIMUM_THREADS={threads}")])
        .args(["--env", &format!("NUMEXPR_NUM_THREADS={threads}")])
        .args(["--env", &format!("POLARS_MAX_THREADS={threads}")])
        .arg("--env").arg(format!("MLRESCURIX_SOURCE_SHA256={source_hash}"))
        .arg("--env").arg(format!("MLRESCURIX_ENVIRONMENT_IDENTITY={}",std::env::var("MLRESCURIX_ENVIRONMENT_IDENTITY").map_err(|e|e.to_string())?))
        .arg("-v").arg(format!("{}:/resource:rw",resource_dir.display()));
    for (key,value) in resources::labels(&root,job_spec)? {cmd.arg("--label").arg(format!("{key}={value}"));}
    cmd.args([image,"python","-c",resources::monitor()]).arg(job_spec.resources.memory_bytes.to_string()).arg(threads.to_string())
        .arg(job_spec.resources.require_memory_certification.to_string())
        .args(["python","-m","mlrescurix_worker.scientific.cli",operation,"--output","/workflow"]);
    match operation {
        "audit"=>{cmd.args(["--csv","/data.csv","--spec","/output/project-spec.json"]);},
        "trial"|"pipeline"=>{cmd.args(["--spec","/output/project-spec.json"]);},
        "finalize"=>{cmd.args(["--candidate",candidate.ok_or("missing candidate")?,"--test-access-id",access.ok_or("missing test access")?]);},
        "baselines"=>{},_=>return Err("unsupported container science operation".into()),
    }
    let id=resources::create(&root,job_spec,&mut cmd).map_err(|e|format!("container create/identity: {e}"))?;
    let mut child=Command::new("docker").args(["start","-a",&id]).stdin(Stdio::null()).stdout(Stdio::inherit()).stderr(Stdio::inherit()).spawn().map_err(|e|e.to_string())?;
    let deadline=Instant::now()+Duration::from_millis(timeout);
    let code=loop{
        if !alive.load(Ordering::Acquire)||Instant::now()>=deadline {
            resources::confirm_job_exit(&root,job_spec).map_err(|e|format!("container exit reconciliation: {e}"))?;let _=child.kill();let _=child.wait();break 124;
        }
        if let Some(status)=child.try_wait().map_err(|e|e.to_string())?{break status.code().unwrap_or(1);}
        thread::sleep(Duration::from_millis(50));
    };
    resources::save_measurement(&root,job_spec,&resource_dir.join("measurement.json")).map_err(|e|format!("container measurement: {e}"))?;
    resources::confirm_job_exit(&root,job_spec).map_err(|e|format!("container exit reconciliation: {e}"))?;
    if resource_dir.exists(){safe_remove(&root,&resource_dir)?;}
    if req.exists(){fs::remove_file(req).map_err(|e|e.to_string())?;}
    if code!=0{return Ok(code);}
    if crate::agents::workspace::BranchWorkspaceManager::compute_dir_hash(Path::new(&source))?!=source_hash{return Err("scientific source changed during execution".into());}
    let (evidence,name)=match operation {
        "trial"|"pipeline"=>{let id=spec.and_then(|v|v["experiment_id"].as_str()).ok_or("missing experiment")?;(workflow.join("trials").join(id).join("trial.json"),"trial.json")},
        "audit"=>(workflow.join("dataset_manifest.json"),"dataset_manifest.json"),
        "baselines"=>(workflow.join("results.json"),"results.json"),
        "finalize"=>(workflow.join("final.json"),"final.json"),_=>unreachable!(),
    };
    if !evidence.is_file(){return Err("scientific worker produced no committed result".into());}
    if operation=="finalize" {
        let final_result:Value=serde_json::from_slice(&fs::read(&evidence).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        if final_result["status"]!="verified_development_only" {return Ok(3);}
    }
    fs::copy(evidence,output.join(name)).map_err(|e|e.to_string())?;Ok(0)
}

pub fn register_feature_transform(root: &Path, project: &str, transform_id: &str, transform_spec: &Value, parent_dataset: &str) -> Result<Value> {
    valid_id(project)?; valid_id(transform_id)?;
    let conn = connect(root)?;
    let exists: bool = sql(conn.query_row("SELECT EXISTS(SELECT 1 FROM scientific_datasets WHERE project=?1 AND id=?2)", params![project, parent_dataset], |r| r.get(0)))?;
    if !exists {
        return Err(format!("parent dataset '{parent_dataset}' does not exist in scientific_datasets"));
    }
    let spec_str = serde_json::to_string(transform_spec).map_err(|e| e.to_string())?;
    if spec_str.len()>MAX_FRAME{return Err("transform metadata exceeds frame bound".into());}
    enforce_state_write_admission(root,Some(&conn),spec_str.len() as u64*4+32*1024,StorageWriteKind::Checkpoint,Some(project))?;
    let now = now_ms();
    let tx = sql(rusqlite::Transaction::new_unchecked(&conn,rusqlite::TransactionBehavior::Immediate))?;
    enforce_state_write_admission(root,Some(&tx),32*1024,StorageWriteKind::Checkpoint,Some(project))?;
    sql(tx.execute(
        "INSERT INTO feature_transforms(project, transform_id, transform_spec, parent_dataset, created_ms) VALUES(?1, ?2, ?3, ?4, ?5)",
        params![project, transform_id, spec_str, parent_dataset, now],
    ))?;
    sql(tx.execute(
        "INSERT INTO events(project, kind, payload) VALUES(?1, 'feature_transform.registered', ?2)",
        params![project, json!({"transform_id": transform_id, "parent_dataset": parent_dataset}).to_string()],
    ))?;
    tx.commit().map_err(|e| e.to_string())?;
    Ok(json!({
        "status": "registered",
        "project": project,
        "transform_id": transform_id,
        "parent_dataset": parent_dataset,
        "created_ms": now,
    }))
}

pub fn register_calibration_record(root: &Path, project: &str, calibration_id: &str, base_candidate_id: &str, method: &str, brier_score: Option<f64>, log_loss: Option<f64>) -> Result<Value> {
    valid_id(project)?; valid_id(calibration_id)?;
    if !matches!(method, "isotonic" | "sigmoid") {
        return Err(format!("unsupported calibration method: '{method}'; must be 'isotonic' or 'sigmoid'"));
    }
    let conn = connect(root)?;
    let exists: bool = sql(conn.query_row("SELECT EXISTS(SELECT 1 FROM trial_results WHERE project=?1 AND experiment_id=?2)", params![project, base_candidate_id], |r| r.get(0)))?;
    if !exists {
        let dag_exists: bool = sql(conn.query_row("SELECT EXISTS(SELECT 1 FROM experiment_dag WHERE project=?1 AND id=?2)", params![project, base_candidate_id], |r| r.get(0)))?;
        if !dag_exists {
            return Err(format!("base candidate '{base_candidate_id}' not found in experiment_dag or trial_results"));
        }
    }
    let now = now_ms();
    let tx = sql(rusqlite::Transaction::new_unchecked(&conn,rusqlite::TransactionBehavior::Immediate))?;
    enforce_state_write_admission(root,Some(&tx),32*1024,StorageWriteKind::Checkpoint,Some(project))?;
    sql(tx.execute(
        "INSERT INTO calibration_records(project, calibration_id, base_candidate_id, method, brier_score, log_loss, created_ms) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![project, calibration_id, base_candidate_id, method, brier_score, log_loss, now],
    ))?;
    sql(tx.execute(
        "INSERT INTO events(project, kind, payload) VALUES(?1, 'calibration.registered', ?2)",
        params![project, json!({"calibration_id": calibration_id, "base_candidate_id": base_candidate_id, "method": method}).to_string()],
    ))?;
    tx.commit().map_err(|e| e.to_string())?;
    Ok(json!({
        "status": "registered",
        "project": project,
        "calibration_id": calibration_id,
        "base_candidate_id": base_candidate_id,
        "method": method,
        "brier_score": brier_score,
        "log_loss": log_loss,
        "created_ms": now,
    }))
}

pub fn register_scientific_ensemble(root: &Path, project: &str, ensemble_id: &str, recipe_type: &str, parent_ids: &[String], weights: Option<&[f64]>, meta_params: Option<&Value>, is_feasible: bool) -> Result<Value> {
    valid_id(project)?; valid_id(ensemble_id)?;
    if !matches!(recipe_type, "voting" | "stacking" | "averaging") {
        return Err(format!("unsupported ensemble recipe type: '{recipe_type}'; must be 'voting', 'stacking', or 'averaging'"));
    }
    if parent_ids.is_empty() {
        return Err("ensemble requires at least one parent candidate".into());
    }
    let mut unique_check = std::collections::HashSet::new();
    for p in parent_ids {
        if !unique_check.insert(p) {
            return Err(format!("duplicate parent candidate id '{p}'"));
        }
    }
    let conn = connect(root)?;
    for p in parent_ids {
        let (status, model_sha): (String, String) = sql(conn.query_row(
            "SELECT status, model_sha256 FROM trial_results WHERE project=?1 AND experiment_id=?2",
            params![project, p],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )).map_err(|_| format!("Parent candidate '{p}' does not exist or has not been authoritatively evaluated in trial_results"))?;
        if status != "evaluated" {
            return Err(format!("Parent candidate '{p}' status is '{status}', expected 'evaluated'"));
        }
        if model_sha.is_empty() {
            return Err(format!("Parent candidate '{p}' lacks verified model artifact sha256 in trial_results"));
        }
    }
    if let Some(w) = weights {
        if w.len() != parent_ids.len() || w.iter().any(|v| !v.is_finite() || *v < 0.0) || !w.iter().sum::<f64>().is_finite() || w.iter().sum::<f64>() <= 0.0 {
            return Err("Ensemble weights must match parents and be finite/nonnegative with positive sum".into());
        }
    }
    // Registration is provenance metadata, not trusted evaluation of the ensemble.
    if is_feasible { return Err("Ensemble feasibility requires trusted full-pipeline evaluation; registration cannot certify it".into()); }
    let parents_json = serde_json::to_string(parent_ids).map_err(|e| e.to_string())?;
    if parent_ids.len()>128||meta_params.is_some_and(|v|v.to_string().len()>MAX_FRAME){return Err("ensemble metadata exceeds bound".into());}
    enforce_state_write_admission(root,Some(&conn),meta_params.map(|v|v.to_string().len() as u64*4).unwrap_or(0)+MAX_FRAME as u64,StorageWriteKind::Checkpoint,Some(project))?;
    let weights_json = weights.map(|w| serde_json::to_string(w).unwrap_or_default());
    let meta_json = meta_params.map(|m| serde_json::to_string(m).unwrap_or_default());
    let now = now_ms();
    let tx = sql(rusqlite::Transaction::new_unchecked(&conn,rusqlite::TransactionBehavior::Immediate))?;
    enforce_state_write_admission(root,Some(&tx),32*1024,StorageWriteKind::Checkpoint,Some(project))?;
    sql(tx.execute(
        "INSERT INTO scientific_ensembles(project, ensemble_id, recipe_type, parent_ids, weights, meta_params, is_feasible, created_ms) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![project, ensemble_id, recipe_type, parents_json, weights_json, meta_json, if is_feasible { 1 } else { 0 }, now],
    ))?;
    sql(tx.execute(
        "INSERT INTO events(project, kind, payload) VALUES(?1, 'ensemble.registered', ?2)",
        params![project, json!({"ensemble_id": ensemble_id, "recipe_type": recipe_type, "parent_ids": parent_ids, "is_feasible": is_feasible}).to_string()],
    ))?;
    tx.commit().map_err(|e| e.to_string())?;
    Ok(json!({
        "status": "registered",
        "project": project,
        "ensemble_id": ensemble_id,
        "recipe_type": recipe_type,
        "parent_ids": parent_ids,
        "is_feasible": is_feasible,
        "created_ms": now,
    }))
}

pub fn inspect_scientific_lineage(root: &Path, project: &str) -> Result<Value> {
    let conn = connect(root)?;
    let mut ensembles_q = sql(conn.prepare("SELECT ensemble_id, recipe_type, parent_ids, weights, meta_params, is_feasible, created_ms FROM scientific_ensembles WHERE project=?1 ORDER BY created_ms"))?;
    let ensembles = sql(ensembles_q.query_map([project], |r| {
        Ok(json!({
            "ensemble_id": r.get::<_, String>(0)?,
            "recipe_type": r.get::<_, String>(1)?,
            "parent_ids": serde_json::from_str::<Value>(&r.get::<_, String>(2)?).unwrap_or(json!([])),
            "weights": r.get::<_, Option<String>>(3)?.and_then(|s| serde_json::from_str::<Value>(&s).ok()),
            "meta_params": r.get::<_, Option<String>>(4)?.and_then(|s| serde_json::from_str::<Value>(&s).ok()),
            "is_feasible": r.get::<_, i64>(5)? == 1,
            "created_ms": r.get::<_, i64>(6)?,
        }))
    }))?.collect::<std::result::Result<Vec<_>, _>>().map_err(|e| e.to_string())?;

    let mut transforms_q = sql(conn.prepare("SELECT transform_id, transform_spec, parent_dataset, created_ms FROM feature_transforms WHERE project=?1 ORDER BY created_ms"))?;
    let transforms = sql(transforms_q.query_map([project], |r| {
        Ok(json!({
            "transform_id": r.get::<_, String>(0)?,
            "transform_spec": serde_json::from_str::<Value>(&r.get::<_, String>(1)?).unwrap_or(json!({})),
            "parent_dataset": r.get::<_, String>(2)?,
            "created_ms": r.get::<_, i64>(3)?,
        }))
    }))?.collect::<std::result::Result<Vec<_>, _>>().map_err(|e| e.to_string())?;

    let mut calib_q = sql(conn.prepare("SELECT calibration_id, base_candidate_id, method, brier_score, log_loss, created_ms FROM calibration_records WHERE project=?1 ORDER BY created_ms"))?;
    let calibrations = sql(calib_q.query_map([project], |r| {
        Ok(json!({
            "calibration_id": r.get::<_, String>(0)?,
            "base_candidate_id": r.get::<_, String>(1)?,
            "method": r.get::<_, String>(2)?,
            "brier_score": r.get::<_, Option<f64>>(3)?,
            "log_loss": r.get::<_, Option<f64>>(4)?,
            "created_ms": r.get::<_, i64>(5)?,
        }))
    }))?.collect::<std::result::Result<Vec<_>, _>>().map_err(|e| e.to_string())?;

    Ok(json!({
        "project": project,
        "ensembles": ensembles,
        "feature_transforms": transforms,
        "calibrations": calibrations,
    }))
}
