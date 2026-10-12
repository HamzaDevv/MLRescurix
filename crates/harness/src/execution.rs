//! M1 trusted local execution kernel. Jobs are typed probes, never shell commands.
use crate::{Result, core::{Policy,Action}};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, io::{Read,Write}, os::unix::{fs::PermissionsExt,process::CommandExt}, path::{Path,PathBuf}, process::{Child,ChildStdin,Command,Stdio}, sync::{Arc,atomic::{AtomicBool,Ordering}}, thread, time::{Duration,Instant,SystemTime,UNIX_EPOCH}};

const MAX_FRAME: usize = 65_536;
const POLL_MS: u64 = 100;
const LOG_CAP: u64 = 1_048_576;
const OUTPUT_CAP: u64 = 4_194_304;
const DISK_HEADROOM: u64 = 64 * 1024 * 1024;
const OWNER_MARKER: &str = "mlrescurix-owned-v1\n";
static STOP: AtomicBool = AtomicBool::new(false);
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
    pub probe: Probe,
}

#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct JobView { pub spec: JobSpec, pub status: String, pub reason: Option<String>, pub pid: Option<u32>, pub submitted_ms: i64, pub started_ms: Option<i64>, pub ended_ms: Option<i64>, pub artifacts: Vec<Value>, pub telemetry: Option<ResourceSample> }
#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct ResourceSample {pub sampled_ms:i64,pub cpu_percent:Option<f64>,pub rss_bytes:Option<u64>,pub peak_rss_bytes:u64}

fn now_ms() -> i64 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis().min(i64::MAX as u128) as i64 }
fn sql<T>(x:rusqlite::Result<T>)->Result<T>{x.map_err(|e|e.to_string())}
fn hash(data:&[u8])->String{format!("{:x}",Sha256::digest(data))}
fn valid_id(s:&str)->Result<()>{if s.is_empty()||s.len()>128||!s.bytes().all(|b|b.is_ascii_alphanumeric()||b"_.:-".contains(&b)){Err("invalid job identity".into())}else{Ok(())}}
fn connect(root:&Path)->Result<Connection>{
    if fs::symlink_metadata(root).map_err(|e|e.to_string())?.file_type().is_symlink(){return Err("state root cannot be a symlink".into());}
    if fs::read_to_string(root.join("OWNED")).map_err(|e|e.to_string())?!=OWNER_MARKER{return Err("unowned state".into());}
    for name in ["state.sqlite","state.sqlite-wal","state.sqlite-shm","OWNED"] {if fs::symlink_metadata(root.join(name)).is_ok_and(|m|m.file_type().is_symlink()){return Err("state file symlink rejected".into());}}
    let mut c=sql(Connection::open(root.join("state.sqlite")))?;
    sql(c.busy_timeout(Duration::from_secs(2)))?;
    sql(c.execute_batch("PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;"))?;
    loop {
        let version:i64=sql(c.query_row("PRAGMA user_version",[],|r|r.get(0)))?;
        if version==3 { break; }
        let tx=sql(c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate))?;
        let confirmed:i64=sql(tx.query_row("PRAGMA user_version",[],|r|r.get(0)))?;
        if confirmed!=version {tx.rollback().map_err(|e|e.to_string())?;continue;}
        match version {
            1=>sql(tx.execute_batch(include_str!("schema_m1.sql")))?,
            2=>sql(tx.execute_batch(include_str!("schema_m2.sql")))?,
            _=>return Err(format!("unsupported database migration version {version}")),
        }
        sql(tx.commit())?;
    }
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
        if !matches!(operation.as_str(),"audit"|"baselines"|"finalize"){return Err("unsupported scientific operation".into());}
        if *operation=="audit" && spec.is_none(){return Err("audit requires an immutable project spec".into());}
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
    let sha=format!("{:x}",digest.finalize());let conn=connect(root)?;
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
    if path.exists(){return Err("owned dataset path already exists".into());}
    let temp=folder.join(format!(".{id}.staging"));
    let mut src=fs::File::open(source).map_err(|e|e.to_string())?;let mut dst=fs::OpenOptions::new().write(true).create_new(true).open(&temp).map_err(|e|e.to_string())?;
    std::io::copy(&mut src,&mut dst).map_err(|e|e.to_string())?;dst.sync_all().map_err(|e|e.to_string())?;
    let copied=fs::read(&temp).map_err(|e|e.to_string())?;if hash(&copied)!=sha{let _=fs::remove_file(&temp);return Err("dataset changed while staging".into());}
    fs::set_permissions(&temp,std::fs::Permissions::from_mode(0o444)).map_err(|e|e.to_string())?;fs::rename(&temp,&path).map_err(|e|e.to_string())?;
    let tx=sql(conn.unchecked_transaction())?;
    sql(tx.execute("INSERT INTO scientific_datasets(project,id,path,bytes,sha256,registered_ms) VALUES(?1,?2,?3,?4,?5,?6)",params![project,id,path.to_string_lossy(),bytes as i64,sha,now_ms()]))?;
    sql(tx.execute("INSERT INTO events(project,kind,payload) VALUES(?1,'dataset.registered',?2)",params![project,json!({"dataset_id":id,"sha256":sha,"bytes":bytes}).to_string()]))?;
    tx.commit().map_err(|e|e.to_string())?;Ok(json!({"project_id":project,"dataset_id":id,"sha256":sha,"bytes":bytes,"reused":false}))
}

pub fn submit(root:&Path,s:JobSpec)->Result<JobView>{
    validate_spec(&s)?;let c=connect(root)?;
    let tx=sql(c.unchecked_transaction())?;
    let queue_project:i64=sql(tx.query_row("SELECT count(*) FROM jobs WHERE project=?1 AND status='queued'",[&s.project_id],|r|r.get(0)))?;let queue_device:i64=sql(tx.query_row("SELECT count(*) FROM jobs WHERE status='queued'",[],|r|r.get(0)))?;
    if queue_project>=128||queue_device>=1024{return Err("bounded execution queue is full".into());}
    let previous:Option<String>=sql(tx.query_row("SELECT spec FROM jobs WHERE id=?1 AND project=?2",params![s.job_id,s.project_id],|r|r.get(0)).optional())?;
    let encoded=serde_json::to_string(&s).map_err(|e|e.to_string())?;
    if let Some(old)=previous {if old!=encoded{return Err("job id reused with a different specification".into());}let row=load_job(&tx,&s.project_id,&s.job_id)?;tx.commit().map_err(|e|e.to_string())?;return Ok(row);}
    let (project_state,project_policy):(String,String)=sql(tx.query_row("SELECT state,policy FROM projects WHERE id=?1",[&s.project_id],|r|Ok((r.get(0)?,r.get(1)?))))?;
    if project_state!="created" {return Err(format!("project does not admit work in state {project_state}"));}
    let _p:Policy=serde_json::from_str(&project_policy).map_err(|e|e.to_string())?;
    let task_json:Option<String>=sql(tx.query_row("SELECT data FROM tasks WHERE project=?1 AND id=?2",params![s.project_id,s.task_id],|r|r.get(0)).optional())?;
    let task_json=task_json.ok_or("unknown task")?;
    let task:crate::core::Task=serde_json::from_str(&task_json).map_err(|e|e.to_string())?;
    if task.status!="active"||task.generation!=s.generation||task.intent_revision!=sql(tx.query_row("SELECT intent_revision FROM projects WHERE id=?1",[&s.project_id],|r|r.get::<_,u64>(0)))?||task.expires_ms.is_none_or(|end|end<now_ms().saturating_add(s.timeout_ms as i64)){return Err("stale or too-short task lease".into());}
    sql(tx.execute("INSERT INTO jobs(id,project,task,generation,spec,status,submitted_ms) VALUES(?1,?2,?3,?4,?5,'queued',?6)",params![s.job_id,s.project_id,s.task_id,s.generation,encoded,now_ms()]))?;
    if let Probe::Scientific{operation,dataset_id,spec:scientific_spec,candidate_id,test_access_id}=&s.probe {
        let dataset:Option<(String,i64,String)>=sql(tx.query_row("SELECT path,bytes,sha256 FROM scientific_datasets WHERE project=?1 AND id=?2",params![s.project_id,dataset_id],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional())?;
        let Some((_,_,dataset_hash))=dataset else{return Err("scientific job requires a registered project dataset".into());};
        if *operation=="audit" && scientific_spec.is_some(){
            let out=serde_json::to_string(scientific_spec.as_ref().unwrap()).map_err(|e|e.to_string())?;
            if out.len()>4096{return Err("scientific project spec exceeds bound".into());}
        }
        if *operation=="finalize" {
            let results_path=root.join("workflows").join(&s.project_id).join("results.json");
            let result:Value=serde_json::from_slice(&fs::read(&results_path).map_err(|_|"candidate has no committed development results")?).map_err(|e|e.to_string())?;
            if result["dataset_sha256"]!=dataset_hash||!result["candidates"].as_array().is_some_and(|items|items.iter().any(|item|item["candidate_id"]==candidate_id.as_deref().unwrap()&&item["status"]=="evaluated")){
                return Err("candidate is not eligible under the registered dataset results".into());
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
    Ok(JobView{spec:serde_json::from_str(&row.0).map_err(|e|e.to_string())?,status:row.1,reason:row.2,pid:row.3,submitted_ms:row.4,started_ms:row.5,ended_ms:row.6,artifacts:serde_json::from_str(&row.7).map_err(|e|e.to_string())?,telemetry:telemetry.map(|(sampled_ms,cpu_percent,rss_bytes,peak_rss_bytes)|ResourceSample{sampled_ms,cpu_percent,rss_bytes,peak_rss_bytes})})
}

pub fn inspect(root:&Path,p:&str,id:&str)->Result<JobView>{let c=connect(root)?;load_job(&c,p,id)}
fn save_action(root:&Path,p:&str,a:&Action)->Result<()>{let c=connect(root)?;sql(c.execute("UPDATE actions SET data=?3 WHERE project=?1 AND id=?2",params![p,a.action_id,serde_json::to_string(a).map_err(|e|e.to_string())?]))?;Ok(())}

pub fn cancel(root:&Path,p:&str,id:&str)->Result<JobView>{
    let c=connect(root)?;let tx=sql(c.unchecked_transaction())?;
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
    let source=PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../python/scientific-requirements.txt");
    let install=Command::new(&python).args(["-m","pip","install","--no-input","--disable-pip-version-check","--only-binary=:all:","--no-cache-dir","-r"]).arg(&source)
        .env("PIP_NO_INPUT","1").stdout(Stdio::null()).stderr(Stdio::null()).spawn().map_err(|e|e.to_string())?;
    let mut child=install;let deadline=Instant::now()+Duration::from_secs(1200);let status=loop{if Instant::now()>=deadline{let _=child.kill();let _=child.wait();return Err("scientific profile installation exceeded 20-minute limit".into());}if let Some(s)=child.try_wait().map_err(|e|e.to_string())?{break s;}thread::sleep(Duration::from_millis(100));};
    if !status.success(){return Err(format!("scientific profile installation failed with {status}"));}
    let verify=Command::new(&python).args(["-c","import numpy,pandas,scipy,sklearn,joblib; assert (numpy.__version__,pandas.__version__,scipy.__version__,sklearn.__version__,joblib.__version__)==('2.2.6','2.3.0','1.16.1','1.8.0','1.5.1')"])
        .env_clear().env("PATH","/usr/bin:/bin").stdout(Stdio::null()).stderr(Stdio::null()).status().map_err(|e|e.to_string())?;
    if !verify.success(){return Err("scientific profile version verification failed".into());}
    Ok(json!({"id":id,"status":"ready","profile":"m2-cpu-v1","python":python,"installed_bytes":environment_size(&expected)?}))
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
    valid_id(id)?;let c=connect(root)?;let tx=sql(c.unchecked_transaction())?;
    let (path,status):(String,String)=sql(tx.query_row("SELECT path,status FROM environments WHERE id=?1",[id],|r|Ok((r.get(0)?,r.get(1)?))))?;
    let leases:i64=sql(tx.query_row("SELECT count(*) FROM environment_leases WHERE environment=?1",[id],|r|r.get(0)))?;if leases>0{return Err("environment is leased by active or queued work".into());}
    if status=="building"{return Err("cannot remove an environment while it is building".into());}
    sql(tx.execute("UPDATE environments SET status='deleting' WHERE id=?1",[id]))?;tx.commit().map_err(|e|e.to_string())?;
    let expected=root.join("environments").join(id);if Path::new(&path)!=expected{return Err("managed environment path does not match its identity".into());}
    if expected.exists(){safe_remove(root,&expected)?;}
    sql(c.execute("DELETE FROM environments WHERE id=?1",[id]))?;Ok(())
}

pub fn pin_artifact(root:&Path,p:&str,job:&str,pinned:bool)->Result<()> {
    let c=connect(root)?;let changed=sql(c.execute("UPDATE managed_artifacts SET pinned=?3 WHERE project=?1 AND job=?2",params![p,job,pinned as i64]))?;
    if changed==0{return Err("job has no managed artifact to pin".into());}Ok(())
}

pub fn collect_artifacts(root:&Path)->Result<u64>{
    let c=connect(root)?;let tx=sql(c.unchecked_transaction())?;
    let mut q=sql(tx.prepare("SELECT j.project,j.id FROM jobs j WHERE j.status IN('succeeded','failed','timed_out','cancelled','rejected','interrupted') AND NOT EXISTS(SELECT 1 FROM managed_artifacts a WHERE a.project=j.project AND a.job=j.id AND a.pinned=1) AND NOT EXISTS(SELECT 1 FROM environment_leases l WHERE l.project=j.project AND l.job=j.id)"))?;
    let rows=sql(q.query_map([],|r|Ok((r.get::<_,String>(0)?,r.get::<_,String>(1)?))))?.collect::<rusqlite::Result<Vec<_>>>().map_err(|e|e.to_string())?;drop(q);let mut reclaimed=0u64;
    for(p,j)in rows {let dir=root.join("artifacts").join(&p).join(&j);if dir.exists(){let n=tree_size(&dir)?;safe_remove(root,&dir)?;reclaimed=reclaimed.saturating_add(n);}let log=root.join("logs").join(&p).join(format!("{j}.log"));if log.exists(){let n=fs::metadata(&log).map_err(|e|e.to_string())?.len();safe_remove(root,&log)?;reclaimed=reclaimed.saturating_add(n);}sql(tx.execute("DELETE FROM managed_artifacts WHERE project=?1 AND job=?2 AND pinned=0",params![p,j]))?;}
    tx.commit().map_err(|e|e.to_string())?;Ok(reclaimed)
}
fn tree_size(path:&Path)->Result<u64>{let mut total=0u64;for e in fs::read_dir(path).map_err(|e|e.to_string())?{let e=e.map_err(|e|e.to_string())?;let m=fs::symlink_metadata(e.path()).map_err(|e|e.to_string())?;if m.file_type().is_symlink(){return Err("managed artifact symlink rejected".into());}if m.is_file(){total=total.saturating_add(m.len());}else if m.is_dir(){total=total.saturating_add(tree_size(&e.path())?);}}Ok(total)}
fn sample_process(root:&Path,project:&str,job:&str,pid:u32)->Result<Option<ResourceSample>>{
    let output=match Command::new("/bin/ps").args(["-o","%cpu=","-o","rss=","-p",&pid.to_string()]).output(){Ok(o) if o.status.success()=>o,_=>return Ok(None)};
    let fields=String::from_utf8_lossy(&output.stdout).split_whitespace().map(str::to_owned).collect::<Vec<_>>();if fields.len()<2{return Ok(None);}
    let cpu=fields[0].parse::<f64>().ok().filter(|x|x.is_finite()&&*x>=0.0);let rss=fields[1].parse::<u64>().ok().map(|x|x.saturating_mul(1024));if cpu.is_none()&&rss.is_none(){return Ok(None);}
    let c=connect(root)?;let old:Option<u64>=sql(c.query_row("SELECT peak_rss_bytes FROM job_telemetry WHERE project=?1 AND job=?2",params![project,job],|r|r.get(0)).optional())?;
    let peak=old.unwrap_or(0).max(rss.unwrap_or(0));let sampled_ms=now_ms();
    sql(c.execute("INSERT INTO job_telemetry(project,job,sampled_ms,cpu_percent,rss_bytes,peak_rss_bytes) VALUES(?1,?2,?3,?4,?5,?6) ON CONFLICT(project,job) DO UPDATE SET sampled_ms=excluded.sampled_ms,cpu_percent=excluded.cpu_percent,rss_bytes=excluded.rss_bytes,peak_rss_bytes=MAX(job_telemetry.peak_rss_bytes,excluded.peak_rss_bytes)",params![project,job,sampled_ms,cpu,rss,peak]))?;
    Ok(Some(ResourceSample{sampled_ms,cpu_percent:cpu,rss_bytes:rss,peak_rss_bytes:peak}))
}

pub struct Supervisor { root:PathBuf, lock:fs::File, executable:PathBuf, max_workers:usize, children:Vec<(JobView,Child,ChildStdin,Instant,Instant)> }
impl Supervisor {
    pub fn start(root:&Path,max_workers:usize)->Result<Self>{
        Self::start_with_executable(root,max_workers,&std::env::current_exe().map_err(|e|e.to_string())?)
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
        sql(c.execute("UPDATE jobs SET status='recovering',reason='supervisor restart; waiting for bounded worker deadline',recovered_deadline_ms=started_ms+CAST(json_extract(spec,'$.timeout_ms') AS INTEGER)+500 WHERE status IN('running','terminating')",[]))?;
        Ok(Self{root:root.to_path_buf(),lock,executable:executable.to_path_buf(),max_workers,children:vec![]})
    }
    pub fn serve(mut self)->Result<()> {
        let _exclusive_lock=&self.lock;
        STOP.store(false,Ordering::Relaxed);
        unsafe{libc::signal(libc::SIGINT,stop_signal as *const () as libc::sighandler_t);libc::signal(libc::SIGTERM,stop_signal as *const () as libc::sighandler_t);}
        loop {if STOP.load(Ordering::Relaxed){self.shutdown()?;return Ok(());}self.tick()?;thread::sleep(Duration::from_millis(POLL_MS));}
    }
    fn shutdown(&mut self)->Result<()>{
        while let Some((view,mut child,_control,started,_last_sample))=self.children.pop(){terminate(&mut child)?;self.finish(&view,"cancelled",Some("supervisor shutdown".into()),vec![],started.elapsed().as_millis() as u64)?;let stage=self.root.join("artifacts").join(&view.spec.project_id).join(format!(".{}.staging",view.spec.job_id));if stage.exists(){safe_remove(&self.root,&stage)?;}}
        Ok(())
    }
    pub fn tick(&mut self)->Result<()> {self.reconcile_queued()?;self.recover_expired()?;self.admit()?;self.reap()?;Ok(())}
    fn reconcile_queued(&self)->Result<()>{
        let c=connect(&self.root)?;let mut q=sql(c.prepare("SELECT j.project,j.id,j.generation,p.state,p.intent_revision,t.data FROM jobs j JOIN projects p ON p.id=j.project JOIN tasks t ON t.project=j.project AND t.id=j.task WHERE j.status='queued'"))?;
        let rows=sql(q.query_map([],|r|Ok((r.get::<_,String>(0)?,r.get::<_,String>(1)?,r.get::<_,u64>(2)?,r.get::<_,String>(3)?,r.get::<_,u64>(4)?,r.get::<_,String>(5)?))))?.collect::<rusqlite::Result<Vec<_>>>().map_err(|e|e.to_string())?;drop(q);
        for(p,id,generation,state,intent,task_data)in rows{let task:crate::core::Task=serde_json::from_str(&task_data).map_err(|e|e.to_string())?;let terminal=if state=="cancelled"{Some(("cancelled","project cancelled before admission"))}else if task.status!="active"||task.generation!=generation||task.intent_revision!=intent{Some(("rejected","task intent changed before admission"))}else{None};if let Some((status,reason))=terminal{let tx=sql(c.unchecked_transaction())?;sql(tx.execute("UPDATE jobs SET status=?3,reason=?4,ended_ms=?5 WHERE project=?1 AND id=?2 AND status='queued'",params![p,id,status,reason,now_ms()]))?;sql(tx.execute("INSERT INTO events(project,kind,payload) VALUES(?1,'job.terminal',?2)",params![p,json!({"job_id":id,"status":status,"reason":reason}).to_string()]))?;tx.commit().map_err(|e|e.to_string())?;}}
        Ok(())
    }
    fn recover_expired(&self)->Result<()> {
        let c=connect(&self.root)?;let mut q=sql(c.prepare("SELECT project,id,spec,stage FROM jobs WHERE status='recovering' AND recovered_deadline_ms<=?1"))?;
        let rows=sql(q.query_map([now_ms()],|r|Ok((r.get::<_,String>(0)?,r.get::<_,String>(1)?,r.get::<_,String>(2)?,r.get::<_,Option<String>>(3)?))))?.collect::<rusqlite::Result<Vec<_>>>().map_err(|e|e.to_string())?;drop(q);
        for (p,id,spec,stage) in rows {
            let s:JobSpec=serde_json::from_str(&spec).map_err(|e|e.to_string())?;let task_json:String=sql(c.query_row("SELECT data FROM tasks WHERE project=?1 AND id=?2",params![p,s.task_id],|r|r.get(0)))?;let task:crate::core::Task=serde_json::from_str(&task_json).map_err(|e|e.to_string())?;let (project_state,intent):(String,u64)=sql(c.query_row("SELECT state,intent_revision FROM projects WHERE id=?1",[&p],|r|Ok((r.get(0)?,r.get(1)?))))?;
            let eligible=matches!(project_state.as_str(),"created"|"paused")&&task.status=="active"&&task.generation==s.generation&&task.intent_revision==intent;
            let final_dir=self.root.join("artifacts").join(&p).join(&id);let stage_path=stage.clone().map(PathBuf::from);
            let mut recovered=if eligible&&final_dir.is_dir(){manifest_dir(&final_dir,s.max_output_bytes).ok()}else{None};
            if recovered.is_none()&&eligible&&!final_dir.exists(){if let Some(path)=&stage_path{if path.is_dir(){if let Ok(artifacts)=manifest_dir(path,s.max_output_bytes){let spec_file=path.join("job.json");if spec_file.exists(){let _=fs::remove_file(spec_file);}if !artifacts.is_empty(){ensure_dir(&self.root,final_dir.parent().unwrap())?;if fs::rename(path,&final_dir).is_ok(){recovered=Some(artifacts);}}}}}}
            if recovered.is_none(){if final_dir.exists(){let _=safe_remove(&self.root,&final_dir);}if let Some(path)=stage_path{if path.exists(){let _=safe_remove(&self.root,&path);}}}
            let view=JobView{spec:s.clone(),status:"recovering".into(),reason:None,pid:None,submitted_ms:0,started_ms:None,ended_ms:None,artifacts:vec![],telemetry:None};
            if let Some(artifacts)=recovered{self.finish(&view,"succeeded",Some("recovered published output after supervisor restart".into()),artifacts,s.timeout_ms)?;}
            else{let tx=sql(c.unchecked_transaction())?;sql(tx.execute("UPDATE jobs SET status='interrupted',reason='worker deadline elapsed while supervisor was stopped',ended_ms=?3,pid=NULL,stage=NULL WHERE project=?1 AND id=?2 AND status='recovering'",params![p,id,now_ms()]))?;sql(tx.execute("DELETE FROM environment_leases WHERE project=?1 AND job=?2",params![p,id]))?;settle_job_action(&tx,&p,&id,s.timeout_ms,"interrupted")?;sql(tx.execute("INSERT INTO events(project,kind,payload) VALUES(?1,'job.terminal',?2)",params![p,json!({"job_id":id,"status":"interrupted"}).to_string()]))?;tx.commit().map_err(|e|e.to_string())?;}
        }Ok(())
    }
    fn admit(&mut self)->Result<()> {
        while self.children.len()<self.max_workers {
            let c=connect(&self.root)?;let tx=sql(c.unchecked_transaction())?;
            let cursor:Option<String>=sql(tx.query_row("SELECT value FROM scheduler WHERE key='last_project'",[],|r|r.get(0)).optional())?;
            let eligible="j.status='queued' AND p.state='created' AND NOT EXISTS(SELECT 1 FROM questions q WHERE q.project=p.id AND q.answer IS NULL) AND NOT EXISTS(SELECT 1 FROM actions a WHERE a.project=j.project AND a.task=j.task AND json_extract(a.data,'$.status') NOT IN('succeeded','failed','cancelled'))";
            let rank="((SELECT count(*) FROM jobs prior WHERE prior.project=j.project AND prior.status IN('running','succeeded','failed','timed_out','cancelled','rejected','interrupted'))*1.0/COALESCE((SELECT weight FROM project_weights w WHERE w.project=j.project),1) - (?2-j.submitted_ms)/60000.0)";
            let now=now_ms();let row:Option<(String,String,String)>=sql(tx.query_row(
                &format!("SELECT j.id,j.project,j.spec FROM jobs j JOIN projects p ON p.id=j.project WHERE {eligible} AND j.project>?1 ORDER BY {rank},j.project,j.submitted_ms,j.id LIMIT 1"),rusqlite::params![cursor.as_deref().unwrap_or(""),now],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional())?;
            let row=match row {Some(x)=>Some(x),None=>sql(tx.query_row(&format!("SELECT j.id,j.project,j.spec FROM jobs j JOIN projects p ON p.id=j.project WHERE {eligible} ORDER BY {rank},j.project,j.submitted_ms,j.id LIMIT 1"),rusqlite::params![cursor.as_deref().unwrap_or(""),now],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional())?};
            let Some((id,p,spec))=row else {tx.commit().map_err(|e|e.to_string())?;break};
            let s:JobSpec=serde_json::from_str(&spec).map_err(|e|e.to_string())?;
            if let Err(reason)=admission(&tx,&s,self.max_workers) {
                if reason.starts_with("capacity unavailable:") || reason=="project worker capacity unavailable" || reason=="device worker capacity unavailable" || reason=="worker slot unavailable" {tx.execute("INSERT INTO scheduler(key,value) VALUES('last_project',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",[&p]).map_err(|e|e.to_string())?;tx.commit().map_err(|e|e.to_string())?;break;}
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
            let action_id=format!("job.{}",s.job_id);let action_exists:bool=sql(tx.query_row("SELECT EXISTS(SELECT 1 FROM actions WHERE project=?1 AND id=?2)",params![p,action_id],|r|r.get(0)))?;
            if action_exists{sql(tx.execute("UPDATE jobs SET status='rejected',reason='job action identity already exists',ended_ms=?3 WHERE project=?1 AND id=?2",params![p,id,now_ms()]))?;tx.commit().map_err(|e|e.to_string())?;continue;}
            ensure_disk_space(&self.root,s.max_output_bytes.saturating_add(LOG_CAP))?;
            let stage=self.root.join("artifacts").join(&p).join(format!(".{}.staging",id));
            ensure_dir(&self.root,stage.parent().unwrap())?;if stage.exists(){safe_remove(&self.root,&stage)?;}ensure_dir(&self.root,&stage)?;
            let log=self.root.join("logs").join(&p).join(format!("{id}.log"));ensure_dir(&self.root,log.parent().unwrap())?;
            let out=fs::File::create(&log).map_err(|e|e.to_string())?;let err=out.try_clone().map_err(|e|e.to_string())?;
            let spec_path=stage.join("job.json");fs::write(&spec_path,&spec).map_err(|e|e.to_string())?;
            let mut cmd=Command::new(&self.executable);cmd.arg("__worker").arg(&spec_path).arg(&stage).stdin(Stdio::null()).stdout(Stdio::from(out)).stderr(Stdio::from(err)).env_clear().env("PATH","/usr/bin:/bin").env("MLRRESCURIX_STATE",&self.root);
            cmd.env("MLRESCURIX_NUM_THREADS",s.threads.to_string());
            for key in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"] {cmd.env(key,s.threads.to_string());}
            if let Probe::Scientific{dataset_id,..}=&s.probe {
                let env_id=s.environment_id.as_deref().ok_or("scientific environment missing")?;
                let python=self.root.join("environments").join(env_id).join("bin/python3");
                let environment_status:String=sql(tx.query_row("SELECT status FROM environments WHERE id=?1",[env_id],|r|r.get(0)))?;
                if environment_status!="ready"||!python.is_file(){return Err("managed scientific interpreter is unavailable".into());}
                let (data_path,sha):(String,String)=sql(tx.query_row("SELECT path,sha256 FROM scientific_datasets WHERE project=?1 AND id=?2",params![p,dataset_id],|r|Ok((r.get(0)?,r.get(1)?))))?;
                let expected=self.root.join("datasets").join(&p).join(format!("{dataset_id}.csv"));
                if Path::new(&data_path)!=expected||hash(&fs::read(&expected).map_err(|e|e.to_string())?)!=sha{return Err("registered dataset snapshot failed integrity check".into());}
                let workflow=self.root.join("workflows").join(&p);ensure_dir(&self.root,&workflow)?;
                let pythonpath=PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../python/src");
                cmd.env("MLRRESCURIX_PYTHON",python).env("MLRRESCURIX_DATASET",expected)
                    .env("MLRRESCURIX_WORKFLOW",workflow).env("MLRRESCURIX_PYTHONPATH",pythonpath);
            }
            cmd.process_group(0);
            cmd.stdin(Stdio::piped());
            let mut child=match cmd.spawn(){Ok(c)=>c,Err(e)=>{safe_remove(&self.root,&stage)?;sql(tx.execute("UPDATE jobs SET status='failed',reason=?3,ended_ms=?4 WHERE project=?1 AND id=?2",params![p,id,format!("worker launch failed: {e}"),now_ms()]))?;tx.commit().map_err(|e|e.to_string())?;continue;}};
            let (budget,capacity)=reservation(&tx,&p,&s)?;
            let lease_id=t.lease_id.clone().ok_or("task has no execution lease")?;let deadline=now_ms().saturating_add(s.timeout_ms as i64);
            let mut action=Action{action_id:action_id.clone(),task_id:s.task_id.clone(),generation:s.generation,intent_revision:t.intent_revision,lease_id,input:spec.clone(),budget:budget.clone(),capacity:capacity.clone(),deadline_ms:deadline,status:"dispatched".into(),attempt_id:Some(format!("{}.1",s.job_id)),external_request_id:Some(s.job_id.clone()),response:None,receipt:None,uncertainty:None};
            sql(tx.execute("INSERT INTO actions(project,id,task,data) VALUES(?1,?2,?3,?4)",params![p,action_id,s.task_id,serde_json::to_string(&action).map_err(|e|e.to_string())?]))?;
            sql(tx.execute("INSERT INTO reservations(project,action,budget,capacity,held) VALUES(?1,?2,?3,?4,1)",params![p,action_id,serde_json::to_string(&budget).map_err(|e|e.to_string())?,serde_json::to_string(&capacity).map_err(|e|e.to_string())?]))?;
            sql(tx.execute("UPDATE jobs SET status='running',action=?3,pid=?4,started_ms=?5,stage=?6,log_path=?7 WHERE project=?1 AND id=?2",params![p,id,action_id,child.id(),now_ms(),stage.to_string_lossy(),log.to_string_lossy()]))?;
            sql(tx.execute("INSERT INTO scheduler(key,value) VALUES('last_project',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",[&p]))?;
            sql(tx.execute("INSERT INTO events(project,kind,payload) VALUES(?1,'job.started',?2)",params![p,json!({"job_id":id,"pid":child.id()}).to_string()]))?;
            if let Some(environment)=&s.environment_id {sql(tx.execute("INSERT INTO environment_leases(environment,project,job) VALUES(?1,?2,?3)",params![environment,p,id]))?;}
            tx.execute("INSERT INTO events(project,kind,payload) VALUES(?1,'action.dispatched',?2)",params![p,json!({"action_id":action_id,"job_id":id}).to_string()]).map_err(|e|e.to_string())?;
            tx.commit().map_err(|e|e.to_string())?;
            let mut control=child.stdin.take().ok_or("worker start gate unavailable")?;
            if let Err(e)=control.write_all(&[1]) {action.status="outcome_unknown".into();action.uncertainty=Some(format!("worker start gate response lost: {e}"));save_action(&self.root,&p,&action)?;let _=terminate(&mut child);}
            let started=Instant::now();self.children.push((JobView{spec:s,status:"running".into(),reason:None,pid:Some(child.id()),submitted_ms:0,started_ms:Some(now_ms()),ended_ms:None,artifacts:vec![],telemetry:None},child,control,started,started-Duration::from_secs(1)));
        }Ok(())
    }
    fn reap(&mut self)->Result<()> {
        let mut index=0;
        while index<self.children.len(){
            let (mut view,mut child,control,started,mut last_sample)=self.children.swap_remove(index);let elapsed=started.elapsed();
            if last_sample.elapsed()>=Duration::from_secs(1){if let Some(sample)=sample_process(&self.root,&view.spec.project_id,&view.spec.job_id,child.id())?{view.telemetry=Some(sample);}last_sample=Instant::now();}
            let cancel:bool={let c=connect(&self.root)?;let (requested,state,intent,task):(bool,String,u64,String)=sql(c.query_row("SELECT j.cancel_requested,p.state,p.intent_revision,t.data FROM jobs j JOIN projects p ON p.id=j.project JOIN tasks t ON t.project=j.project AND t.id=j.task WHERE j.project=?1 AND j.id=?2",params![view.spec.project_id,view.spec.job_id],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?))))?;let t:crate::core::Task=serde_json::from_str(&task).map_err(|e|e.to_string())?;requested||state=="cancelled"||t.generation!=view.spec.generation||t.intent_revision!=intent||t.status!="active"||t.expires_ms.is_none_or(|end|end<=now_ms())};
            let timed_out=elapsed>Duration::from_millis(view.spec.timeout_ms);
            if (cancel||timed_out)&&child.try_wait().map_err(|e|e.to_string())?.is_none(){terminate(&mut child)?;}
            if child.try_wait().map_err(|e|e.to_string())?.is_some(){
                let status=child.wait().map_err(|e|e.to_string())?;
                let reason=if cancel{Some("cancelled".to_string())}else if timed_out||status.code()==Some(124){Some("deadline exceeded".to_string())}else if !status.success(){Some(format!("worker exited with {status}"))}else{None};
                let (artifacts,final_reason)=if reason.is_none(){match self.publish(&view){Ok(a)=>(a,None),Err(e)=>{let stage=self.root.join("artifacts").join(&view.spec.project_id).join(format!(".{}.staging",view.spec.job_id));if stage.exists(){let _=safe_remove(&self.root,&stage);} (vec![],Some(format!("artifact validation: {e}")))}}}else{(vec![],reason)};
                let ok=final_reason.is_none();if !ok{let stage=self.root.join("artifacts").join(&view.spec.project_id).join(format!(".{}.staging",view.spec.job_id));if stage.exists(){safe_remove(&self.root,&stage)?;}}
                self.finish(&view,if ok{"succeeded"}else if cancel{"cancelled"}else if timed_out||status.code()==Some(124){"timed_out"}else{"failed"},final_reason,artifacts,elapsed.as_millis() as u64)?;
            }else{self.children.push((view,child,control,started,last_sample));index+=1;}
        }Ok(())
    }
    fn publish(&self,v:&JobView)->Result<Vec<Value>>{
        let stage=self.root.join("artifacts").join(&v.spec.project_id).join(format!(".{}.staging",v.spec.job_id));
        let mut artifacts=vec![];let mut total=0u64;
        for entry in fs::read_dir(&stage).map_err(|e|e.to_string())? {let entry=entry.map_err(|e|e.to_string())?;let meta=fs::symlink_metadata(entry.path()).map_err(|e|e.to_string())?;if meta.file_type().is_symlink()||!meta.is_file(){return Err("worker emitted unsupported artifact type".into());}let name=entry.file_name().to_string_lossy().into_owned();if name=="job.json"{continue;}let bytes=fs::read(entry.path()).map_err(|e|e.to_string())?;total=total.checked_add(bytes.len() as u64).ok_or("artifact size overflow")?;if total>v.spec.max_output_bytes{return Err("worker exceeded artifact output limit".into());}artifacts.push(json!({"name":name,"bytes":bytes.len(),"sha256":hash(&bytes)}));}
        let c=connect(&self.root)?;let task:String=sql(c.query_row("SELECT data FROM tasks WHERE project=?1 AND id=?2",params![v.spec.project_id,v.spec.task_id],|r|r.get(0)))?;let t:crate::core::Task=serde_json::from_str(&task).map_err(|e|e.to_string())?;let (state,intent):(String,u64)=sql(c.query_row("SELECT state,intent_revision FROM projects WHERE id=?1",[&v.spec.project_id],|r|Ok((r.get(0)?,r.get(1)?))))?;
        if !matches!(state.as_str(),"created"|"paused")||t.status!="active"||t.generation!=v.spec.generation||t.intent_revision!=intent{return Err("stale worker result fenced before publication".into());}
        let final_dir=self.root.join("artifacts").join(&v.spec.project_id).join(&v.spec.job_id);if final_dir.exists(){return Err("artifact destination already exists".into());}
        if artifacts.is_empty(){safe_remove(&self.root,&stage)?;return Ok(artifacts);}
        let spec_path=stage.join("job.json");if spec_path.exists(){fs::remove_file(spec_path).map_err(|e|e.to_string())?;}
        fs::rename(&stage,&final_dir).map_err(|e|e.to_string())?;Ok(artifacts)
    }
    fn finish(&self,v:&JobView,status:&str,reason:Option<String>,artifacts:Vec<Value>,elapsed_ms:u64)->Result<()>{
        let c=connect(&self.root)?;let tx=sql(c.unchecked_transaction())?;
        if let Probe::Scientific{operation,dataset_id:_,candidate_id,test_access_id,..}=&v.spec.probe {
            let result_name=match operation.as_str(){"audit"=>"dataset_manifest.json","baselines"=>"results.json","finalize"=>"final.json",_=>""};
            let workflow=self.root.join("workflows").join(&v.spec.project_id);let result_path=workflow.join(result_name);
            let result_hash=if status=="succeeded"&&result_path.is_file(){Some(hash(&fs::read(&result_path).map_err(|e|e.to_string())?))}else{None};
            sql(tx.execute("UPDATE scientific_jobs SET result_sha256=?3 WHERE project=?1 AND job=?2",params![v.spec.project_id,v.spec.job_id,result_hash]))?;
            if operation=="finalize"&&status=="succeeded" {
                let access=test_access_id.as_deref().ok_or("final test key missing")?;let candidate=candidate_id.as_deref().ok_or("candidate missing")?;
                let gate=serde_json::from_str::<Value>(&fs::read_to_string(workflow.join("test_access.json")).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
                let final_result=serde_json::from_str::<Value>(&fs::read_to_string(&result_path).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
                if gate["test_access_id"]!=access||gate["candidate_id"]!=candidate||gate["status"]!="consumed"||final_result["test_access_id"]!=access{return Err("final assessment identity failed verification".into());}
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
        for a in &artifacts {sql(tx.execute("INSERT INTO managed_artifacts(project,job,name,bytes,sha256) VALUES(?1,?2,?3,?4,?5)",params![v.spec.project_id,v.spec.job_id,a["name"].as_str().unwrap_or(""),a["bytes"].as_u64().unwrap_or(0),a["sha256"].as_str().unwrap_or("")]))?;}
        if let Some(log)=sql(tx.query_row("SELECT log_path FROM jobs WHERE project=?1 AND id=?2",params![v.spec.project_id,v.spec.job_id],|r|r.get::<_,Option<String>>(0)))? {if let Ok(m)=fs::metadata(&log){if m.len()>LOG_CAP {let f=fs::OpenOptions::new().write(true).open(log).map_err(|e|e.to_string())?;f.set_len(LOG_CAP).map_err(|e|e.to_string())?;}}}
        sql(tx.execute("INSERT INTO events(project,kind,payload) VALUES(?1,'job.terminal',?2)",params![v.spec.project_id,json!({"job_id":v.spec.job_id,"status":status,"reason":reason,"artifacts":artifacts}).to_string()]))?;
        tx.commit().map_err(|e|e.to_string())?;Ok(())
    }
}

fn reconcile_final_tests(root:&Path,c:&Connection)->Result<()> {
    let mut q=sql(c.prepare("SELECT project,access_id,dataset_sha256,candidate_id FROM final_test_gates WHERE status='reserved'"))?;
    let rows=sql(q.query_map([],|r|Ok((r.get::<_,String>(0)?,r.get::<_,String>(1)?,r.get::<_,String>(2)?,r.get::<_,String>(3)?))))?
        .collect::<rusqlite::Result<Vec<_>>>().map_err(|e|e.to_string())?;drop(q);
    for (project,access,dataset,candidate) in rows {
        let workflow=root.join("workflows").join(&project);
        let gate_path=workflow.join("test_access.json");let result_path=workflow.join("final.json");
        if !gate_path.is_file()||gate_path.symlink_metadata().map_err(|e|e.to_string())?.file_type().is_symlink()||!result_path.is_file(){continue;}
        let gate:Value=serde_json::from_slice(&fs::read(&gate_path).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        let result:Value=serde_json::from_slice(&fs::read(&result_path).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        if gate["status"]=="consumed"&&gate["test_access_id"]==access&&gate["dataset_sha256"]==dataset&&gate["candidate_id"]==candidate&&result["test_access_id"]==access {
            let frozen=gate["frozen_model_sha256"].as_str().ok_or("final gate has no frozen artifact hash")?;
            let hash_result=hash(&fs::read(&result_path).map_err(|e|e.to_string())?);
            let tx=sql(c.unchecked_transaction())?;
            sql(tx.execute("UPDATE final_test_gates SET status='consumed',frozen_model_sha256=?4,result_sha256=?5 WHERE project=?1 AND access_id=?2 AND dataset_sha256=?3 AND status='reserved'",params![project,access,dataset,frozen,hash_result]))?;
            sql(tx.execute("INSERT INTO events(project,kind,payload) VALUES(?1,'final_test.reconciled',?2)",params![project,json!({"test_access_id":access,"candidate_id":candidate,"result_sha256":hash_result}).to_string()]))?;
            tx.commit().map_err(|e|e.to_string())?;
        }
    }
    Ok(())
}

fn admission(c:&Connection,s:&JobSpec,worker_ceiling:usize)->Result<()>{
    let (device,project):(String,String)=sql(c.query_row("SELECT d.policy,p.policy FROM device d JOIN projects p ON p.id=?1 WHERE d.id=1",[&s.project_id],|r|Ok((r.get(0)?,r.get(1)?))))?;
    let d:Policy=serde_json::from_str(&device).map_err(|e|e.to_string())?;let p:Policy=serde_json::from_str(&project).map_err(|e|e.to_string())?;
    if d.capacity.keys().chain(p.capacity.keys()).any(|k|!matches!(k.as_str(),"workers"|"cpu_threads")){return Err("M1 cannot enforce a configured capacity dimension".into());}
    let requested=BTreeMap::from([("workers".to_string(),1),("cpu_threads".to_string(),s.threads as i64)]);
    for (policy,scope) in [(&d,None),(&p,Some(&s.project_id))] {
        let mut used:BTreeMap<String,i64>=policy.capacity.keys().map(|k|(k.clone(),0)).collect();
        let mut reservations=sql(c.prepare("SELECT capacity FROM reservations WHERE held=1 AND (?1 IS NULL OR project=?1)"))?;
        for row in sql(reservations.query_map([scope],|r|r.get::<_,String>(0)))? {let a:BTreeMap<String,i64>=serde_json::from_str(&sql(row)?).map_err(|e|e.to_string())?;for(k,v)in a{if let Some(sum)=used.get_mut(&k){*sum=sum.checked_add(v).ok_or("capacity overflow")?;}}}
        for (k,limit) in &policy.capacity {let request=requested.get(k).ok_or_else(||format!("M1 cannot enforce capacity dimension {k}"))?;if request>limit{return Err(format!("job exceeds {k} limit"));}if used.get(k).copied().unwrap_or(0).checked_add(*request).ok_or("capacity overflow")?>*limit{return Err(format!("capacity unavailable: {k}"));}}
    }
    let active:i64=sql(c.query_row("SELECT count(*) FROM jobs WHERE status IN('running','terminating','recovering')",[],|r|r.get(0)))?;
    if active>=worker_ceiling as i64{return Err("worker slot unavailable".into());}
    let device_existing:i64=sql(c.query_row("SELECT COALESCE(SUM(json_extract(capacity,'$.workers')),0) FROM reservations WHERE held=1",[],|r|r.get(0)))?;
    if let Some(n)=d.capacity.get("workers"){if device_existing>=*n{return Err("device worker capacity unavailable".into());}}
    let _=reservation(c,&s.project_id,s)?;
    Ok(())
}

fn reservation(c:&Connection,p:&str,s:&JobSpec)->Result<(BTreeMap<String,i64>,BTreeMap<String,i64>)>{
    let (device,project):(String,String)=sql(c.query_row("SELECT d.policy,p.policy FROM device d JOIN projects p ON p.id=?1 WHERE d.id=1",[p],|r|Ok((r.get(0)?,r.get(1)?))))?;
    let d:Policy=serde_json::from_str(&device).map_err(|e|e.to_string())?;let pr:Policy=serde_json::from_str(&project).map_err(|e|e.to_string())?;
    if d.budget.keys().ne(pr.budget.keys()){return Err("M1 requires matching device/project budget dimensions".into());}
    if !d.budget.contains_key("wall_ms"){return Err("M1 execution policy must declare wall_ms budget".into());}
    let budget:BTreeMap<String,i64>=d.budget.keys().map(|k|(k.clone(),if k=="wall_ms"{s.timeout_ms as i64}else{0})).collect();
    let capacity:BTreeMap<String,i64>=d.capacity.keys().map(|k|(k.clone(),match k.as_str(){"workers"=>1,"cpu_threads"=>s.threads as i64,_=>-1})).collect();
    if capacity.values().any(|v|*v<0)||d.capacity.keys().ne(pr.capacity.keys()){return Err("M1 capacity dimensions must be workers and optional cpu_threads".into());}
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
fn ensure_disk_space(root:&Path,output:u64)->Result<()>{
    let p=std::ffi::CString::new(root.as_os_str().as_encoded_bytes()).map_err(|e|e.to_string())?;
    let mut stats=unsafe{std::mem::zeroed::<libc::statvfs>()};
    if unsafe{libc::statvfs(p.as_ptr(),&mut stats)}!=0{return Err("cannot inspect available state storage".into());}
    let free=(stats.f_bavail as u64).saturating_mul(stats.f_frsize as u64);
    if !disk_space_allows(free,output,DISK_HEADROOM){return Err("insufficient free disk for bounded job outputs and storage headroom".into());}Ok(())
}
fn disk_space_allows(free:u64,output:u64,headroom:u64)->bool{free>=output.saturating_add(headroom)}

#[cfg(test)] mod tests {use super::*;#[test]fn low_disk_admission_boundary(){assert!(disk_space_allows(100,40,60));assert!(!disk_space_allows(99,40,60));assert!(disk_space_allows(u64::MAX,10,u64::MAX));}}

/// Entrypoint for the private probe process. The control plane never accepts an executable path.
pub fn worker(spec_path:&Path,output:&Path)->Result<i32>{
    let bytes=fs::read(spec_path).map_err(|e|e.to_string())?;let s:JobSpec=serde_json::from_slice(&bytes).map_err(|e|e.to_string())?;validate_spec(&s)?;
    let mut control=std::io::stdin();let mut gate=[0u8;1];if control.read_exact(&mut gate).is_err()||gate[0]!=1{return Ok(125);}
    let alive=Arc::new(AtomicBool::new(true));let watcher=alive.clone();thread::spawn(move||{let mut byte=[0u8;1];loop{match control.read(&mut byte){Ok(0)|Err(_)=>{watcher.store(false,Ordering::Release);break},Ok(_)=>{}}}});
    match s.probe {
        Probe::Sleep{millis}=>{let deadline=Instant::now()+Duration::from_millis(millis.min(s.timeout_ms));while Instant::now()<deadline{if !alive.load(Ordering::Acquire){return Ok(125)}thread::sleep(Duration::from_millis(10));}if millis>s.timeout_ms{return Ok(124)}},
        Probe::Fail{code}=>return Ok(code as i32),
        Probe::Write{name,content}=>{if !alive.load(Ordering::Acquire){return Ok(125)}let p=output.join(name);let mut f=fs::OpenOptions::new().write(true).create_new(true).open(p).map_err(|e|e.to_string())?;f.write_all(content.as_bytes()).map_err(|e|e.to_string())?;f.sync_all().map_err(|e|e.to_string())?;},
        Probe::Scientific{operation,dataset_id:_,spec,candidate_id,test_access_id}=>{
            if !alive.load(Ordering::Acquire){return Ok(125)}
            let python=std::env::var_os("MLRRESCURIX_PYTHON").ok_or("scientific interpreter was not provisioned")?;
            let dataset=std::env::var_os("MLRRESCURIX_DATASET").ok_or("registered dataset path unavailable")?;
            let workflow=PathBuf::from(std::env::var_os("MLRRESCURIX_WORKFLOW").ok_or("workflow directory unavailable")?);
            let pythonpath=std::env::var_os("MLRRESCURIX_PYTHONPATH").ok_or("scientific worker package unavailable")?;
            let req=output.join("project-spec.json");if let Some(value)=spec{fs::write(&req,serde_json::to_vec(&value).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;}
            let mut command=Command::new(python);command.arg("-m").arg("mlrescurix_worker.scientific.cli").env_clear()
                .env("PATH",format!("{}:/usr/bin:/bin",PathBuf::from(std::env::var_os("MLRRESCURIX_PYTHON").unwrap()).parent().unwrap().display()))
                .env("PYTHONPATH",pythonpath).env("TMPDIR",output.join("tmp"));fs::create_dir_all(output.join("tmp")).map_err(|e|e.to_string())?;
            match operation.as_str(){
                "audit"=>{command.arg("audit").arg("--csv").arg(dataset).arg("--spec").arg(&req).arg("--output").arg(&workflow);},
                "baselines"=>{command.arg("baselines").arg("--output").arg(&workflow);},
                "finalize"=>{command.arg("finalize").arg("--output").arg(&workflow).arg("--candidate").arg(candidate_id.ok_or("candidate missing")?).arg("--test-access-id").arg(test_access_id.ok_or("test access id missing")?);},
                _=>return Err("unsupported scientific operation".into()),
            }
            command.stdout(Stdio::inherit()).stderr(Stdio::inherit()).stdin(Stdio::null());
            let mut child=command.spawn().map_err(|e|format!("scientific worker launch failed: {e}"))?;
            loop {if !alive.load(Ordering::Acquire){let _=child.kill();let _=child.wait();return Ok(125);}match child.try_wait().map_err(|e|e.to_string())?{Some(status)=>{if !status.success(){return Ok(status.code().unwrap_or(1));}break;},None=>thread::sleep(Duration::from_millis(50))}}
            if req.exists(){fs::remove_file(&req).map_err(|e|e.to_string())?;}let tmp=output.join("tmp");if tmp.exists(){safe_remove(Path::new(&std::env::var("MLRRESCURIX_STATE").map_err(|e|e.to_string())?),&tmp)?;}
            let result_name=match operation.as_str(){"audit"=>"dataset_manifest.json","baselines"=>"results.json","finalize"=>"final.json",_=>unreachable!()};
            let evidence=workflow.join(result_name);if !evidence.is_file(){return Err("scientific worker produced no committed result".into());}
            if operation=="finalize" {
                let final_result:Value=serde_json::from_slice(&fs::read(&evidence).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
                if final_result["status"]!="verified_development_only" {return Ok(3);}
            }
            let target=output.join(result_name);fs::copy(evidence,target).map_err(|e|e.to_string())?;
        }
    }
    Ok(0)
}
