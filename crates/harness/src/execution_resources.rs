//! Docker resource receipts and lease-fenced lifecycle. No schema migration.
use super::*;
const PREFIX: &str = "io.mlrescurix.";
const OWNER: &str = "mlrescurix.scientific.v1";
const RECEIPT_RESERVE: u64 = 64 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ResourceRequirements {
    pub memory_bytes: u64,
    pub require_memory_certification: bool,
}
impl Default for ResourceRequirements {
    fn default() -> Self { Self { memory_bytes: 1024 * 1024 * 1024, require_memory_certification: false } }
}
impl ResourceRequirements {
    pub fn is_default(&self) -> bool { self == &Self::default() }
    pub fn validate(&self) -> Result<()> {
        if !(64 * 1024 * 1024..=64 * 1024 * 1024 * 1024).contains(&self.memory_bytes) { return Err("memory limit must be 64 MiB..=64 GiB".into()); }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceFeatures {
    pub available: bool,
    pub cgroup_v2: bool,
    pub memory_limit: bool,
    pub swap_limit: bool,
    pub cpu_quota: bool,
    pub pids_limit: bool,
    pub reason: Option<String>,
}
impl ResourceFeatures {
    fn from_info(info: &Value) -> Self {
        Self { available: true, cgroup_v2: info["CgroupVersion"] == "2", memory_limit: info["MemoryLimit"] == true,
            swap_limit: info["SwapLimit"] == true, cpu_quota: info["CpuCfsQuota"] == true, pids_limit: info["PidsLimit"] == true, reason: None }
    }
    pub fn require(&self) -> Result<()> {
        if !self.available || !self.cgroup_v2 || !self.memory_limit || !self.swap_limit || !self.cpu_quota || !self.pids_limit {
            return Err("required memory certification unsupported: Docker cgroup v2 memory/swap/CPU/PID enforcement unavailable".into());
        }
        // memory.peak and events availability must also be proved inside EACH job.
        Ok(())
    }
}
trait DockerApi {
    fn call(&self, args: &[String]) -> Result<String>;
}
struct Docker;
impl DockerApi for Docker {
    fn call(&self, args: &[String]) -> Result<String> {
        let out = Command::new("docker").args(args).stdin(Stdio::null()).output().map_err(|e|e.to_string())?;
        if !out.status.success() { return Err(format!("Docker command failed: {}", String::from_utf8_lossy(&out.stderr))); }
        if out.stdout.len() > 1024 * 1024 { return Err("Docker response exceeds bound".into()); }
        String::from_utf8(out.stdout).map_err(|e|e.to_string())
    }
}
fn call(api: &impl DockerApi, args: &[&str]) -> Result<String> { api.call(&args.iter().map(|s|s.to_string()).collect::<Vec<_>>()) }
pub fn detect() -> ResourceFeatures {
    match call(&Docker, &["info", "--format", "{{json .}}"])
        .and_then(|s|serde_json::from_str::<Value>(&s).map_err(|e|e.to_string())) {
        Ok(v) => ResourceFeatures::from_info(&v),
        Err(e) => ResourceFeatures { available:false,cgroup_v2:false,memory_limit:false,swap_limit:false,cpu_quota:false,pids_limit:false,reason:Some(e) },
    }
}
pub fn check_certification_request(spec: &JobSpec) -> Result<()> {
    if spec.resources.require_memory_certification {
        if !matches!(spec.probe, Probe::Scientific{..}) { return Err("required memory certification unsupported for host control probes".into()); }
        detect().require()?;
    }
    Ok(())
}

struct LifecycleLock(fs::File);
impl LifecycleLock {
    fn acquire(root: &Path) -> Result<Self> {
        let f=fs::OpenOptions::new().read(true).write(true).create(true).custom_flags(libc::O_NOFOLLOW)
            .open(root.join("container-lifecycle.lock")).map_err(|e|e.to_string())?;
        if unsafe{libc::flock(f.as_raw_fd(),libc::LOCK_EX)}!=0 { return Err("container lifecycle lock unavailable".into()); }
        Ok(Self(f))
    }
}
impl Drop for LifecycleLock { fn drop(&mut self) { unsafe{libc::flock(self.0.as_raw_fd(),libc::LOCK_UN);} } }
fn state_identity(root: &Path) -> Result<String> { Ok(hash(fs::canonicalize(root).map_err(|e|e.to_string())?.as_os_str().as_encoded_bytes())) }
fn key(project: &str, job: &str) -> String { format!("container.{}",hash(format!("{project}/{job}").as_bytes())) }
pub(super) fn record(c: &Connection, project: &str, job: &str) -> Result<Option<Value>> {
    let v: Option<String> = sql(c.query_row("SELECT value FROM scheduler WHERE key=?1",[key(project,job)],|r|r.get(0)).optional())?;
    v.map(|s|serde_json::from_str(&s).map_err(|e|e.to_string())).transpose()
}
fn persist(root: &Path, c: &Connection, value: &Value) -> Result<()> {
    let encoded=value.to_string();
    if encoded.len()>16*1024 {return Err("container receipt exceeds 16 KiB bound".into());}
    // FULL WAL transaction; includes identity and all labels together. Headroom is
    // charged even during recovery; inability to persist leaves leases held.
    enforce_state_write_admission(root,Some(c),32*1024,StorageWriteKind::EmergencyRecovery,None)?;
    sql(c.execute("INSERT INTO scheduler(key,value) VALUES(?1,?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        params![key(value["project"].as_str().ok_or("missing project")?,value["job"].as_str().ok_or("missing job")?),encoded]))?;
    Ok(())
}
pub fn inspect(root: &Path, project: &str, job: &str) -> Result<Option<Value>> { record(&connect(root)?,project,job) }

fn owner_id(root:&Path,c:&Connection)->Result<String> {
    if let Some(id)=sql(c.query_row("SELECT value FROM scheduler WHERE key='container_owner'",[],|r|r.get::<_,String>(0)).optional())?{return Ok(id);}
    enforce_state_write_admission(root,Some(c),16*1024,StorageWriteKind::EmergencyRecovery,None)?;
    let mut entropy=[0u8;32];fs::File::open("/dev/urandom").and_then(|mut f|f.read_exact(&mut entropy)).map_err(|e|e.to_string())?;
    let id=hash(&entropy);
    sql(c.execute("INSERT INTO scheduler(key,value) VALUES('container_owner',?1) ON CONFLICT(key) DO NOTHING",[&id]))?;
    sql(c.query_row("SELECT value FROM scheduler WHERE key='container_owner'",[],|r|r.get(0)))
}
fn expected_labels(root: &Path, c: &Connection, spec: &JobSpec, action: &Action) -> Result<BTreeMap<String,String>> {
    Ok(BTreeMap::from([
        (format!("{PREFIX}owner"),OWNER.into()), (format!("{PREFIX}owner-id"),sql(c.query_row("SELECT value FROM scheduler WHERE key='container_owner'",[],|r|r.get::<_,String>(0)))?), (format!("{PREFIX}state-root"),state_identity(root)?),
        (format!("{PREFIX}project"),spec.project_id.clone()),(format!("{PREFIX}job"),spec.job_id.clone()),
        (format!("{PREFIX}task"),spec.task_id.clone()),(format!("{PREFIX}lease"),action.lease_id.clone()),
        (format!("{PREFIX}generation"),spec.generation.to_string()),(format!("{PREFIX}intent"),action.intent_revision.to_string()),
        (format!("{PREFIX}spec"),hash(serde_json::to_string(spec).map_err(|e|e.to_string())?.as_bytes())),
    ]))
}
fn action_for(c: &Connection, spec: &JobSpec) -> Result<Action> {
    let data:String=sql(c.query_row("SELECT data FROM actions WHERE project=?1 AND id=?2",params![spec.project_id,format!("job.{}",spec.job_id)],|r|r.get(0)))?;
    serde_json::from_str(&data).map_err(|e|e.to_string())
}
fn validate_container(root: &Path, c: &Connection, container: &Value) -> Result<(JobSpec,Value)> {
    let labels=&container["Config"]["Labels"];
    if labels[format!("{PREFIX}owner")]!=OWNER || labels[format!("{PREFIX}state-root")]!=state_identity(root)? { return Err("foreign container ownership".into()); }
    let project=labels[format!("{PREFIX}project")].as_str().ok_or("missing container project")?;
    let job=labels[format!("{PREFIX}job")].as_str().ok_or("missing container job")?;
    valid_id(project)?;valid_id(job)?;
    let view=load_job(c,project,job)?;
    let action=action_for(c,&view.spec)?;
    let expected=expected_labels(root,c,&view.spec,&action)?;
    for (k,v) in &expected { if labels[k]!=*v { return Err(format!("container lease/epoch mismatch: {k}")); } }
    let id=container["Id"].as_str().ok_or("missing exact container ID")?;
    if id.len()!=64 || !id.bytes().all(|b|b.is_ascii_hexdigit()) {return Err("invalid exact container ID".into());}
    if let Some(old)=record(c,project,job)? { if old["id"]!=id || old["labels"]!=serde_json::to_value(&expected).unwrap() { return Err("container identity conflicts with durable receipt".into()); } }
    // A durable job/action is sufficient to adopt an orphan created just before
    // ID persistence. Unknown labels or different lease epochs are never reaped.
    Ok((view.spec,json!({"id":id,"labels":expected,"project":project,"job":job,"exit_confirmed":false})))
}
fn inspect_id(api: &impl DockerApi,id:&str)->Result<Value> {
    let raw=call(api,&["inspect",id])?;let values:Vec<Value>=serde_json::from_str(&raw).map_err(|e|e.to_string())?;
    if values.len()!=1 {return Err("ambiguous Docker identity".into());}
    Ok(values[0].clone())
}
fn reap_one(root:&Path,c:&Connection,api:&impl DockerApi,container:&Value)->Result<()> {
    let (spec,mut receipt)=validate_container(root,c,container)?;
    let id=receipt["id"].as_str().unwrap().to_string();
    if let Some(old)=record(c,&spec.project_id,&spec.job_id)? { receipt=old; }
    // Persist identity before any stop/remove operation. Exit uncertainty retains
    // both environment and execution reservations.
    persist(root,c,&receipt)?;
    if container["State"]["Running"]==true {
        let sample=call(api,&["exec",&id,"python","-c",SAMPLE]);
        receipt["pre_stop_measurement"]=match sample { Ok(s)=>serde_json::from_str(&s).unwrap_or(json!({"status":"invalid"})), Err(e)=>json!({"status":"unavailable","reason":e}) };
        receipt["measurement_complete"]=json!(false);
        persist(root,c,&receipt)?;
        call(api,&["stop","--time","2",&id])?;
    }
    let after=inspect_id(api,&id)?;
    if receipt.get("measurement").is_none() && after["State"]["Running"]==false {
        let raw=root.join("resource-receipts").join(hash(format!("{}/{}",spec.project_id,spec.job_id).as_bytes())).join("measurement.json");
        save_measurement(root,&spec,&raw)?;
        receipt=record(c,&spec.project_id,&spec.job_id)?.ok_or("container receipt lost")?;
    }
    validate_container(root,c,&after)?;
    if after["State"]["Running"]!=false || after["State"]["Status"]=="restarting" {return Err("container exit not confirmed; leases remain held".into());}
    receipt["exit_confirmed"]=json!(true);receipt["docker_state"]=after["State"].clone();
    persist(root,c,&receipt)?;
    // No force removal, and always by exact ID, after durable exit observation.
    call(api,&["rm",&id])?;
    receipt["removed"]=json!(true);persist(root,c,&receipt)?;
    let folder=root.join("resource-receipts").join(hash(format!("{}/{}",spec.project_id,spec.job_id).as_bytes()));
    if folder.exists(){safe_remove(root,&folder)?;}
    Ok(())
}
fn discover(root:&Path,c:&Connection,api:&impl DockerApi,target:Option<&JobSpec>)->Result<()> {
    let ids=api.call(&["ps".into(),"-aq".into(),"--no-trunc".into(),"--filter".into(),format!("label={PREFIX}owner={OWNER}"),"--filter".into(),format!("label={PREFIX}state-root={}",state_identity(root)?)])?;
    for id in ids.split_whitespace() {
        let container=inspect_id(api,id)?;
        if let Some(spec)=target { if container["Config"]["Labels"][format!("{PREFIX}project")]!=spec.project_id || container["Config"]["Labels"][format!("{PREFIX}job")]!=spec.job_id {continue;} }
        reap_one(root,c,api,&container)?;
    }
    // A disappeared container is safe only after an authoritative daemon listing.
    let mut q=sql(c.prepare("SELECT key,value FROM scheduler WHERE key LIKE 'container.%'"))?;
    let rows=sql(q.query_map([],|r|r.get::<_,String>(1)))?.collect::<rusqlite::Result<Vec<_>>>().map_err(|e|e.to_string())?;drop(q);
    for data in rows {let mut receipt:Value=serde_json::from_str(&data).map_err(|e|e.to_string())?;
        if let Some(spec)=target {if receipt["project"]!=spec.project_id||receipt["job"]!=spec.job_id {continue;}}
        if receipt["exit_confirmed"]!=true && !ids.split_whitespace().any(|id|receipt["id"]==id) {
            // Validate against the daemon individually too (filtered list alone
            // cannot establish absence of an ID whose labels were mutated).
            let raw=api.call(&["ps".into(),"-aq".into(),"--no-trunc".into()])?;
            if raw.split_whitespace().any(|id|receipt["id"]==id) {return Err("recorded container changed ownership; refusing settlement".into());}
            receipt["exit_confirmed"]=json!(true);receipt["measurement_complete"]=json!(false);receipt["absent_from_daemon"]=json!(true);persist(root,c,&receipt)?;
        }
    }
    // Older runners used unlabelled name-only containers. Do not infer their
    // ownership or absence from a label-filtered list during an upgrade.
    let names=call(api,&["ps","-a","--format","{{.Names}}"])?;
    let mut q=sql(c.prepare("SELECT project,id FROM jobs WHERE status IN('running','terminating','recovering') AND json_extract(spec,'$.probe.kind')='scientific'"))?;
    let rows=sql(q.query_map([],|r|Ok((r.get::<_,String>(0)?,r.get::<_,String>(1)?))))?.collect::<rusqlite::Result<Vec<_>>>().map_err(|e|e.to_string())?;
    for(project,job)in rows {
        let legacy=format!("mlr-science-{}",&hash(format!("{project}/{job}").as_bytes())[..24]);
        if names.lines().any(|name|name==legacy) {return Err("legacy unlabelled scientific container requires ownership verification; leases remain held".into());}
    }
    Ok(())
}
pub(super) fn recover(root:&Path)->Result<()> {
    let _lock=LifecycleLock::acquire(root)?;let c=connect(root)?;
    let active:bool=sql(c.query_row("SELECT EXISTS(SELECT 1 FROM jobs j JOIN environments e ON e.id=json_extract(j.spec,'$.environment_id') WHERE e.python LIKE 'docker:%' AND j.status IN('running','terminating','recovering'))",[],|r|r.get(0)))?;
    if !detect().available {
        if active {return Err("Docker unavailable; survivor discovery impossible and leases remain held".into());}
        return Ok(());
    }
    discover(root,&c,&Docker,None)
}
pub(super) fn confirm_job_exit(root:&Path,spec:&JobSpec)->Result<()> {
    if !matches!(spec.probe,Probe::Scientific{..}) {return Ok(());}
    let _lock=LifecycleLock::acquire(root)?;let c=connect(root)?;
    if record(&c,&spec.project_id,&spec.job_id)?.is_some_and(|r|r["removed"]==true&&r["exit_confirmed"]==true) {return Ok(());}
    // Handles cancellation/death before worker ID persistence as well.
    discover(root,&c,&Docker,Some(spec))?;
    if let Some(receipt)=record(&c,&spec.project_id,&spec.job_id)? {
        if receipt["exit_confirmed"]!=true {return Err("Docker exit not confirmed; refusing resource release".into());}
    }
    Ok(())
}

// Docker's daemon reports features, but the in-container namespace is the only
// authority for actual effective limits and available peak/events interfaces.
const SAMPLE: &str = r#"import json,pathlib
p=pathlib.Path('/sys/fs/cgroup')
r={}
for n in ['memory.max','memory.swap.max','memory.current','memory.peak','memory.events','cpu.max','cpu.stat','pids.max','pids.events']:
 try:r[n]=(p/n).read_text().strip()
 except OSError:r[n]=None
print(json.dumps(r))
"#;
const MONITOR: &str = include_str!("resource_monitor.py");

pub(super) fn create(root:&Path,spec:&JobSpec,cmd:&mut Command)->Result<String> {
    let _lock=LifecycleLock::acquire(root)?;
    let mut c=connect(root)?;let tx=sql(c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate))?;
    let view=load_job(&tx,&spec.project_id,&spec.job_id)?;
    let action=action_for(&tx,spec)?;
    let task:String=sql(tx.query_row("SELECT data FROM tasks WHERE project=?1 AND id=?2",params![spec.project_id,spec.task_id],|r|r.get(0)))?;
    let task:crate::core::Task=serde_json::from_str(&task).map_err(|e|e.to_string())?;
    let (state,intent):(String,u64)=sql(tx.query_row("SELECT state,intent_revision FROM projects WHERE id=?1",[&spec.project_id],|r|Ok((r.get(0)?,r.get(1)?))))?;
    if view.status!="running" || task.status!="active" || task.generation!=spec.generation || task.lease_id.as_deref()!=Some(&action.lease_id) || task.intent_revision!=intent || state=="cancelled" {return Err("stale worker fenced before container creation".into());}
    let labels=expected_labels(root,&tx,spec,&action)?;
    enforce_state_write_admission(root,Some(&tx),RECEIPT_RESERVE,StorageWriteKind::EmergencyRecovery,Some(&spec.project_id))?;
    let out=cmd.output().map_err(|e|e.to_string())?;
    if !out.status.success(){return Err(format!("Docker create failed: {}",String::from_utf8_lossy(&out.stderr)));}
    let id=String::from_utf8_lossy(&out.stdout).trim().to_string();
    if id.len()!=64||!id.bytes().all(|b|b.is_ascii_hexdigit()){return Err("Docker create did not return exact ID".into());}
    let container=inspect_id(&Docker,&id)?;
    validate_container(root,&tx,&container)?;
    persist(root,&tx,&json!({"id":id,"labels":labels,"project":spec.project_id,"job":spec.job_id,"exit_confirmed":false,"measurement_complete":false}))?;
    tx.commit().map_err(|e|e.to_string())?;Ok(id)
}
pub(super) fn labels(root:&Path,spec:&JobSpec)->Result<BTreeMap<String,String>> {let c=connect(root)?;owner_id(root,&c)?;expected_labels(root,&c,spec,&action_for(&c,spec)?)}
pub(super) fn monitor()-> &'static str {MONITOR}
fn effective_measurement(value:&Value,memory:u64,threads:u32)->bool {
    let number=|key:&str|value[key].as_str().and_then(|s|s.parse::<u64>().ok());
    let counters=|key:&str,required:&str|value[key].as_str().is_some_and(|text|text.lines().any(|line|{
        let fields=line.split_whitespace().collect::<Vec<_>>();fields.len()==2&&fields[0]==required&&fields[1].parse::<u64>().is_ok()
    }));
    let cpu=value["cpu.max"].as_str().is_some_and(|text|{
        let fields=text.split_whitespace().collect::<Vec<_>>();
        if fields.len()!=2{return false;}match (fields[0].parse::<u64>(),fields[1].parse::<u64>()) {
            (Ok(quota),Ok(period))=>period>0&&period.checked_mul(threads as u64)==Some(quota),_=>false,
        }
    });
    number("memory.max")==Some(memory)&&number("memory.swap.max")==Some(0)&&number("memory.peak").is_some()
        &&number("memory.current").is_some()&&number("pids.max")==Some(64)&&cpu
        &&counters("memory.events","oom")&&counters("memory.events","oom_kill")
        &&counters("cpu.stat","usage_usec")&&counters("pids.events","max")
}
pub(super) fn save_measurement(root:&Path,spec:&JobSpec,path:&Path)->Result<()> {
    let mut c=connect(root)?;let tx=sql(c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate))?;
    let mut receipt=record(&tx,&spec.project_id,&spec.job_id)?.ok_or("missing container identity receipt")?;
    let measurement=if path.exists() {
        let meta=fs::symlink_metadata(path).map_err(|e|e.to_string())?;
        if !meta.is_file()||meta.file_type().is_symlink()||meta.len()>8192{return Err("unsafe or oversized cgroup receipt".into());}
        let mut file=fs::OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW).open(path).map_err(|e|e.to_string())?;
        let mut bytes=Vec::new();Read::by_ref(&mut file).take(8193).read_to_end(&mut bytes).map_err(|e|e.to_string())?;
        if bytes.len()>8192{return Err("cgroup receipt grew beyond bound".into());}
        serde_json::from_slice::<Value>(&bytes).map_err(|e|e.to_string())?
    } else {json!({"status":"missing"})};
    receipt["measurement_complete"]=json!(measurement["complete"]==true && measurement["supported"]==true && measurement["memory_bytes"]==spec.resources.memory_bytes && measurement["threads"]==spec.threads
        && measurement["exit_code"].as_i64().is_some()
        && effective_measurement(&measurement["before"],spec.resources.memory_bytes,spec.threads)
        && effective_measurement(&measurement["after"],spec.resources.memory_bytes,spec.threads));
    receipt["measurement"]=measurement;persist(root,&tx,&receipt)?;tx.commit().map_err(|e|e.to_string())?;Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use crate::core::{Operation,Request};
    struct FakeDocker { containers:RefCell<Vec<Value>>, calls:RefCell<Vec<Vec<String>>>, fail_stop:bool, fail_remove:bool, stay_running:bool }
    impl DockerApi for FakeDocker {
        fn call(&self,args:&[String])->Result<String> {
            self.calls.borrow_mut().push(args.to_vec());
            match args[0].as_str() {
                "ps" if args.iter().any(|a|a=="{{.Names}}")=>Ok(String::new()),
                "ps"=>Ok(self.containers.borrow().iter().map(|v|v["Id"].as_str().unwrap()).collect::<Vec<_>>().join("\n")),
                "inspect"=>Ok(json!([self.containers.borrow().iter().find(|v|v["Id"]==args[1]).ok_or("not found")?]).to_string()),
                "exec"=>Ok(json!({"memory.peak":"2097152","memory.events":"oom 0\noom_kill 0"}).to_string()),
                "stop"=>{if self.fail_stop{return Err("injected stop failure".into());}for v in self.containers.borrow_mut().iter_mut(){if !self.stay_running&&v["Id"]==*args.last().unwrap(){v["State"]=json!({"Running":false,"Status":"exited","ExitCode":137});}}Ok(String::new())},
                "rm"=>{if self.fail_remove{return Err("injected removal failure".into());}self.containers.borrow_mut().retain(|v|v["Id"]!=args[1]);Ok(String::new())},
                _=>Err(format!("unexpected command {args:?}")),
            }
        }
    }
    fn fixture()->(tempfile::TempDir,PathBuf,JobSpec) {
        let temp=tempfile::tempdir().unwrap();let root=temp.path().join("state");
        let policy=Policy{budget:BTreeMap::from([("wall_ms".into(),600000)]),capacity:BTreeMap::from([("workers".into(),2)]),max_tasks:128};
        let rt=crate::Runtime::init(&root,policy.clone()).unwrap();migrate(&root).unwrap();
        for (id,op) in [
            ("create",Operation::CreateProject{intent:"resource faults".into(),finalization_id:"f".into(),policy}),
            ("branch",Operation::CreateBranch{branch_id:"b".into(),base_hash:"base".into(),code_hash:"code".into()}),
            ("task",Operation::CreateTask{task_id:"t".into(),branch_id:"b".into(),context:crate::contracts::Context{role_id:"lead".into(),role_version:"v1".into(),prompt_hash:"prompt".into(),model:None,phase:"search".into(),hypothesis:None,base_hash:"base".into(),code_hash:"code".into(),evidence:vec![],artifacts:vec![],pending_questions:vec![],calls:vec![],jobs:vec![],next_action:"search".into(),stop_conditions:vec![]}}),
            ("lease",Operation::Claim{task_id:"t".into(),owner:"test".into(),lease_id:"lease".into(),now_ms:now_ms(),expires_ms:now_ms()+600000,expected_generation:1}),
        ] {assert_eq!(rt.command(Request{command_id:id.into(),project_id:"p".into(),expected_revision:None,operation:op}).unwrap().status,"applied");}
        let spec=JobSpec{job_id:"j".into(),project_id:"p".into(),task_id:"t".into(),generation:2,timeout_ms:10000,threads:1,max_output_bytes:4096,environment_id:None,resources:Default::default(),probe:Probe::Sleep{millis:10000}};
        submit(&root,spec.clone()).unwrap();
        let c=connect(&root).unwrap();let tx=c.unchecked_transaction().unwrap();
        let task:String=tx.query_row("SELECT data FROM tasks",[],|r|r.get(0)).unwrap();let task:crate::core::Task=serde_json::from_str(&task).unwrap();
        let action=Action{action_id:"job.j".into(),task_id:"t".into(),generation:2,intent_revision:task.intent_revision,lease_id:"lease".into(),input:serde_json::to_string(&spec).unwrap(),budget:BTreeMap::from([("wall_ms".into(),10000)]),capacity:BTreeMap::from([("workers".into(),1)]),deadline_ms:now_ms()+10000,status:"dispatched".into(),attempt_id:Some("j.1".into()),external_request_id:Some("j".into()),response:None,receipt:None,uncertainty:None};
        tx.execute("INSERT INTO actions(project,id,task,data) VALUES('p','job.j','t',?1)",[serde_json::to_string(&action).unwrap()]).unwrap();
        tx.execute("INSERT INTO reservations(project,action,budget,capacity,held) VALUES('p','job.j',?1,?2,1)",params![serde_json::to_string(&action.budget).unwrap(),serde_json::to_string(&action.capacity).unwrap()]).unwrap();
        tx.execute("UPDATE jobs SET status='running',action='job.j',started_ms=?1",[now_ms()]).unwrap();tx.commit().unwrap();
        (temp,root,spec)
    }
    fn fake(root:&Path,spec:&JobSpec)->FakeDocker {
        let labels=labels(root,spec).unwrap();
        FakeDocker{containers:RefCell::new(vec![json!({"Id":"a".repeat(64),"Config":{"Labels":labels},"State":{"Running":true,"Status":"running"}})]),calls:RefCell::new(vec![]),fail_stop:false,fail_remove:false,stay_running:false}
    }
    #[test]
    fn feature_detection_and_required_certification_refuse_without_dispatch() {
        for info in [json!({}),json!({"CgroupVersion":"1","MemoryLimit":true,"SwapLimit":true,"CpuCfsQuota":true,"PidsLimit":true})] {
            assert!(ResourceFeatures::from_info(&info).require().is_err());
        }
        let good=json!({"CgroupVersion":"2","MemoryLimit":true,"SwapLimit":true,"CpuCfsQuota":true,"PidsLimit":true});
        assert!(ResourceFeatures::from_info(&good).require().is_ok());
        let (_temp,root,mut spec)=fixture();spec.job_id="refused".into();spec.resources.require_memory_certification=true;
        assert!(submit(&root,spec).unwrap_err().contains("unsupported for host"));
        let c=connect(&root).unwrap();assert_eq!(c.query_row("SELECT count(*) FROM jobs WHERE id='refused'",[],|r|r.get::<_,i64>(0)).unwrap(),0);
        assert_eq!(c.query_row("PRAGMA user_version",[],|r|r.get::<_,i64>(0)).unwrap(),9);
    }
    #[test]
    fn survivor_discovery_adopts_missing_identity_then_reaps_exact_id() {
        let (_temp,root,spec)=fixture();let c=connect(&root).unwrap();let api=fake(&root,&spec);
        discover(&root,&c,&api,None).unwrap();
        let receipt=inspect(&root,"p","j").unwrap().unwrap();
        assert_eq!(receipt["id"],"a".repeat(64));assert_eq!(receipt["exit_confirmed"],true);assert_eq!(receipt["removed"],true);
        assert_eq!(receipt["measurement_complete"],false);assert_eq!(receipt["pre_stop_measurement"]["memory.peak"],"2097152");
        let calls=api.calls.borrow();let stop=calls.iter().position(|c|c[0]=="stop").unwrap();let rm=calls.iter().position(|c|c[0]=="rm").unwrap();
        assert!(stop<rm);assert_eq!(calls[rm],vec!["rm".into(),"a".repeat(64)]);drop(calls);
        discover(&root,&c,&api,None).unwrap(); // repeated restart is idempotent
        assert_eq!(api.calls.borrow().iter().filter(|c|c[0]=="stop").count(),1);
        assert_eq!(c.query_row("SELECT held FROM reservations",[],|r|r.get::<_,i64>(0)).unwrap(),1); // lifecycle never settles early
    }
    #[test]
    fn survivor_ownership_epoch_and_identity_conflicts_are_never_reaped() {
        for label in ["owner","owner-id","state-root","project","lease","generation","intent","spec"] {
            let (_temp,root,spec)=fixture();let api=fake(&root,&spec);api.containers.borrow_mut()[0]["Config"]["Labels"][format!("{PREFIX}{label}")]=json!("foreign");
            assert!(discover(&root,&connect(&root).unwrap(),&api,None).is_err());
            assert!(!api.calls.borrow().iter().any(|c|c[0]=="stop"||c[0]=="rm"));
        }
        let (_temp,root,spec)=fixture();let c=connect(&root).unwrap();let api=fake(&root,&spec);
        let (_,mut receipt)=validate_container(&root,&c,&api.containers.borrow()[0]).unwrap();receipt["id"]=json!("b".repeat(64));persist(&root,&c,&receipt).unwrap();
        assert!(discover(&root,&c,&api,None).unwrap_err().contains("conflicts"));
    }
    #[test]
    fn failed_stop_or_remove_keeps_receipts_and_reservations_for_restart() {
        for fail_stop in [true,false] {
            let (_temp,root,spec)=fixture();let c=connect(&root).unwrap();let mut api=fake(&root,&spec);api.fail_stop=fail_stop;api.fail_remove=!fail_stop;
            assert!(discover(&root,&c,&api,None).is_err());
            let receipt=inspect(&root,"p","j").unwrap().unwrap();assert_eq!(receipt["exit_confirmed"]==true,!fail_stop);
            assert_eq!(c.query_row("SELECT held FROM reservations",[],|r|r.get::<_,i64>(0)).unwrap(),1);
            api.fail_stop=false;api.fail_remove=false;discover(&root,&c,&api,None).unwrap();assert_eq!(inspect(&root,"p","j").unwrap().unwrap()["removed"],true);
        }
    }
    #[test]
    fn completing_one_job_does_not_stop_other_live_workers() {
        let (_temp,root,spec)=fixture();let c=connect(&root).unwrap();let api=fake(&root,&spec);let mut foreign=api.containers.borrow()[0].clone();foreign["Id"]=json!("b".repeat(64));foreign["Config"]["Labels"][format!("{PREFIX}job")]=json!("other");api.containers.borrow_mut().push(foreign);
        discover(&root,&c,&api,Some(&spec)).unwrap();assert_eq!(api.containers.borrow().len(),1);assert_eq!(api.containers.borrow()[0]["State"]["Running"],true);
    }
    #[test]
    fn missing_measurement_and_state_ceiling_do_not_claim_certification() {
        let (_temp,root,spec)=fixture();let c=connect(&root).unwrap();let api=fake(&root,&spec);let (_,receipt)=validate_container(&root,&c,&api.containers.borrow()[0]).unwrap();persist(&root,&c,&receipt).unwrap();
        save_measurement(&root,&spec,&root.join("missing")).unwrap();assert_eq!(inspect(&root,"p","j").unwrap().unwrap()["measurement_complete"],false);
        fs::write(root.join("storage_budgets.json"),json!({"max_state_total_bytes":1}).to_string()).unwrap();
        assert!(discover(&root,&c,&api,None).is_err());assert!(!api.calls.borrow().iter().any(|c|c[0]=="stop"));
    }
    #[test]
    #[ignore="requires explicit locally installed digest-pinned Docker image"]
    fn live_cgroup_receipt_and_daemon_survivor_restart() {
        let image=std::env::var("MLRESCURIX_BUNDLE_IMAGE").unwrap();assert!(image.contains("@sha256:"));
        detect().require().unwrap();
        for crash in [false,true] {
            let (_temp,root,spec)=fixture();let folder=root.join("resource-receipts").join(hash(b"p/j"));ensure_dir(&root,&folder).unwrap();
            let mut cmd=Command::new("docker");cmd.args(["create","--pull=never","--network","none","--read-only","--cap-drop","ALL","--security-opt","no-new-privileges:true","--memory","1073741824","--memory-swap","1073741824","--cpus","1","--pids-limit","64","-v"]).arg(format!("{}:/resource:rw",folder.display()));
            for(k,v)in labels(&root,&spec).unwrap(){cmd.arg("--label").arg(format!("{k}={v}"));}
            cmd.args([image.as_str(),"python","-c",monitor(),"1073741824","1","true","python","-c",if crash{"import time; time.sleep(60)"}else{"x=bytearray(32*1024*1024)"}]);
            let id=create(&root,&spec,&mut cmd).unwrap();
            if crash {
                call(&Docker,&["start",&id]).unwrap();
                std::thread::sleep(Duration::from_millis(600));
                // Durable state reopen + daemon discovery (no worker Child/name).
                // Crash after Docker effect but before durable ID result.
                connect(&root).unwrap().execute("DELETE FROM scheduler WHERE key=?1",[key("p","j")]).unwrap();
                let mut supervisor=Supervisor::start(&root,1).unwrap();
                connect(&root).unwrap().execute("UPDATE jobs SET recovered_deadline_ms=0 WHERE id='j'",[]).unwrap();
                supervisor.tick().unwrap();drop(supervisor);
                let c=connect(&root).unwrap();
                assert_eq!(c.query_row("SELECT held FROM reservations",[],|r|r.get::<_,i64>(0)).unwrap(),0);
                assert_eq!(c.query_row("SELECT count(*) FROM charges",[],|r|r.get::<_,i64>(0)).unwrap(),1);
                drop(Supervisor::start(&root,1).unwrap());
                let r=inspect(&root,"p","j").unwrap().unwrap();assert_eq!(r["exit_confirmed"],true);assert_eq!(r["measurement_complete"],false);
            } else {
                call(&Docker,&["start","-a",&id]).unwrap();save_measurement(&root,&spec,&folder.join("measurement.json")).unwrap();
                let r=inspect(&root,"p","j").unwrap().unwrap();assert_eq!(r["measurement_complete"],true,"{r}");
                assert!(r["measurement"]["after"]["memory.peak"].as_str().unwrap().parse::<u64>().unwrap()>=32*1024*1024);
                recover(&root).unwrap();
            }
            let r=inspect(&root,"p","j").unwrap().unwrap();assert_eq!(r["removed"],true);
            if let Ok(dir)=std::env::var("MLRESCURIX_EVIDENCE_DIR") {fs::write(Path::new(&dir).join(if crash{"live-survivor.json"}else{"live-cgroup.json"}),serde_json::to_vec_pretty(&r).unwrap()).unwrap();}
        }
    }
    #[test]
    #[ignore="requires explicit locally installed digest-pinned Docker image"]
    fn live_cgroup_oom_is_enforced_and_event_is_received_before_teardown() {
        let image=std::env::var("MLRESCURIX_BUNDLE_IMAGE").unwrap();assert!(image.contains("@sha256:"));detect().require().unwrap();
        let (_temp,root,mut spec)=fixture();spec.resources.memory_bytes=64*1024*1024;
        let c=connect(&root).unwrap();let mut action=action_for(&c,&spec).unwrap();action.input=serde_json::to_string(&spec).unwrap();
        c.execute("UPDATE jobs SET spec=?1 WHERE id='j'",[&action.input]).unwrap();save_action(&root,"p",&action).unwrap();
        let folder=root.join("resource-receipts").join(hash(b"p/j"));ensure_dir(&root,&folder).unwrap();
        let mut cmd=Command::new("docker");cmd.args(["create","--pull=never","--network","none","--read-only","--cap-drop","ALL","--security-opt","no-new-privileges:true","--memory","67108864","--memory-swap","67108864","--cpus","1","--pids-limit","64","-v"]).arg(format!("{}:/resource:rw",folder.display()));
        for(k,v)in labels(&root,&spec).unwrap(){cmd.arg("--label").arg(format!("{k}={v}"));}
        cmd.args([image.as_str(),"python","-c",monitor(),"67108864","1","true","python","-c","x=bytearray(128*1024*1024)"]);
        let id=create(&root,&spec,&mut cmd).unwrap();
        let status=Command::new("docker").args(["start","-a",&id]).status().unwrap();assert!(!status.success());
        save_measurement(&root,&spec,&folder.join("measurement.json")).unwrap();
        let r=inspect(&root,"p","j").unwrap().unwrap();assert_eq!(r["measurement_complete"],true,"{r}");
        let events=r["measurement"]["after"]["memory.events"].as_str().unwrap();
        assert!(events.lines().any(|line|line.split_whitespace().collect::<Vec<_>>()==vec!["oom_kill","1"]),"{events}");
        recover(&root).unwrap();let r=inspect(&root,"p","j").unwrap().unwrap();
        if let Ok(dir)=std::env::var("MLRESCURIX_EVIDENCE_DIR") {fs::write(Path::new(&dir).join("live-oom.json"),serde_json::to_vec_pretty(&r).unwrap()).unwrap();}
    }

    #[test]
    fn stop_acknowledgement_without_exit_cannot_release_or_remove() {
        let (_temp,root,spec)=fixture();let c=connect(&root).unwrap();let mut api=fake(&root,&spec);api.stay_running=true;
        assert!(discover(&root,&c,&api,None).unwrap_err().contains("exit not confirmed"));
        assert!(!api.calls.borrow().iter().any(|c|c[0]=="rm"));
        assert_eq!(inspect(&root,"p","j").unwrap().unwrap()["exit_confirmed"],false);
        assert_eq!(c.query_row("SELECT held FROM reservations",[],|r|r.get::<_,i64>(0)).unwrap(),1);
    }
    #[test]
    fn claimed_complete_measurement_without_effective_limits_is_not_certified() {
        let (_temp,root,spec)=fixture();let c=connect(&root).unwrap();let api=fake(&root,&spec);let(_,r)=validate_container(&root,&c,&api.containers.borrow()[0]).unwrap();persist(&root,&c,&r).unwrap();
        let file=root.join("claimed.json");fs::write(&file,json!({"complete":true,"supported":true,"memory_bytes":spec.resources.memory_bytes,"threads":1,"exit_code":0}).to_string()).unwrap();
        save_measurement(&root,&spec,&file).unwrap();assert_eq!(inspect(&root,"p","j").unwrap().unwrap()["measurement_complete"],false);
    }

}
