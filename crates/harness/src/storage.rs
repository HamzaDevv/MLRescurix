use crate::{Result,contracts::*,core::*};
use rusqlite::{Connection,Transaction,TransactionBehavior,params,OptionalExtension};
use serde::de::DeserializeOwned;
use serde_json::{Value,json};
use sha2::{Digest,Sha256};
use std::{path::Path,sync::mpsc,thread};
const QUEUE:usize=64;
const PAGE:usize=256;
const MARKER:&str="mlrescurix-owned-v1\n";
type Reply=mpsc::Sender<Result<Value>>;
enum Job { Command(Request,Reply), Snapshot(String,Reply), Events(String,u64,Reply), Restore(String,String,Reply) }
#[derive(Clone)]
pub struct Runtime { sender:mpsc::SyncSender<Job> }
fn ser<T:serde::Serialize>(v:&T)->Result<String>{serde_json::to_string(v).map_err(|e|e.to_string())}
fn de<T:DeserializeOwned>(s:&str)->Result<T>{serde_json::from_str(s).map_err(|e|e.to_string())}
fn sql<T>(r:rusqlite::Result<T>)->Result<T>{r.map_err(|e|e.to_string())}
fn hash(s:&str)->String {format!("{:x}",Sha256::digest(s.as_bytes()))}
impl Runtime {
    pub fn init(path:&Path,policy:Policy)->Result<Self>{
        validate_policy(&policy)?;
        // Never adopt an existing directory or state, even if empty.
        std::fs::create_dir(path).map_err(|e|e.to_string())?;
        let mut c=connect(path)?;
        let tx=sql(c.transaction_with_behavior(TransactionBehavior::Immediate))?;
        sql(tx.execute_batch(include_str!("schema.sql")))?;
        sql(tx.execute("INSERT INTO device VALUES(1,?1)",[ser(&policy)?]))?;
        sql(tx.commit())?;
        std::fs::write(path.join("OWNED"),MARKER).map_err(|e|e.to_string())?;
        Self::spawn(c)
    }
    pub fn open(path:&Path)->Result<Self>{
        if std::fs::symlink_metadata(path).map_err(|e|e.to_string())?.file_type().is_symlink() {return Err("state root cannot be a symlink".into());}
        if std::fs::read_to_string(path.join("OWNED")).map_err(|e|e.to_string())?!=MARKER {return Err("unowned state".into());}
        for name in ["state.sqlite","state.sqlite-wal","state.sqlite-shm","OWNED"] {
            if let Ok(m)=std::fs::symlink_metadata(path.join(name)) {if m.file_type().is_symlink(){return Err("state file symlink rejected".into());}}
        }
        let c=connect(path)?;
        let version:i64=sql(c.query_row("PRAGMA user_version",[],|r|r.get(0)))?;
        if !(1..=9).contains(&version) {return Err(format!("unsupported database migration version {version}"));}
        let integrity:String=sql(c.query_row("PRAGMA quick_check",[],|r|r.get(0)))?;
        if integrity!="ok" {return Err("database integrity check failed".into());}
        Self::spawn(c)
    }
    fn spawn(mut c:Connection)->Result<Self>{
        let(sender,receiver)=mpsc::sync_channel(QUEUE);
        thread::Builder::new().name("mlrescurix-writer".into()).spawn(move||{
            for job in receiver {
                let(r,reply)=match job {
                    Job::Command(req,reply)=>(execute(&mut c,req).and_then(|v|serde_json::to_value(v).map_err(|e|e.to_string())),reply),
                    Job::Snapshot(p,reply)=>(snapshot(&mut c,&p),reply),
                    Job::Events(p,after,reply)=>(events(&c,&p,after).and_then(|e|serde_json::to_value(e).map_err(|e|e.to_string())),reply),
                    Job::Restore(p,t,reply)=>(restore(&mut c,&p,&t).and_then(|v|serde_json::to_value(v).map_err(|e|e.to_string())),reply),
                }; let _=reply.send(r);
            }
        }).map_err(|e|e.to_string())?;
        Ok(Self{sender})
    }
    fn call(&self,make:impl FnOnce(Reply)->Job)->Result<Value>{
        let(s,r)=mpsc::channel();self.sender.try_send(make(s)).map_err(|e|format!("writer unavailable/backpressure: {e}"))?;
        r.recv().map_err(|e|e.to_string())?
    }
    pub fn command(&self,req:Request)->Result<Receipt>{
        if ser(&req)?.len()>MAX_FRAME{return Err("command frame exceeds bound".into());}
        serde_json::from_value(self.call(|r|Job::Command(req,r))?).map_err(|e|e.to_string())
    }
    pub fn snapshot(&self,p:&str)->Result<Value>{self.call(|r|Job::Snapshot(p.into(),r))}
    pub fn events(&self,p:&str,after:u64)->Result<Vec<Event>>{serde_json::from_value(self.call(|r|Job::Events(p.into(),after,r))?).map_err(|e|e.to_string())}
    pub fn restore(&self,p:&str,t:&str)->Result<RecoveryTask>{serde_json::from_value(self.call(|r|Job::Restore(p.into(),t.into(),r))?).map_err(|e|e.to_string())}
}
fn connect(path:&Path)->Result<Connection>{
    let c=sql(Connection::open(path.join("state.sqlite")))?;
    sql(c.busy_timeout(std::time::Duration::from_secs(2)))?;
    sql(c.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA foreign_keys=ON; PRAGMA wal_autocheckpoint=256;"))?;Ok(c)
}
fn validate_policy(p:&Policy)->Result<()>{amounts(&p.budget)?;amounts(&p.capacity)?;if p.max_tasks==0 || p.max_tasks>1024 {return Err("invalid task ceiling".into());}Ok(())}
fn event(c:&Connection,p:&str,kind:&str,payload:Value)->Result<()> {sql(c.execute("INSERT INTO events(project,kind,payload) VALUES(?1,?2,?3)",params![p,kind,ser(&payload)?]))?;Ok(())}
fn revision(c:&Connection,p:&str)->Result<u64>{sql(c.query_row("SELECT revision FROM projects WHERE id=?1",[p],|r|r.get(0)))}
fn intent_rev(c:&Connection,p:&str)->Result<u64>{sql(c.query_row("SELECT intent_revision FROM projects WHERE id=?1",[p],|r|r.get(0)))}
fn task(c:&Connection,p:&str,t:&str)->Result<Task>{de(&sql(c.query_row("SELECT data FROM tasks WHERE project=?1 AND id=?2",params![p,t],|r|r.get::<_,String>(0)))?)}
fn action(c:&Connection,p:&str,a:&str)->Result<Action>{de(&sql(c.query_row("SELECT data FROM actions WHERE project=?1 AND id=?2",params![p,a],|r|r.get::<_,String>(0)))?)}
fn save_task(c:&Connection,p:&str,t:&Task)->Result<()> {sql(c.execute("UPDATE tasks SET data=?3 WHERE project=?1 AND id=?2",params![p,t.task_id,ser(t)?]))?;Ok(())}
fn save_action(c:&Connection,p:&str,a:&Action)->Result<()> {sql(c.execute("UPDATE actions SET data=?3 WHERE project=?1 AND id=?2",params![p,a.action_id,ser(a)?]))?;Ok(())}
fn active(c:&Connection,p:&str)->Result<()> {let s:String=sql(c.query_row("SELECT state FROM projects WHERE id=?1",[p],|r|r.get(0)))?;if s!="created" {return Err(format!("admission unavailable in state {s}"));}Ok(())}
fn fence(c:&Connection,p:&str,t:&str,g:u64,lease:&str,now:Option<i64>)->Result<Task>{
    let t=task(c,p,t)?;
    if t.generation!=g || t.lease_id.as_deref()!=Some(lease) || t.intent_revision!=intent_rev(c,p)? || t.status!="active" {return Err("stale task/intent/lease fence".into());}
    if let Some(now)=now {if t.expires_ms.is_none_or(|end|end<=now) {return Err("expired lease".into());}}
    Ok(t)
}
fn rows<T:DeserializeOwned>(c:&Connection,table:&str,p:&str)->Result<Vec<T>>{
    let mut s=sql(c.prepare(&format!("SELECT data FROM {table} WHERE project=?1 ORDER BY id")))?;
    let rs=sql(s.query_map([p],|r|r.get::<_,String>(0)))?;
    rs.map(|r|de(&sql(r)?)).collect()
}
fn execute(c:&mut Connection,req:Request)->Result<Receipt>{
    id(&req.command_id)?;id(&req.project_id)?;let encoded=ser(&req)?;
    let tx=sql(c.transaction_with_behavior(TransactionBehavior::Immediate))?;
    let old:Option<(String,String)>=sql(tx.query_row("SELECT request,receipt FROM commands WHERE id=?1 AND project=?2",params![&req.command_id,&req.project_id],|r|Ok((r.get(0)?,r.get(1)?))).optional())?;
    if let Some((request,receipt))=old {if request!=encoded{return Err("command identity reused with different payload/scope".into());}return de(&receipt);}
    // Isolate rejected operation effects while retaining a durable rejection receipt.
    sql(tx.execute_batch("SAVEPOINT operation"))?;
    let result=(||{
        if let Some(expected)=req.expected_revision {if revision(&tx,&req.project_id)?!=expected {return Err("revision conflict".into());}}
        apply(&tx,&req.project_id,&req.operation)
    })();
    let(status,result)=match result {
        Ok(v)=>{sql(tx.execute_batch("RELEASE operation"))?;("applied",v)},
        Err(e)=>{sql(tx.execute_batch("ROLLBACK TO operation; RELEASE operation"))?;("rejected",json!({"reason":e}))},
    };
    // Creation failures have no valid project to attach a receipt to.
    if revision(&tx,&req.project_id).is_err(){return Err(result.to_string());}
    if status=="applied" {sql(tx.execute("UPDATE projects SET revision=revision+1 WHERE id=?1",[&req.project_id]))?;}
    let receipt=Receipt{command_id:req.command_id.clone(),project_id:req.project_id.clone(),status:status.into(),revision:revision(&tx,&req.project_id)?,result};
    sql(tx.execute("INSERT INTO commands VALUES(?1,?2,?3,?4)",params![req.command_id,req.project_id,encoded,ser(&receipt)?]))?;
    event(&tx,&req.project_id,"command.receipt",json!(receipt))?;
    sql(tx.commit())?;tracing::debug!(command=%receipt.command_id,status=%receipt.status,"durable receipt");Ok(receipt)
}
fn apply(c:&Transaction<'_>,p:&str,op:&Operation)->Result<Value>{
    match op {
        Operation::CreateProject{intent,finalization_id,policy}=>{
            text(intent)?;id(finalization_id)?;validate_policy(policy)?;
            sql(c.execute("INSERT INTO projects VALUES(?1,0,1,1,?2,?3,'created',?4)",params![p,intent,finalization_id,ser(policy)?]))?;
            event(c,p,"project.created",json!({"intent":intent,"finalization_id":finalization_id,"required_outcomes":["code","report_card","inference"]}))?;
        }
        Operation::Lifecycle{state}=>{
            if !["created","paused","cancelled"].contains(&state.as_str()) {return Err("M0 cannot publish official scores or finalize models".into());}
            let current:String=sql(c.query_row("SELECT state FROM projects WHERE id=?1",[p],|r|r.get(0)))?;
            if current=="cancelled" {return Err("cancelled project is terminal".into());}
            if state=="created" && current!="paused" {return Err("resume requires paused project".into());}
            if state=="cancelled" {
                sql(c.execute("UPDATE projects SET state=?2,intent_revision=intent_revision+1 WHERE id=?1",params![p,state]))?;
                invalidate_tasks(c,p,"cancelled")?;
            } else {sql(c.execute("UPDATE projects SET state=?2 WHERE id=?1",params![p,state]))?;}
            event(c,p,"project.state_changed",json!({"state":state,"execution_stopped":false,"active_execution":"drain"}))?;
        }
        Operation::Steer{intent}=>{
            active(c,p)?;text(intent)?;
            sql(c.execute("UPDATE projects SET intent=?2,intent_revision=intent_revision+1 WHERE id=?1",params![p,intent]))?;
            invalidate_tasks(c,p,"ready")?;event(c,p,"intent.changed",json!({"intent":intent}))?;
        }
        Operation::CreateBranch{branch_id,base_hash,code_hash}=>{
            active(c,p)?;for s in [branch_id,base_hash,code_hash]{id(s)?;}
            let n:i64=sql(c.query_row("SELECT count(*) FROM branches WHERE project=?1",[p],|r|r.get(0)))?;
            if n>=1024{return Err("branch ceiling".into());}
            sql(c.execute("INSERT INTO branches VALUES(?1,?2,?3,?4)",params![p,branch_id,base_hash,code_hash]))?;
            event(c,p,"branch.created",json!({"branch_id":branch_id}))?;
        }
        Operation::CreateTask{task_id,branch_id,context}=>{
            active(c,p)?;id(task_id)?;id(branch_id)?;context.validate()?;
            let policy:Policy=de(&sql(c.query_row("SELECT policy FROM projects WHERE id=?1",[p],|r|r.get::<_,String>(0)))?)?;
            let device:Policy=de(&sql(c.query_row("SELECT policy FROM device",[],|r|r.get::<_,String>(0)))?)?;
            let local:i64=sql(c.query_row("SELECT count(*) FROM tasks WHERE project=?1",[p],|r|r.get(0)))?;
            let global:i64=sql(c.query_row("SELECT count(*) FROM tasks",[],|r|r.get(0)))?;
            if local>=policy.max_tasks as i64 || global>=device.max_tasks as i64 {return Err("task ceiling".into());}
            let draft=Task{task_id:task_id.clone(),branch_id:branch_id.clone(),generation:1,intent_revision:intent_rev(c,p)?,owner:None,lease_id:None,expires_ms:None,status:"ready".into(),context:context.clone()};
            validate_context_refs(c,p,context,&draft)?;
            let t=Task{task_id:task_id.clone(),branch_id:branch_id.clone(),generation:1,intent_revision:intent_rev(c,p)?,owner:None,lease_id:None,expires_ms:None,status:"ready".into(),context:context.clone()};
            sql(c.execute("INSERT INTO tasks VALUES(?1,?2,?3,?4)",params![p,task_id,branch_id,ser(&t)?]))?;
            boundary(c,p,&t,"task_created")?;
        }
        Operation::Claim{task_id,owner,lease_id,now_ms,expires_ms,expected_generation}=>{
            active(c,p)?;id(owner)?;id(lease_id)?;valid_time(*now_ms)?;valid_time(*expires_ms)?;
            if expires_ms<=now_ms {return Err("lease expiry must follow claim time".into());}
            let mut t=task(c,p,task_id)?;
            if t.status=="cancelled" || t.generation!=*expected_generation || t.expires_ms.is_some_and(|end|end>*now_ms) {return Err("ownership conflict".into());}
            t.generation=next(t.generation)?;t.intent_revision=intent_rev(c,p)?;t.owner=Some(owner.clone());t.lease_id=Some(lease_id.clone());t.expires_ms=Some(*expires_ms);t.status="active".into();
            save_task(c,p,&t)?;boundary(c,p,&t,"ownership_acquired")?;return Ok(json!({"generation":t.generation,"lease_id":lease_id}));
        }
        Operation::Prepare{task_id,generation,lease_id,action_id,input,budget,capacity,deadline_ms,now_ms}=>{
            active(c,p)?;id(action_id)?;text(input)?;amounts(budget)?;amounts(capacity)?;valid_time(*deadline_ms)?;valid_time(*now_ms)?;
            let t=fence(c,p,task_id,*generation,lease_id,Some(*now_ms))?;
            if deadline_ms<=now_ms {return Err("deadline must follow admission".into());}
            if !pending_questions(c,p)?.is_empty(){return Err("pending human answer blocks dispatch admission".into());}
            if let Ok(a)=action(c,p,action_id) {
                if a.task_id!=*task_id || a.input!=*input || a.budget!=*budget || a.capacity!=*capacity || a.deadline_ms!=*deadline_ms {return Err("action identity payload conflict".into());}
                return Ok(json!(a));
            }
            if rows::<Action>(c,"actions",p)?.iter().any(|a|a.task_id==*task_id && ["dispatched","outcome_unknown","responded"].contains(&a.status.as_str())) {return Err("prior action requires receipt reconciliation before retry".into());}
            let pending=rows::<Action>(c,"actions",p)?.into_iter().filter(|a|!["succeeded","failed","cancelled"].contains(&a.status.as_str())).count();
            if pending>=1024 {return Err("pending action ceiling".into());}
            admit(c,p,budget,capacity)?;
            let a=Action{action_id:action_id.clone(),task_id:task_id.clone(),generation:*generation,intent_revision:t.intent_revision,lease_id:lease_id.clone(),input:input.clone(),budget:budget.clone(),capacity:capacity.clone(),deadline_ms:*deadline_ms,status:"prepared".into(),attempt_id:None,external_request_id:None,response:None,receipt:None,uncertainty:None};
            sql(c.execute("INSERT INTO actions VALUES(?1,?2,?3,?4)",params![p,action_id,task_id,ser(&a)?]))?;
            sql(c.execute("INSERT INTO reservations VALUES(?1,?2,?3,?4,1)",params![p,action_id,ser(budget)?,ser(capacity)?]))?;
            boundary(c,p,&t,"intent_prepared")?;return Ok(json!(a));
        }
        Operation::Dispatch{task_id,generation,lease_id,action_id,attempt_id,external_request_id,now_ms}=>{
            active(c,p)?;valid_time(*now_ms)?;id(attempt_id)?;if let Some(s)=external_request_id{id(s)?;}
            let t=fence(c,p,task_id,*generation,lease_id,Some(*now_ms))?;
            if !pending_questions(c,p)?.is_empty(){return Err("pending approval is not consent".into());}
            let mut a=owned_action(c,p,action_id,&t)?;
            if a.status!="prepared" {return Ok(json!({"action":a,"dispatch_allowed":false,"reconcile_required":true}));}
            if a.deadline_ms<=*now_ms {return Err("action deadline expired".into());}
            a.status="dispatched".into();a.attempt_id=Some(attempt_id.clone());a.external_request_id=external_request_id.clone();save_action(c,p,&a)?;
            boundary(c,p,&t,"before_dispatch")?;return Ok(json!({"action":a,"dispatch_allowed":true}));
        }
        Operation::Response{task_id,generation,lease_id,action_id,frame,now_ms}=>{
            valid_time(*now_ms)?;let t=fence(c,p,task_id,*generation,lease_id,Some(*now_ms))?;
            let mut a=owned_action(c,p,action_id,&t)?;
            let parsed=decode_frame(frame.as_bytes())?;
            match parsed.message {
                Message::Observation{project_id,task_id:ft,action_id:fa,generation:fg,..} if project_id==p && ft==*task_id && fa==*action_id && fg==*generation=>{},
                _=>return Err("response identity/authority mismatch".into()),
            }
            if a.deadline_ms<=*now_ms {return Err("late response fenced; reconcile separately".into());}
            if let Some(old)=&a.response {if old!=frame {return Err("response conflict".into());}return Ok(json!(a));}
            if a.status!="dispatched" {return Err("response requires dispatched action".into());}
            a.response=Some(frame.clone());a.status="responded".into();save_action(c,p,&a)?;boundary(c,p,&t,"validated_response")?;
        }
        Operation::Commit{task_id,generation,lease_id,action_id,actual,outcome,now_ms}=>{
            valid_time(*now_ms)?;let t=fence(c,p,task_id,*generation,lease_id,Some(*now_ms))?;
            let mut a=owned_action(c,p,action_id,&t)?;
            if let Some(receipt)=a.receipt {return Ok(receipt);}
            if a.status!="responded" || !["succeeded","failed"].contains(&outcome.as_str()) {return Err("commit requires validated response and ordinary outcome".into());}
            settle(c,p,&mut a,outcome,actual,"validated local response")?;
            boundary(c,p,&t,"outcome_committed")?;return Ok(a.receipt.unwrap());
        }
        Operation::MarkUnknown{action_id,reason}=>{
            text(reason)?;let mut a=action(c,p,action_id)?;
            if a.status!="dispatched" && a.status!="outcome_unknown" {return Err("only dispatched action can become uncertain".into());}
            a.status="outcome_unknown".into();a.uncertainty=Some(reason.clone());save_action(c,p,&a)?;
            let t=task(c,p,&a.task_id)?;boundary(c,p,&t,"outcome_unknown")?;
        }
        Operation::Resolve{action_id,resolution,actual,proof}=>{
            // Trusted local reconciliation input, never a worker-observation capability.
            text(proof)?;let mut a=action(c,p,action_id)?;
            if let Some(receipt)=a.receipt{return Ok(receipt);}
            if !["outcome_unknown","responded"].contains(&a.status.as_str()) {return Err("reconcile uncertain/stored-response action first".into());}
            if !["succeeded","failed","cancelled"].contains(&resolution.as_str()) {return Err("invalid reconciliation resolution".into());}
            settle(c,p,&mut a,resolution,actual,proof)?;
            let t=task(c,p,&a.task_id)?;boundary(c,p,&t,"outcome_reconciled")?;return Ok(a.receipt.unwrap());
        }
        Operation::Checkpoint{task_id,generation,lease_id,context,boundary:b,now_ms}=>{
            valid_time(*now_ms)?;context.validate()?;if !["yield","compaction","terminal"].contains(&b.as_str()){return Err("invalid explicit boundary".into());}
            let mut t=fence(c,p,task_id,*generation,lease_id,Some(*now_ms))?;
            if context.role_id!=t.context.role_id || context.role_version!=t.context.role_version || context.prompt_hash!=t.context.prompt_hash || context.base_hash!=t.context.base_hash || context.code_hash!=t.context.code_hash {return Err("role, prompt and code identities are immutable within a task".into());}
            validate_context_refs(c,p,context,&t)?;t.context=context.clone();save_task(c,p,&t)?;boundary(c,p,&t,b)?;
        }
        Operation::Ask{question_id,text:q}=>{
            id(question_id)?;text(q)?;
            if pending_questions(c,p)?.len()>=64{return Err("pending question ceiling".into());}
            sql(c.execute("INSERT INTO questions VALUES(?1,?2,?3,NULL)",params![p,question_id,q]))?;
            refresh_boundaries(c,p,"needs_input")?;event(c,p,"agent.needs_input",json!({"question_id":question_id}))?;
        }
        Operation::Answer{question_id,answer}=>{
            text(answer)?;
            let n=sql(c.execute("UPDATE questions SET answer=?3 WHERE project=?1 AND id=?2 AND answer IS NULL",params![p,question_id,answer]))?;
            if n!=1 {return Err("question absent/already answered".into());}refresh_boundaries(c,p,"human_answered")?;
        }
        Operation::RegisterReference{reference_id,category,digest}=>{
            id(reference_id)?;id(digest)?;if !["evidence","artifact","call","job"].contains(&category.as_str()){return Err("invalid reference category".into());}
            sql(c.execute("INSERT INTO refs VALUES(?1,?2,?3,?4)",params![p,reference_id,category,digest]))?;
        }
        Operation::Timer{timer_id,task_id,generation,due_ms,expected_project_revision,reason}=>{
            id(timer_id)?;text(reason)?;valid_time(*due_ms)?;
            let t=task(c,p,task_id)?;if t.generation!=*generation{return Err("stale timer generation".into());}
            let n:i64=sql(c.query_row("SELECT count(*) FROM timers WHERE project=?1 AND status='pending'",[p],|r|r.get(0)))?;
            if n>=256{return Err("pending timer ceiling".into());}
            sql(c.execute("INSERT INTO timers VALUES(?1,?2,?3,?4,?5,?6,?7,'pending',?8)",params![p,timer_id,task_id,generation,due_ms,expected_project_revision,reason,format!("{p}:{timer_id}")]))?;
        }
        Operation::DeliverTimers{now_ms}=>{valid_time(*now_ms)?;deliver_timers(c,p,*now_ms)?;}
        Operation::Recover{now_ms}=>{
            valid_time(*now_ms)?;
            // Validate before altering ownership: corruption is visible, never silently overwritten.
            let mut tasks=rows::<Task>(c,"tasks",p)?;
            for t in &tasks {validated_checkpoint(c,p,&t.task_id)?;}
            let pre_recovery_revision=revision(c,p)?;
            for t in &mut tasks {
                let intent_still_current=t.intent_revision==intent_rev(c,p)?;
                let old_generation=t.generation;
                t.generation=next(t.generation)?;t.owner=None;t.lease_id=None;t.expires_ms=None;
                if t.status!="cancelled" {t.status="ready".into();}t.intent_revision=intent_rev(c,p)?;
                save_task(c,p,t)?;
                if intent_still_current && t.status!="cancelled" {
                    sql(c.execute("UPDATE timers SET generation=?4,expected_revision=?5 WHERE project=?1 AND task=?2 AND generation=?3 AND status='pending'",params![p,t.task_id,old_generation,t.generation,pre_recovery_revision]))?;
                }
            }
            for mut a in rows::<Action>(c,"actions",p)? {
                if a.status=="dispatched" {a.status="outcome_unknown".into();a.uncertainty=Some("restart: dispatch may have executed/billed".into());save_action(c,p,&a)?;}
                // Prepared actions are known undispatched but still retain accounting until explicitly cancelled below.
                if a.status=="prepared" {let z=zero(&a.budget);settle(c,p,&mut a,"cancelled",&z,"restart before dispatch; no execution")?;}
            }
            refresh_boundaries(c,p,"recovery_fenced")?;deliver_timers(c,p,*now_ms)?;
            event(c,p,"project.recovered",json!({"now_ms":now_ms,"uncertain_accounting_retained":true}))?;
        }
    }
    Ok(json!({"ok":true}))
}
fn valid_time(v:i64)->Result<()> {if v<0 || v as u64>MAX_INTEGER {Err("invalid UTC millisecond time".into())}else{Ok(())}}
fn next(v:u64)->Result<u64>{if v>=MAX_INTEGER {Err("generation exhausted".into())}else{Ok(v+1)}}
fn owned_action(c:&Connection,p:&str,a:&str,t:&Task)->Result<Action>{let a=action(c,p,a)?;if a.task_id!=t.task_id || a.generation!=t.generation || a.intent_revision!=t.intent_revision || a.lease_id!=t.lease_id.clone().unwrap_or_default(){return Err("stale action fence".into());}Ok(a)}
fn zero(a:&Amounts)->Amounts {a.keys().map(|k|(k.clone(),0)).collect()}
fn settle(c:&Connection,p:&str,a:&mut Action,outcome:&str,actual:&Amounts,proof:&str)->Result<()> {
    amounts(actual)?;if actual.keys().ne(a.budget.keys()){return Err("charge dimensions must equal reservation dimensions".into());}
    // Actual overspend is recorded honestly, then prevents further admission.
    sql(c.execute("INSERT INTO charges VALUES(?1,?2,?3,?4)",params![p,a.action_id,ser(actual)?,proof]))?;
    sql(c.execute("UPDATE reservations SET held=0 WHERE project=?1 AND action=?2",params![p,a.action_id]))?;
    a.status=outcome.into();a.receipt=Some(json!({"action_id":a.action_id,"status":outcome,"charged":actual,"proof":proof}));save_action(c,p,a)?;
    event(c,p,"action.committed",a.receipt.clone().unwrap())?;Ok(())
}
fn invalidate_tasks(c:&Connection,p:&str,status:&str)->Result<()> {
    for mut t in rows::<Task>(c,"tasks",p)? {t.generation=next(t.generation)?;t.intent_revision=intent_rev(c,p)?;t.owner=None;t.lease_id=None;t.expires_ms=None;t.status=status.into();save_task(c,p,&t)?;}
    for mut a in rows::<Action>(c,"actions",p)? {
        if a.status=="prepared" {let z=zero(&a.budget);settle(c,p,&mut a,"cancelled",&z,"fenced before dispatch")?;}
        else if a.status=="dispatched" {a.status="outcome_unknown".into();a.uncertainty=Some("intent/cancellation fenced execution; accounting held".into());save_action(c,p,&a)?;}
    }
    refresh_boundaries(c,p,"intent_changed")
}
fn pending_questions(c:&Connection,p:&str)->Result<Vec<String>> {
    let mut s=sql(c.prepare("SELECT id FROM questions WHERE project=?1 AND answer IS NULL ORDER BY id"))?;
    sql(s.query_map([p],|r|r.get::<_,String>(0)))?.map(sql).collect()
}
fn refresh_boundaries(c:&Connection,p:&str,b:&str)->Result<()> {for t in rows::<Task>(c,"tasks",p)?{boundary(c,p,&t,b)?;}Ok(())}
fn boundary(c:&Connection,p:&str,t:&Task,b:&str)->Result<()> {
    let mut context=t.context.clone();context.pending_questions=pending_questions(c,p)?;
    let pending=rows::<Action>(c,"actions",p)?.into_iter().filter(|a|a.task_id==t.task_id && !["succeeded","failed","cancelled"].contains(&a.status.as_str())).map(|a|a.action_id).collect::<Vec<_>>();
    context.calls=pending;context.validate()?;
    validate_context_refs(c,p,&context,t)?;
    let sequence:u64=sql(c.query_row("SELECT coalesce(max(sequence),0)+1 FROM checkpoints WHERE project=?1 AND task=?2",params![p,t.task_id],|r|r.get(0)))?;
    event(c,p,"task.boundary",json!({"task_id":t.task_id,"boundary":b,"sequence":sequence}))?;
    let event_cursor:i64=sql(c.query_row("SELECT coalesce(max(cursor),0) FROM events WHERE project=?1",[p],|r|r.get(0)))?;
    let cp=Checkpoint{checkpoint_id:format!("{}.{sequence}",t.task_id),sequence,project_id:p.into(),branch_id:t.branch_id.clone(),task_id:t.task_id.clone(),spec_revision:sql(c.query_row("SELECT spec_revision FROM projects WHERE id=?1",[p],|r|r.get(0)))?,intent_revision:intent_rev(c,p)?,generation:t.generation,event_cursor:event_cursor.to_string(),boundary:b.into(),context};
    cp.validate()?;let data=ser(&cp)?;
    if data.len()>MAX_FRAME {return Err("checkpoint size bound".into());}
    sql(c.execute("INSERT INTO checkpoints VALUES(?1,?2,?3,?4,?5)",params![p,t.task_id,sequence,data,hash(&data)]))?;
    sql(c.execute("DELETE FROM checkpoints WHERE project=?1 AND task=?2 AND sequence<=?3",params![p,t.task_id,sequence.saturating_sub(4)]))?;
    Ok(())
}
fn validate_context_refs(c:&Connection,p:&str,ctx:&Context,t:&Task)->Result<()> {
    let (base,code):(String,String)=sql(c.query_row("SELECT base_hash,code_hash FROM branches WHERE project=?1 AND id=?2",params![p,t.branch_id],|r|Ok((r.get(0)?,r.get(1)?))))?;
    if ctx.base_hash!=base || ctx.code_hash!=code {return Err("checkpoint branch/code identity mismatch".into());}
    for (category,list) in [("evidence",&ctx.evidence),("artifact",&ctx.artifacts),("job",&ctx.jobs)] {
        for id in list {let exists:bool=sql(c.query_row("SELECT EXISTS(SELECT 1 FROM refs WHERE project=?1 AND id=?2 AND category=?3)",params![p,id,category],|r|r.get(0)))?;if !exists{return Err(format!("dangling {category} reference {id}"));}}
    }
    for a in &ctx.calls {let stored=action(c,p,a)?;if stored.task_id!=t.task_id {return Err("cross-task action reference".into());}}
    for q in &ctx.pending_questions {let exists:bool=sql(c.query_row("SELECT EXISTS(SELECT 1 FROM questions WHERE project=?1 AND id=?2)",params![p,q],|r|r.get(0)))?;if !exists{return Err("dangling question".into());}}
    Ok(())
}
fn validated_checkpoint(c:&Connection,p:&str,t:&str)->Result<Checkpoint> {
    let(data,digest):(String,String)=sql(c.query_row("SELECT data,digest FROM checkpoints WHERE project=?1 AND task=?2 ORDER BY sequence DESC LIMIT 1",params![p,t],|r|Ok((r.get(0)?,r.get(1)?))))?;
    if hash(&data)!=digest {return Err("checkpoint integrity failure; recovery blocked".into());}
    let cp:Checkpoint=de(&data)?;cp.validate()?;
    if cp.project_id!=p || cp.task_id!=t {return Err("checkpoint scope mismatch".into());}
    let task=task(c,p,t)?;if cp.branch_id!=task.branch_id {return Err("checkpoint branch mismatch".into());}
    let current_spec:u64=sql(c.query_row("SELECT spec_revision FROM projects WHERE id=?1",[p],|r|r.get(0)))?;
    if cp.spec_revision!=current_spec || cp.intent_revision>intent_rev(c,p)? || cp.generation>task.generation {return Err("checkpoint revision mismatch".into());}
    let ec=cursor(&cp.event_cursor)?;
    let exists:bool=sql(c.query_row("SELECT EXISTS(SELECT 1 FROM events WHERE project=?1 AND cursor=?2)",params![p,ec],|r|r.get(0)))?;
    if !exists {return Err("dangling event cursor".into());}
    validate_context_refs(c,p,&cp.context,&task)?;Ok(cp)
}
fn totals(c:&Connection,p:Option<&str>,column:&str,held:bool)->Result<Amounts> {
    let(table,filter)=if held {("reservations","held=1")}else{("charges","1=1")};
    let mut s=sql(c.prepare(&format!("SELECT {column} FROM {table} WHERE {filter} AND (?1 IS NULL OR project=?1)")))?;
    let policy_json:String=if let Some(project)=p {sql(c.query_row("SELECT policy FROM projects WHERE id=?1",[project],|r|r.get(0)))?}else{sql(c.query_row("SELECT policy FROM device",[],|r|r.get(0)))?};
    let policy:Policy=de(&policy_json)?;
    let mut total:Amounts=policy.budget.keys().map(|k|(k.clone(),0)).collect();
    if column=="capacity" {total=policy.capacity.keys().map(|k|(k.clone(),0)).collect();}
    for row in sql(s.query_map([p],|r|r.get::<_,String>(0)))? {
        let a:Amounts=de(&sql(row)?)?;
        for(k,v)in a {let sum=total.entry(k).or_insert(0i64);*sum=sum.checked_add(v).ok_or("ledger overflow")?;}
    } Ok(total)
}
fn admit(c:&Connection,p:&str,budget:&Amounts,capacity:&Amounts)->Result<()> {
    let device:Policy=de(&sql(c.query_row("SELECT policy FROM device",[],|r|r.get::<_,String>(0)))?)?;
    let project:Policy=de(&sql(c.query_row("SELECT policy FROM projects WHERE id=?1",[p],|r|r.get::<_,String>(0)))?)?;
    for(scope,policy)in [(None,&device),(Some(p),&project)] {
        if budget.keys().ne(policy.budget.keys()) || capacity.keys().ne(policy.capacity.keys()) {return Err("reservations must specify every policy dimension".into());}
        let held=totals(c,scope,"budget",true)?;let charged=totals(c,scope,"amount",false)?;let resources=totals(c,scope,"capacity",true)?;
        for(k,v)in budget {let limit=policy.budget.get(k).ok_or("unsupported budget dimension")?;
            let used=held.get(k).copied().unwrap_or(0).checked_add(charged.get(k).copied().unwrap_or(0)).and_then(|x|x.checked_add(*v)).ok_or("budget overflow")?;
            if used>*limit {return Err(format!("budget exhausted: {k}"));}
        }
        // Debt in any dimension blocks admission even if the new action requests another dimension.
        for(k,limit)in &policy.budget {if held.get(k).copied().unwrap_or(0).checked_add(charged.get(k).copied().unwrap_or(0)).ok_or("budget overflow")?>*limit {return Err("existing budget debt blocks admission".into());}}
        for(k,v)in capacity {let limit=policy.capacity.get(k).ok_or("unsupported capacity dimension")?;
            if resources.get(k).copied().unwrap_or(0).checked_add(*v).ok_or("capacity overflow")?>*limit{return Err(format!("capacity unavailable: {k}"));}
        }
    }Ok(())
}
fn deliver_timers(c:&Connection,p:&str,now:i64)->Result<()> {
    let mut s=sql(c.prepare("SELECT id,task,generation,expected_revision,reason,delivery_key FROM timers WHERE project=?1 AND due_ms<=?2 AND status='pending' ORDER BY due_ms,id LIMIT 256"))?;
    let records=sql(s.query_map(params![p,now],|r|Ok((r.get::<_,String>(0)?,r.get::<_,String>(1)?,r.get::<_,u64>(2)?,r.get::<_,u64>(3)?,r.get::<_,String>(4)?,r.get::<_,String>(5)?))))?.map(sql).collect::<Result<Vec<_>>>()?;
    for(id,task_id,generation,rev,reason,key)in records {
        let t=task(c,p,&task_id)?;let obsolete=t.generation!=generation || revision(c,p)?!=rev || t.status=="cancelled";
        let status=if obsolete {"obsolete"}else{"delivered"};
        sql(c.execute("UPDATE timers SET status=?3 WHERE project=?1 AND id=?2",params![p,id,status]))?;
        event(c,p,if obsolete{"timer.obsolete"}else{"timer.due"},json!({"timer_id":id,"task_id":task_id,"delivery_key":key,"reason":reason,"approval_granted":false}))?;
    }Ok(())
}
fn events(c:&Connection,p:&str,after:u64)->Result<Vec<Event>> {
    if after>i64::MAX as u64 {return Err("cursor out of range".into());}
    let mut s=sql(c.prepare("SELECT cursor,kind,payload FROM events WHERE project=?1 AND cursor>?2 ORDER BY cursor LIMIT ?3"))?;
    sql(s.query_map(params![p,after,PAGE],|r|Ok((r.get::<_,i64>(0)?,r.get::<_,String>(1)?,r.get::<_,String>(2)?))))?.map(|r|{let(cursor,kind,payload)=sql(r)?;Ok(Event{cursor:cursor.to_string(),project_id:p.into(),kind,payload:de(&payload)?})}).collect()
}
fn snapshot(c:&mut Connection,p:&str)->Result<Value> {
    let tx=sql(c.transaction())?;
    let(revision,spec_revision,intent_revision,intent,state,finalization_id):(u64,u64,u64,String,String,String)=sql(tx.query_row("SELECT revision,spec_revision,intent_revision,intent,state,finalization_id FROM projects WHERE id=?1",[p],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?))))?;
    let cursor:i64=sql(tx.query_row("SELECT coalesce(max(cursor),0) FROM events WHERE project=?1",[p],|r|r.get(0)))?;
    let project_policy:Policy=de(&sql(tx.query_row("SELECT policy FROM projects WHERE id=?1",[p],|r|r.get::<_,String>(0)))?)?;
    let device_policy:Policy=de(&sql(tx.query_row("SELECT policy FROM device",[],|r|r.get::<_,String>(0)))?)?;
    let mut s=sql(tx.prepare("SELECT id,text,answer FROM questions WHERE project=?1 ORDER BY id LIMIT 256"))?;
    let questions=sql(s.query_map([p],|r|Ok(json!({"id":r.get::<_,String>(0)?,"text":r.get::<_,String>(1)?,"answer":r.get::<_,Option<String>>(2)?}))))?.map(sql).collect::<Result<Vec<_>>>()?;
    Ok(json!({"project_id":p,"revision":revision,"spec_revision":spec_revision,"intent_revision":intent_revision,"intent":intent,"state":state,"finalization_id":finalization_id,"required_outcomes":["code","report_card","inference"],"event_cursor":cursor.to_string(),"questions":questions,
        "project_ledger":{"policy":project_policy,"reserved":totals(&tx,Some(p),"budget",true)?,"charged":totals(&tx,Some(p),"amount",false)?,"capacity_held":totals(&tx,Some(p),"capacity",true)?},
        "device_ledger":{"policy":device_policy,"reserved":totals(&tx,None,"budget",true)?,"charged":totals(&tx,None,"amount",false)?,"capacity_held":totals(&tx,None,"capacity",true)?},"tasks":rows::<Task>(&tx,"tasks",p)?,"official_scores":null,"final_model":null}))
}
fn restore(c:&mut Connection,p:&str,t:&str)->Result<RecoveryTask> {
    let tx=sql(c.transaction())?;let cp=validated_checkpoint(&tx,p,t)?;
    let task=task(&tx,p,t)?;let actions=rows::<Action>(&tx,"actions",p)?.into_iter().filter(|a|a.task_id==t).collect::<Vec<_>>();
    let state:String=sql(tx.query_row("SELECT state FROM projects WHERE id=?1",[p],|r|r.get(0)))?;
    let eligibility=if state!="created" {state.as_str()}else if !pending_questions(&tx,p)?.is_empty(){"awaiting_human"}else if actions.iter().any(|a|a.status=="outcome_unknown" || a.status=="dispatched" || a.status=="responded"){"reconcile_actions"}else if task.owner.is_none(){"acquire_ownership"}else{"continue_from_safe_boundary"};
    let checkpoint_cursor=cursor(&cp.event_cursor)?;
    let replay=events(&tx,p,checkpoint_cursor)?;
    let replay_next_cursor=replay.last().map(|e|e.cursor.clone()).unwrap_or(cp.event_cursor.clone());
    let more_count:i64=sql(tx.query_row("SELECT count(*) FROM events WHERE project=?1 AND cursor>?2",params![p,replay_next_cursor.parse::<i64>().unwrap_or(0)],|r|r.get(0)))?;
    let replay_has_more=more_count>0;
    let eligibility=if replay_has_more {"replay_events"}else{eligibility};
    Ok(RecoveryTask{task,checkpoint:cp,events:replay,replay_next_cursor,replay_has_more,actions,eligibility:eligibility.into()})
}
