use mlrescurix_harness::{Runtime,Policy,Request,Operation,contracts::{Context,Amounts,Message,Frame,decode_frame}};
use std::{collections::BTreeMap,path::{Path,PathBuf},process::Command};
use serde_json::{json,Value};
use sha2::Digest;
use mlrescurix_harness::execution::{self,JobSpec,Probe,Supervisor};
fn amounts(v:i64)->Amounts { BTreeMap::from([("tokens".into(),v),("calls".into(),v)]) }
fn resources(v:i64)->Amounts { BTreeMap::from([("workers".into(),v)]) }
fn policy(limit:i64,workers:i64)->Policy { Policy{budget:amounts(limit),capacity:resources(workers),max_tasks:1024} }
fn m1policy(limit:i64,workers:i64)->Policy {Policy{budget:BTreeMap::from([("calls".into(),limit),("tokens".into(),limit*1000),("wall_ms".into(),limit*10000)]),capacity:resources(workers),max_tasks:1024}}
fn context(base:&str,code:&str)->Context {Context{role_id:"researcher".into(),role_version:"1".into(),prompt_hash:"prompt-hash".into(),model:None,phase:"research".into(),hypothesis:None,base_hash:base.into(),code_hash:code.into(),evidence:vec![],artifacts:vec![],pending_questions:vec![],calls:vec![],jobs:vec![],next_action:"inspect".into(),stop_conditions:vec!["budget".into()]}}
fn req(project:&str,cmd:&str,op:Operation)->Request {Request{command_id:cmd.into(),project_id:project.into(),expected_revision:None,operation:op}}
fn send(r:&Runtime,p:&str,c:&str,o:Operation)->mlrescurix_harness::Receipt {r.command(req(p,c,o)).unwrap()}
fn create(r:&Runtime,p:&str) {assert_eq!(send(r,p,"project",Operation::CreateProject{intent:"preserve user request".into(),finalization_id:format!("final-{p}"),policy:policy(100,8)}).status,"applied");}
fn create_m1(r:&Runtime,p:&str){assert_eq!(send(r,p,"project",Operation::CreateProject{intent:"M1 execution fixture".into(),finalization_id:format!("final-{p}"),policy:m1policy(100,8)}).status,"applied");}
fn setup(path:&Path,p:&str)->Runtime {let r=Runtime::init(path,policy(1000,32)).unwrap();create(&r,p);r}
fn branch_task(r:&Runtime,p:&str,branch:&str,task:&str) {send(r,p,&format!("b{branch}"),Operation::CreateBranch{branch_id:branch.into(),base_hash:format!("base-{branch}"),code_hash:format!("code-{branch}")});send(r,p,&format!("t{task}"),Operation::CreateTask{task_id:task.into(),branch_id:branch.into(),context:context(&format!("base-{branch}"),&format!("code-{branch}"))});}
fn claim(r:&Runtime,p:&str,t:&str)->u64 {let v=send(r,p,&format!("claim{t}"),Operation::Claim{task_id:t.into(),owner:"lead".into(),lease_id:format!("lease{t}"),now_ms:1,expires_ms:1000,expected_generation:1});serde_json::from_value(v.result["generation"].clone()).unwrap_or(1)}
fn prepare(r:&Runtime,p:&str,t:&str,a:&str,g:u64) {send(r,p,&format!("prepare{a}"),Operation::Prepare{task_id:t.into(),generation:g,lease_id:format!("lease{t}"),action_id:a.into(),input:"run scripted observation".into(),budget:amounts(4),capacity:resources(1),deadline_ms:900,now_ms:2});}
fn dispatch(r:&Runtime,p:&str,t:&str,a:&str,g:u64) {send(r,p,&format!("dispatch{a}"),Operation::Dispatch{task_id:t.into(),generation:g,lease_id:format!("lease{t}"),action_id:a.into(),attempt_id:format!("attempt{a}"),external_request_id:Some(format!("request{a}")),now_ms:3});}
fn response(r:&Runtime,p:&str,t:&str,a:&str,g:u64) {let frame=serde_json::to_string(&Frame{version:1,message:Message::Observation{project_id:p.into(),task_id:t.into(),action_id:a.into(),generation:g,phase:"scripted".into(),detail:"fake result observation".into()}}).unwrap();send(r,p,&format!("response{a}"),Operation::Response{task_id:t.into(),generation:g,lease_id:format!("lease{t}"),action_id:a.into(),frame,now_ms:4});}
#[test] fn cross_language_fixture_corpus(){
 let root=Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts/v1/fixtures");let manifest:Value=serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
 for case in manifest.as_array().unwrap(){let data=std::fs::read(root.join(case["file"].as_str().unwrap())).unwrap();assert_eq!(decode_frame(&data).is_ok(),case["accepted"].as_bool().unwrap(),"{}",case["file"]);}
}
#[test] fn ten_projects_three_branches_reserve_atomically_and_cancel_in_isolation(){
 let d=tempfile::tempdir().unwrap();let r=Runtime::init(&d.path().join("state"),policy(100,12)).unwrap();
 for i in 0..10 {let p=format!("p{i}");create(&r,&p);if i==0 {for j in 0..3 {branch_task(&r,&p,&format!("b{j}"),&format!("t{j}"));}}}
 for j in 0..3 {let t=format!("t{j}");let g=claim(&r,"p0",&t);prepare(&r,"p0",&t,&format!("a{j}"),g);}
 let snap=r.snapshot("p0").unwrap();assert_eq!(snap["tasks"].as_array().unwrap().len(),3);assert_eq!(snap["project_ledger"]["capacity_held"]["workers"],3);
 let failed=send(&r,"p1","overflow",Operation::CreateBranch{branch_id:"isolation".into(),base_hash:"base".into(),code_hash:"code".into()});assert_eq!(failed.status,"applied");
 send(&r,"p0","cancel",Operation::Lifecycle{state:"cancelled".into()});assert_eq!(r.snapshot("p0").unwrap()["project_ledger"]["capacity_held"]["workers"],0);assert_eq!(r.snapshot("p1").unwrap()["state"],"created");
}
#[test] fn duplicates_fencing_uncertainty_and_charge_debt(){
 let d=tempfile::tempdir().unwrap();let r=setup(&d.path().join("state"),"p");branch_task(&r,"p","b","t");let g=claim(&r,"p","t");prepare(&r,"p","t","a",g);
 let duplicate=send(&r,"p","preparea",Operation::Prepare{task_id:"t".into(),generation:g,lease_id:"leaset".into(),action_id:"a".into(),input:"run scripted observation".into(),budget:amounts(4),capacity:resources(1),deadline_ms:900,now_ms:2});assert_eq!(duplicate.status,"applied");assert_eq!(duplicate.result["action_id"],"a");
 let duplicate_action=send(&r,"p","prepare-action-retry",Operation::Prepare{task_id:"t".into(),generation:g,lease_id:"leaset".into(),action_id:"a".into(),input:"run scripted observation".into(),budget:amounts(4),capacity:resources(1),deadline_ms:900,now_ms:2});assert_eq!(duplicate_action.status,"applied");assert_eq!(duplicate_action.result,duplicate.result);
 dispatch(&r,"p","t","a",g);send(&r,"p","unknown",Operation::MarkUnknown{action_id:"a".into(),reason:"lost response".into()});
 let before=r.snapshot("p").unwrap();assert_eq!(before["project_ledger"]["reserved"]["tokens"],4);assert_eq!(before["project_ledger"]["charged"]["tokens"],0);
 let stale=send(&r,"p","stale",Operation::Commit{task_id:"t".into(),generation:g,lease_id:"leaset".into(),action_id:"a".into(),actual:amounts(0),outcome:"succeeded".into(),now_ms:5});assert_eq!(stale.status,"rejected");
 let paid=send(&r,"p","reconcile",Operation::Resolve{action_id:"a".into(),resolution:"succeeded".into(),actual:amounts(110),proof:"scripted receipt".into()});assert_eq!(paid.status,"applied");
 let snap=r.snapshot("p").unwrap();assert_eq!(snap["project_ledger"]["charged"]["tokens"],110);
 let denied=send(&r,"p","newbranch",Operation::CreateBranch{branch_id:"b2".into(),base_hash:"b".into(),code_hash:"c".into()});assert_eq!(denied.status,"applied");
 let old_generation=g;let g=send(&r,"p","claim2",Operation::Claim{task_id:"t".into(),owner:"lead".into(),lease_id:"newlease".into(),now_ms:1000,expires_ms:2000,expected_generation:g});assert_eq!(g.status,"applied");let g2=g.result["generation"].as_u64().unwrap();let late=send(&r,"p","late-old-generation",Operation::Commit{task_id:"t".into(),generation:old_generation,lease_id:"leaset".into(),action_id:"a".into(),actual:amounts(0),outcome:"failed".into(),now_ms:1001});assert_eq!(late.status,"rejected");let debt=send(&r,"p","debt-prepare",Operation::Prepare{task_id:"t".into(),generation:g2,lease_id:"newlease".into(),action_id:"a2".into(),input:"new attempt".into(),budget:amounts(1),capacity:resources(1),deadline_ms:1900,now_ms:1001});assert_eq!(debt.status,"rejected");
}
#[test] fn event_and_timer_cursors_replay_and_questions_survive(){
 let d=tempfile::tempdir().unwrap();let r=setup(&d.path().join("state"),"p");branch_task(&r,"p","b","t");let g=claim(&r,"p","t");send(&r,"p","ask",Operation::Ask{question_id:"q".into(),text:"May continue?".into()});
 let denied=send(&r,"p","prep-blocked",Operation::Prepare{task_id:"t".into(),generation:g,lease_id:"leaset".into(),action_id:"a".into(),input:"run".into(),budget:amounts(1),capacity:resources(1),deadline_ms:900,now_ms:2});assert_eq!(denied.status,"rejected");
 let snap=r.snapshot("p").unwrap();assert_eq!(snap["questions"][0]["id"],"q");let cursor=snap["event_cursor"].as_str().unwrap().parse::<u64>().unwrap();assert!(!r.events("p",cursor-1).unwrap().is_empty());
 let restored=r.restore("p","t").unwrap();assert_eq!(restored.eligibility,"awaiting_human");
 let timer=send(&r,"p","timer",Operation::Timer{timer_id:"wake".into(),task_id:"t".into(),generation:g,due_ms:1,expected_project_revision:snap["revision"].as_u64().unwrap()+1,reason:"retry after delay".into()});
 let recovered=send(&r,"p","recover",Operation::Recover{now_ms:10});assert_eq!(recovered.status,"applied");
 assert!(r.events("p",0).unwrap().iter().any(|e|e.kind=="timer.due"));
 let after=r.snapshot("p").unwrap();assert_eq!(after["questions"][0]["answer"],Value::Null);
 let answer=send(&r,"p","answer",Operation::Answer{question_id:"q".into(),answer:"yes".into()});assert_eq!(answer.status,"applied");assert!(timer.revision<recovered.revision);
}
#[test] fn fake_observations_cannot_publish_official_scores_or_finalize(){
 let d=tempfile::tempdir().unwrap();let r=setup(&d.path().join("state"),"p");
 let official=send(&r,"p","official",Operation::Lifecycle{state:"completed".into()});assert_eq!(official.status,"rejected");
 let finalize=send(&r,"p","finalize",Operation::Lifecycle{state:"finalized".into()});assert_eq!(finalize.status,"rejected");
 let fake=json!({"version":1,"kind":"observation","payload":{"project_id":"p","task_id":"t","action_id":"a","generation":1,"phase":"scripted","detail":"score=0.99","official_score":0.99}});assert!(decode_frame(fake.to_string().as_bytes()).is_err());
}
#[test] fn state_root_is_owned_and_migrations_are_checked(){let d=tempfile::tempdir().unwrap();let p=d.path().join("state");let _=Runtime::init(&p,policy(10,2)).unwrap();assert!(Runtime::init(&p,policy(10,2)).is_err());assert!(Runtime::open(&p).is_ok());let other=d.path().join("other");std::fs::create_dir(&other).unwrap();assert!(Runtime::open(&other).is_err());}

fn child_actor(){let Some(path)=std::env::var_os("MLR_CRASH_PATH") else{return};let stage=std::env::var("MLR_CRASH_STAGE").unwrap();let p=PathBuf::from(path);let r=setup(&p,"p");branch_task(&r,"p","b","t");let g=claim(&r,"p","t");if stage!="before_intent"{prepare(&r,"p","t","a",g);}if stage=="after_dispatch"||stage=="after_response"||stage=="after_commit"{dispatch(&r,"p","t","a",g);}if stage=="after_response"||stage=="after_commit"{response(&r,"p","t","a",g);}if stage=="after_commit"{send(&r,"p","commit",Operation::Commit{task_id:"t".into(),generation:g,lease_id:"leaset".into(),action_id:"a".into(),actual:amounts(3),outcome:"succeeded".into(),now_ms:5});}std::process::exit(91)}
#[test] fn durable_crash_actor(){child_actor();}
#[test] fn process_crashes_recover_each_commit_boundary(){
 for (idx,stage) in ["before_intent","after_intent","after_dispatch","after_response","after_commit"].iter().enumerate(){
  let d=tempfile::tempdir().unwrap();let path=d.path().join("state");
  let output=Command::new(std::env::current_exe().unwrap()).arg("--exact").arg("durable_crash_actor").arg("--nocapture").env("MLR_CRASH_PATH",&path).env("MLR_CRASH_STAGE",stage).output().unwrap();assert_eq!(output.status.code(),Some(91),"stage {stage}: {}",String::from_utf8_lossy(&output.stderr));
  let r=Runtime::open(&path).unwrap();let before=r.snapshot("p").unwrap();let snap=send(&r,"p","recover",Operation::Recover{now_ms:10});assert_eq!(snap.status,"applied");let after=r.snapshot("p").unwrap();
  match idx {0=>assert_eq!(after["project_ledger"]["charged"]["tokens"],0),1=>{assert_eq!(after["project_ledger"]["charged"]["tokens"],0);assert_eq!(after["project_ledger"]["reserved"]["tokens"],0);},2|3=>{assert_eq!(after["project_ledger"]["reserved"]["tokens"],4);assert_eq!(after["project_ledger"]["charged"]["tokens"],0);},_=>{assert_eq!(after["project_ledger"]["charged"]["tokens"],3);assert_eq!(after["project_ledger"]["reserved"]["tokens"],0);}}
  assert!(before["event_cursor"].as_str().unwrap().parse::<u64>().unwrap()<=after["event_cursor"].as_str().unwrap().parse::<u64>().unwrap());
 }
}

#[test]
fn simultaneous_reservations_respect_device_and_project_caps() {
    let d=tempfile::tempdir().unwrap();
    let r=Runtime::init(&d.path().join("state"),policy(100,2)).unwrap();
    for i in 0..10 {let p=format!("p{i}");send(&r,&p,"project",Operation::CreateProject{intent:"independent".into(),finalization_id:format!("f{i}"),policy:policy(100,2)});branch_task(&r,&p,&format!("b{i}"),&format!("t{i}"));claim(&r,&p,&format!("t{i}"));}
    let joins=(0..10).map(|i|{let r=r.clone();std::thread::spawn(move||{
        let p=format!("p{i}");let t=format!("t{i}");
        r.command(req(&p,"reserve",Operation::Prepare{task_id:t.clone(),generation:2,lease_id:format!("lease{t}"),action_id:format!("a{i}"),input:"one worker".into(),budget:amounts(1),capacity:resources(1),deadline_ms:900,now_ms:2})).unwrap().status
    })}).collect::<Vec<_>>();
    let results=joins.into_iter().map(|j|j.join().unwrap()).collect::<Vec<_>>();
    assert_eq!(results.iter().filter(|s|s.as_str()=="applied").count(),2);
    assert_eq!(r.snapshot("p0").unwrap()["device_ledger"]["capacity_held"]["workers"],2);
    for i in 0..10 {let s=r.snapshot(&format!("p{i}")).unwrap();assert!(s["project_ledger"]["capacity_held"]["workers"]==0 || s["project_ledger"]["capacity_held"]["workers"]==1);}
}

#[test]
fn corrupted_checkpoint_and_dangling_references_fail_explicitly() {
    let d=tempfile::tempdir().unwrap();let path=d.path().join("state");let r=setup(&path,"p");branch_task(&r,"p","b","t");
    let db=rusqlite::Connection::open(path.join("state.sqlite")).unwrap();
    db.execute("UPDATE checkpoints SET digest='corrupt' WHERE project='p' AND task='t'",[]).unwrap();
    assert!(r.restore("p","t").unwrap_err().contains("integrity"));
    assert_eq!(send(&r,"p","recover",Operation::Recover{now_ms:10}).status,"rejected");
    drop(db);drop(r);
    let d2=tempfile::tempdir().unwrap();let path2=d2.path().join("state");let r=setup(&path2,"p");
    send(&r,"p","ref",Operation::RegisterReference{reference_id:"e1".into(),category:"evidence".into(),digest:"sha256".into()});
    send(&r,"p","b",Operation::CreateBranch{branch_id:"b".into(),base_hash:"base".into(),code_hash:"code".into()});
    let mut ctx=context("base","code");ctx.evidence.push("e1".into());
    send(&r,"p","task",Operation::CreateTask{task_id:"t".into(),branch_id:"b".into(),context:ctx});
    let db=rusqlite::Connection::open(path2.join("state.sqlite")).unwrap();db.execute("DELETE FROM refs WHERE project='p' AND id='e1'",[]).unwrap();
    assert!(r.restore("p","t").unwrap_err().contains("dangling evidence"));
}

#[test]
fn lifecycle_cli_initializes_fresh_state_and_returns_receipts() {
    let d=tempfile::tempdir().unwrap();let state=d.path().join("new-owned-state");
    let policy_file=d.path().join("policy.json");
    std::fs::write(&policy_file,serde_json::to_vec(&policy(100,3)).unwrap()).unwrap();
    let bin=env!("CARGO_BIN_EXE_mlrescurix-harness");
    let init=Command::new(bin).args(["init","--state",state.to_str().unwrap(),"--policy",policy_file.to_str().unwrap()]).output().unwrap();
    assert!(init.status.success(),"{}",String::from_utf8_lossy(&init.stderr));
    let request_file=d.path().join("create-project.json");
    let request=req("project","create",Operation::CreateProject{intent:"synthetic lifecycle smoke".into(),finalization_id:"f1".into(),policy:policy(20,2)});
    std::fs::write(&request_file,serde_json::to_vec(&request).unwrap()).unwrap();
    let cmd=Command::new(bin).args(["command","--state",state.to_str().unwrap(),"--request",request_file.to_str().unwrap()]).output().unwrap();
    assert!(cmd.status.success(),"{}",String::from_utf8_lossy(&cmd.stderr));
    let receipt:Value=serde_json::from_slice(&cmd.stdout).unwrap();assert_eq!(receipt["status"],"applied");
    let snap=Command::new(bin).args(["snapshot","--state",state.to_str().unwrap(),"--project","project"]).output().unwrap();
    assert!(snap.status.success());let value:Value=serde_json::from_slice(&snap.stdout).unwrap();assert_eq!(value["required_outcomes"].as_array().unwrap().len(),3);
}

fn execution_task(r:&Runtime,p:&str,t:&str){branch_task(r,p,&format!("b{t}"),t);let now=chrono_free_now();send(r,p,&format!("claim{t}"),Operation::Claim{task_id:t.into(),owner:"local-worker".into(),lease_id:format!("lease{t}"),now_ms:now,expires_ms:now+3_600_000,expected_generation:1});}
fn job(p:&str,id:&str,t:&str,probe:Probe,timeout:u64)->JobSpec{JobSpec{job_id:id.into(),project_id:p.into(),task_id:t.into(),generation:2,timeout_ms:timeout,threads:1,max_output_bytes:1024,environment_id:None,probe}}
fn run_until_terminal(s:&mut Supervisor,root:&Path,p:&str,id:&str){for _ in 0..100{ s.tick().unwrap();let j=execution::inspect(root,p,id).unwrap();if !["queued","running","terminating","recovering"].contains(&j.status.as_str()){return;}std::thread::sleep(std::time::Duration::from_millis(10));}panic!("job did not finish: {}",execution::inspect(root,p,id).unwrap().status)}

#[test]
fn m1_trusted_worker_publishes_hashed_artifact_and_records_terminal_state(){
 let d=tempfile::tempdir().unwrap();let root=d.path().join("state");let r=Runtime::init(&root,m1policy(100,8)).unwrap();create_m1(&r,"p");execution_task(&r,"p","t");
 let spec=job("p","j","t",Probe::Write{name:"result.txt".into(),content:"evidence".into()},1000);assert_eq!(execution::submit(&root,spec).unwrap().status,"queued");
 let bin=Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));let mut s=Supervisor::start_with_executable(&root,1,bin).unwrap();run_until_terminal(&mut s,&root,"p","j");
 let got=execution::inspect(&root,"p","j").unwrap();assert_eq!(got.status,"succeeded");assert_eq!(got.artifacts[0]["sha256"],format!("{:x}",sha2::Sha256::digest(b"evidence")));
 let snapshot=r.snapshot("p").unwrap();assert!(snapshot["project_ledger"]["charged"]["wall_ms"].as_i64().unwrap()>0);assert_eq!(snapshot["project_ledger"]["reserved"]["wall_ms"],0);
 let db=rusqlite::Connection::open(root.join("state.sqlite")).unwrap();let count:i64=db.query_row("SELECT count(*) FROM managed_artifacts WHERE project='p' AND job='j'",[],|x|x.get(0)).unwrap();assert_eq!(count,1);
 execution::pin_artifact(&root,"p","j",true).unwrap();assert_eq!(execution::collect_artifacts(&root).unwrap(),0);execution::pin_artifact(&root,"p","j",false).unwrap();assert!(execution::collect_artifacts(&root).unwrap()>0);assert!(!root.join("artifacts/p/j").exists());
}

#[test]
fn m1_managed_python_environment_is_owned_and_leases_block_removal(){
 let d=tempfile::tempdir().unwrap();let root=d.path().join("state");let r=Runtime::init(&root,m1policy(100,8)).unwrap();create_m1(&r,"p");execution_task(&r,"p","t");
 let env=execution::create_environment(&root,"py313",Path::new("/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13")).unwrap();assert_eq!(env["status"],"ready");let mut s_job=job("p","envjob","t",Probe::Sleep{millis:100},1000);s_job.environment_id=Some("py313".into());execution::submit(&root,s_job).unwrap();
 let bin=Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));let mut s=Supervisor::start_with_executable(&root,1,bin).unwrap();
 for _ in 0..50{s.tick().unwrap();if execution::inspect(&root,"p","envjob").unwrap().status=="running"{break;}std::thread::sleep(std::time::Duration::from_millis(10));}
 assert!(execution::remove_environment(&root,"py313").unwrap_err().contains("leased"));run_until_terminal(&mut s,&root,"p","envjob");execution::remove_environment(&root,"py313").unwrap();assert!(!root.join("environments/py313").exists());
}

#[test]
fn m1_restart_reconciles_published_artifact_and_discards_incomplete_stage(){
 let d=tempfile::tempdir().unwrap();let root=d.path().join("state");let r=Runtime::init(&root,m1policy(100,8)).unwrap();create_m1(&r,"p");execution_task(&r,"p","t");
 let published=job("p","published","t",Probe::Write{name:"x".into(),content:"done".into()},20);let interrupted=job("p","interrupted","t",Probe::Sleep{millis:20},20);
 execution::submit(&root,published.clone()).unwrap();execution::submit(&root,interrupted.clone()).unwrap();
 let db=rusqlite::Connection::open(root.join("state.sqlite")).unwrap();let started=chrono_free_now()-10_000;
 let final_dir=root.join("artifacts/p/published");std::fs::create_dir_all(&final_dir).unwrap();std::fs::write(final_dir.join("x"),"done").unwrap();
 let stage=root.join("artifacts/p/.interrupted.staging");std::fs::create_dir_all(&stage).unwrap();std::fs::write(stage.join("job.json"),"{}").unwrap();
 for s in [&published,&interrupted] {let folder=if s.job_id=="published"{final_dir.to_string_lossy().to_string()}else{stage.to_string_lossy().to_string()};db.execute("UPDATE jobs SET status='running',pid=999999,started_ms=?2,stage=?3 WHERE project='p' AND id=?1 AND status='queued'",rusqlite::params![s.job_id,started,folder]).unwrap();}
 drop(db);let bin=Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));let mut supervisor=Supervisor::start_with_executable(&root,1,bin).unwrap();supervisor.tick().unwrap();
 assert_eq!(execution::inspect(&root,"p","published").unwrap().status,"succeeded");assert_eq!(execution::inspect(&root,"p","published").unwrap().artifacts[0]["name"],"x");
 assert_eq!(execution::inspect(&root,"p","interrupted").unwrap().status,"interrupted");assert!(!stage.exists());
}
fn chrono_free_now()->i64{std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as i64}

#[test]
fn m1_deadline_crash_and_project_cancellation_reap_only_owned_workers(){
 let d=tempfile::tempdir().unwrap();let root=d.path().join("state");let r=Runtime::init(&root,m1policy(100,8)).unwrap();create_m1(&r,"p");create_m1(&r,"q");execution_task(&r,"p","t");execution_task(&r,"q","u");
 execution::submit(&root,job("p","slow","t",Probe::Sleep{millis:500},50)).unwrap();execution::submit(&root,job("q","other","u",Probe::Write{name:"x".into(),content:"q".into()},1000)).unwrap();
 let bin=Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));let mut s=Supervisor::start_with_executable(&root,1,bin).unwrap();
 for _ in 0..100{ s.tick().unwrap();let j=execution::inspect(&root,"p","slow").unwrap();if j.status=="running"{break;}std::thread::sleep(std::time::Duration::from_millis(10)); }
 send(&r,"p","cancel-project",Operation::Lifecycle{state:"cancelled".into()});run_until_terminal(&mut s,&root,"p","slow");
 assert_eq!(execution::inspect(&root,"p","slow").unwrap().status,"cancelled");
 run_until_terminal(&mut s,&root,"q","other");assert_eq!(execution::inspect(&root,"q","other").unwrap().status,"succeeded");
 drop(s);
 let mut s=Supervisor::start_with_executable(&root,1,bin).unwrap();execution::submit(&root,job("q","deadline","u",Probe::Sleep{millis:200},30)).unwrap();run_until_terminal(&mut s,&root,"q","deadline");
 assert_eq!(execution::inspect(&root,"q","deadline").unwrap().status,"timed_out");
}

#[test]
fn m1_pause_drains_active_job_and_resume_reopens_queued_admission(){
 let d=tempfile::tempdir().unwrap();let root=d.path().join("state");let r=Runtime::init(&root,m1policy(100,8)).unwrap();create_m1(&r,"p");execution_task(&r,"p","t1");execution_task(&r,"p","t2");
 execution::submit(&root,job("p","active","t1",Probe::Sleep{millis:120},1000)).unwrap();execution::submit(&root,job("p","queued","t2",Probe::Write{name:"ok".into(),content:"resumed".into()},1000)).unwrap();
 let bin=Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));let mut s=Supervisor::start_with_executable(&root,1,bin).unwrap();
 for _ in 0..50{s.tick().unwrap();if execution::inspect(&root,"p","active").unwrap().status=="running"{break;}std::thread::sleep(std::time::Duration::from_millis(10));}
 send(&r,"p","pause",Operation::Lifecycle{state:"paused".into()});run_until_terminal(&mut s,&root,"p","active");assert_eq!(execution::inspect(&root,"p","active").unwrap().status,"succeeded");assert_eq!(execution::inspect(&root,"p","queued").unwrap().status,"queued");
 send(&r,"p","resume",Operation::Lifecycle{state:"created".into()});run_until_terminal(&mut s,&root,"p","queued");assert_eq!(execution::inspect(&root,"p","queued").unwrap().status,"succeeded");
}

#[test]
fn m1_supervisor_death_closes_worker_control_and_keeps_budget_reserved_until_recovery(){
 let d=tempfile::tempdir().unwrap();let root=d.path().join("state");let r=Runtime::init(&root,m1policy(100,8)).unwrap();create_m1(&r,"p");execution_task(&r,"p","t");
 execution::submit(&root,job("p","orphan","t",Probe::Sleep{millis:30_000},500)).unwrap();let bin=Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));
 let mut service=Command::new(bin).args(["supervise","--state",root.to_str().unwrap(),"--workers","1"]).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).spawn().unwrap();
 let mut pid=None;for _ in 0..100{let j=execution::inspect(&root,"p","orphan").unwrap();if j.status=="running"{pid=j.pid;break;}std::thread::sleep(std::time::Duration::from_millis(10));}let worker=pid.expect("supervisor admitted job");
 service.kill().unwrap();service.wait().unwrap();for _ in 0..100{let alive=unsafe{libc::kill(worker as i32,0)}==0;if !alive{break;}std::thread::sleep(std::time::Duration::from_millis(10));}
 let alive=unsafe{libc::kill(worker as i32,0)}==0;assert!(!alive,"worker remained after its supervisor control pipe closed");
 let mut recovered=Supervisor::start_with_executable(&root,1,bin).unwrap();for _ in 0..200{recovered.tick().unwrap();let j=execution::inspect(&root,"p","orphan").unwrap();if j.status=="interrupted"{break;}std::thread::sleep(std::time::Duration::from_millis(10));}
 assert_eq!(execution::inspect(&root,"p","orphan").unwrap().status,"interrupted");let snap=r.snapshot("p").unwrap();assert!(snap["project_ledger"]["charged"]["wall_ms"].as_i64().unwrap()>=500);
}

#[test]
fn m1_process_telemetry_is_optional_and_bounded_to_latest_value(){
 let d=tempfile::tempdir().unwrap();let root=d.path().join("state");let r=Runtime::init(&root,m1policy(100,8)).unwrap();create_m1(&r,"p");execution_task(&r,"p","t");execution::submit(&root,job("p","sampled","t",Probe::Sleep{millis:1500},3000)).unwrap();
 let bin=Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));let mut supervisor=Supervisor::start_with_executable(&root,1,bin).unwrap();let mut sample=None;
 for _ in 0..30{supervisor.tick().unwrap();sample=execution::inspect(&root,"p","sampled").unwrap().telemetry;if sample.is_some(){break;}std::thread::sleep(std::time::Duration::from_millis(50));}
 if let Some(sample)=sample{assert!(sample.rss_bytes.unwrap()>0);assert!(sample.cpu_percent.unwrap().is_finite());}
 run_until_terminal(&mut supervisor,&root,"p","sampled");let db=rusqlite::Connection::open(root.join("state.sqlite")).unwrap();assert!(db.query_row::<i64,_,_>("SELECT count(*) FROM job_telemetry WHERE project='p' AND job='sampled'",[],|x|x.get(0)).unwrap()<=1);
}

#[test]
fn m1_fair_queue_and_contract_bounds(){
 let d=tempfile::tempdir().unwrap();let root=d.path().join("state");let r=Runtime::init(&root,m1policy(100,8)).unwrap();create_m1(&r,"a");create_m1(&r,"b");execution_task(&r,"a","ta");execution_task(&r,"b","tb");
 for (p,id,t) in [("a","a1","ta"),("a","a2","ta"),("b","b1","tb")] {execution::submit(&root,job(p,id,t,Probe::Sleep{millis:80},1000)).unwrap();}
 let bin=Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));let mut s=Supervisor::start_with_executable(&root,1,bin).unwrap();run_until_terminal(&mut s,&root,"a","a1");
 for _ in 0..30{s.tick().unwrap();std::thread::sleep(std::time::Duration::from_millis(10));if execution::inspect(&root,"b","b1").unwrap().status!="queued"{break;}}
 assert_ne!(execution::inspect(&root,"b","b1").unwrap().status,"queued");
 let mut bad=job("a","bad","ta",Probe::Write{name:"../escape".into(),content:"x".into()},100);assert!(execution::validate_spec(&bad).is_err());bad.max_output_bytes=u64::MAX;assert!(execution::validate_spec(&bad).is_err());
}

#[test]
fn m1_ten_projects_and_three_branch_jobs_stay_within_two_worker_allocation(){
 let d=tempfile::tempdir().unwrap();let root=d.path().join("state");let r=Runtime::init(&root,m1policy(1000,32)).unwrap();
 for i in 0..10{create_m1(&r,&format!("p{i}"));}
 for j in 0..3{execution_task(&r,"p0",&format!("t{j}"));execution::submit(&root,job("p0",&format!("j{j}"),&format!("t{j}"),Probe::Sleep{millis:40},1000)).unwrap();}
 for i in 1..10{let p=format!("p{i}");let t=format!("t{i}");execution_task(&r,&p,&t);execution::submit(&root,job(&p,&format!("j{i}"),&t,Probe::Sleep{millis:40},1000)).unwrap();}
 let bin=Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));let mut s=Supervisor::start_with_executable(&root,2,bin).unwrap();assert!(Supervisor::start_with_executable(&root,2,bin).is_err());
 let mut max=0;for _ in 0..200{s.tick().unwrap();let db=rusqlite::Connection::open(root.join("state.sqlite")).unwrap();let running:i64=db.query_row("SELECT count(*) FROM jobs WHERE status IN('running','terminating')",[],|x|x.get(0)).unwrap();max=max.max(running);let done:i64=db.query_row("SELECT count(*) FROM jobs WHERE status='succeeded'",[],|x|x.get(0)).unwrap();if done==12{break;}std::thread::sleep(std::time::Duration::from_millis(10));}
 assert!(max<=2);let db=rusqlite::Connection::open(root.join("state.sqlite")).unwrap();assert_eq!(db.query_row::<i64,_,_>("SELECT count(*) FROM jobs WHERE status='succeeded'",[],|x|x.get(0)).unwrap(),12);
}

#[test]
fn m2_dataset_registration_copies_and_hashes_only_into_owned_state(){
 let d=tempfile::tempdir().unwrap();let root=d.path().join("state");let r=Runtime::init(&root,m1policy(100,8)).unwrap();create_m1(&r,"p");
 let source=d.path().join("selected.csv");std::fs::write(&source,"x,target\n1,0\n2,1\n").unwrap();
 let registered=execution::register_dataset(&root,"p","data-1",&source).unwrap();
 assert_eq!(registered["bytes"],std::fs::metadata(&source).unwrap().len());
 assert_eq!(std::fs::read(root.join("datasets/p/data-1.csv")).unwrap(),std::fs::read(&source).unwrap());
 assert_eq!(execution::register_dataset(&root,"p","data-1",&source).unwrap()["reused"],true);
 assert_eq!(r.snapshot("p").unwrap()["project_id"],"p");
 std::fs::write(&source,"x,target\n9,9\n").unwrap();assert!(execution::register_dataset(&root,"p","data-1",&source).is_err());
}

#[test]
fn m2_scientific_jobs_are_typed_and_reference_registered_datasets(){
 let d=tempfile::tempdir().unwrap();let root=d.path().join("state");let r=Runtime::init(&root,m1policy(100,8)).unwrap();create_m1(&r,"p");
 let source=d.path().join("selected.csv");std::fs::write(&source,"x,target\n1,0\n2,1\n").unwrap();execution::register_dataset(&root,"p","data-1",&source).unwrap();execution_task(&r,"p","audit-task");
 let mut spec=job("p","audit","audit-task",Probe::Scientific{operation:"audit".into(),dataset_id:"data-1".into(),spec:Some(serde_json::json!({"project_id":"p","target":"target","task":"binary_classification","metric":"balanced_accuracy"})),candidate_id:None,test_access_id:None},30_000);
 spec.environment_id=Some("scientific".into());assert!(execution::validate_spec(&spec).is_ok());
 assert_eq!(execution::submit(&root,spec).unwrap().status,"queued");
 let db=rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
 assert_eq!(db.query_row::<String,_,_>("SELECT operation FROM scientific_jobs WHERE project='p' AND job='audit'",[],|r|r.get(0)).unwrap(),"audit");
}

#[test]
fn m2_final_test_access_is_reserved_transactionally_once_per_dataset(){
 let d=tempfile::tempdir().unwrap();let root=d.path().join("state");let r=Runtime::init(&root,m1policy(100,8)).unwrap();create_m1(&r,"p");
 let source=d.path().join("selected.csv");std::fs::write(&source,"x,target\n1,0\n2,1\n").unwrap();execution::register_dataset(&root,"p","data-1",&source).unwrap();
 let workflow=root.join("workflows/p");std::fs::create_dir_all(&workflow).unwrap();
 let db=rusqlite::Connection::open(root.join("state.sqlite")).unwrap();let sha:String=db.query_row("SELECT sha256 FROM scientific_datasets WHERE project='p' AND id='data-1'",[],|r|r.get(0)).unwrap();
 std::fs::write(workflow.join("results.json"),serde_json::json!({"dataset_sha256":sha,"candidates":[{"candidate_id":"linear","status":"evaluated"}]}).to_string()).unwrap();
 execution_task(&r,"p","final-task-1");
 let mk=|job_id:&str,task_id:&str,access:&str|{let mut request=job("p",job_id,task_id,Probe::Scientific{operation:"finalize".into(),dataset_id:"data-1".into(),spec:None,candidate_id:Some("linear".into()),test_access_id:Some(access.into())},30_000);request.environment_id=Some("scientific".into());request};
 execution::submit(&root,mk("final-1","final-task-1","test-key-1")).unwrap();
 execution_task(&r,"p","final-task-2");
 assert!(execution::submit(&root,mk("final-2","final-task-2","test-key-2")).is_err());
 assert_eq!(db.query_row::<i64,_,_>("SELECT count(*) FROM final_test_gates WHERE project='p' AND status='reserved'",[],|r|r.get(0)).unwrap(),1);
}

#[test]
fn m2_restart_reconciles_consumed_final_test_identity(){
 let d=tempfile::tempdir().unwrap();let root=d.path().join("state");let r=Runtime::init(&root,m1policy(100,8)).unwrap();create_m1(&r,"p");
 let source=d.path().join("selected.csv");std::fs::write(&source,"x,target\n1,0\n2,1\n").unwrap();execution::register_dataset(&root,"p","data-1",&source).unwrap();
 let workflow=root.join("workflows/p");std::fs::create_dir_all(&workflow).unwrap();let db=rusqlite::Connection::open(root.join("state.sqlite")).unwrap();let sha:String=db.query_row("SELECT sha256 FROM scientific_datasets WHERE project='p' AND id='data-1'",[],|r|r.get(0)).unwrap();
 std::fs::write(workflow.join("results.json"),serde_json::json!({"dataset_sha256":sha,"candidates":[{"candidate_id":"linear","status":"evaluated"}]}).to_string()).unwrap();execution_task(&r,"p","final-task");
 let mut request=job("p","final-job","final-task",Probe::Scientific{operation:"finalize".into(),dataset_id:"data-1".into(),spec:None,candidate_id:Some("linear".into()),test_access_id:Some("test-once".into())},30_000);request.environment_id=Some("scientific".into());execution::submit(&root,request).unwrap();
 std::fs::write(workflow.join("test_access.json"),serde_json::json!({"test_access_id":"test-once","candidate_id":"linear","dataset_sha256":sha,"frozen_model_sha256":"model-hash","status":"consumed"}).to_string()).unwrap();
 std::fs::write(workflow.join("final.json"),serde_json::json!({"status":"verified_development_only","test_access_id":"test-once"}).to_string()).unwrap();
 let bin=Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));let _supervisor=Supervisor::start_with_executable(&root,1,bin).unwrap();
 assert_eq!(db.query_row::<String,_,_>("SELECT status FROM final_test_gates WHERE project='p' AND access_id='test-once'",[],|r|r.get(0)).unwrap(),"consumed");
}
