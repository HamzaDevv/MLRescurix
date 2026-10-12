use mlrescurix_harness::{Runtime,Policy,Request,Operation,contracts::{Context,Amounts,Message,Frame,decode_frame}};
use std::{collections::BTreeMap,path::{Path,PathBuf},process::Command};
use serde_json::{json,Value};
fn amounts(v:i64)->Amounts { BTreeMap::from([("tokens".into(),v),("calls".into(),v)]) }
fn resources(v:i64)->Amounts { BTreeMap::from([("workers".into(),v)]) }
fn policy(limit:i64,workers:i64)->Policy { Policy{budget:amounts(limit),capacity:resources(workers),max_tasks:1024} }
fn context(base:&str,code:&str)->Context {Context{role_id:"researcher".into(),role_version:"1".into(),prompt_hash:"prompt-hash".into(),model:None,phase:"research".into(),hypothesis:None,base_hash:base.into(),code_hash:code.into(),evidence:vec![],artifacts:vec![],pending_questions:vec![],calls:vec![],jobs:vec![],next_action:"inspect".into(),stop_conditions:vec!["budget".into()]}}
fn req(project:&str,cmd:&str,op:Operation)->Request {Request{command_id:cmd.into(),project_id:project.into(),expected_revision:None,operation:op}}
fn send(r:&Runtime,p:&str,c:&str,o:Operation)->mlrescurix_harness::Receipt {r.command(req(p,c,o)).unwrap()}
fn create(r:&Runtime,p:&str) {assert_eq!(send(r,p,"project",Operation::CreateProject{intent:"preserve user request".into(),finalization_id:format!("final-{p}"),policy:policy(100,8)}).status,"applied");}
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
