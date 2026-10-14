use std::{collections::BTreeMap,fs,path::Path,time::{Duration,SystemTime,UNIX_EPOCH}};
use serde_json::{json,Value};
use mlrescurix_harness::{Runtime,core::*,execution::{self,JobSpec,Probe,Supervisor}};
use mlrescurix_harness::contracts::Context;
fn now()->i64{SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as i64}
pub fn policy()->Policy{Policy{budget:BTreeMap::from([("wall_ms".into(),600_000)]),capacity:BTreeMap::from([("workers".into(),2)]),max_tasks:128}}
pub fn setup(root:&Path,p:&str){
    let r=Runtime::open(root).unwrap();
    for (id,op) in [
        (format!("project-{p}"),Operation::CreateProject{intent:"Docker pipeline integration fixture".into(),finalization_id:format!("f-{p}"),policy:policy()}),
        (format!("branch-{p}"),Operation::CreateBranch{branch_id:"main".into(),base_hash:"base".into(),code_hash:"code".into()}),
        (format!("task-{p}"),Operation::CreateTask{task_id:"t".into(),branch_id:"main".into(),context:Context{role_id:"lead".into(),role_version:"v1".into(),prompt_hash:"prompt".into(),model:None,phase:"search".into(),hypothesis:None,base_hash:"base".into(),code_hash:"code".into(),evidence:vec![],artifacts:vec![],pending_questions:vec![],calls:vec![],jobs:vec![],next_action:"search".into(),stop_conditions:vec![]}}),
        (format!("claim-{p}"),Operation::Claim{task_id:"t".into(),owner:"fixture".into(),lease_id:"l".into(),now_ms:now(),expires_ms:now()+600_000,expected_generation:1})
    ] {let receipt=r.command(Request{command_id:id,project_id:p.into(),expected_revision:None,operation:op}).unwrap();assert_eq!(receipt.status,"applied","{}",receipt.result);}
}
fn job(root:&Path,p:&str,id:&str,operation:&str,spec:Option<Value>){
    execution::submit(root,JobSpec{ resources: execution::resources::ResourceRequirements{require_memory_certification:true,..Default::default()},job_id:id.into(),project_id:p.into(),task_id:"t".into(),generation:2,timeout_ms:45_000,threads:1,max_output_bytes:4_194_304,environment_id:Some("science".into()),probe:Probe::Scientific{operation:operation.into(),dataset_id:"d".into(),spec,candidate_id:None,test_access_id:None}}).unwrap();
}
fn wait(root:&Path,p:&str,id:&str,s:&mut Supervisor){
    for _ in 0..600 {s.tick().unwrap();let view=execution::inspect(root,p,id).unwrap();if view.status=="succeeded"{let receipt=view.resource_receipt.as_ref().expect("container receipt");assert_eq!(receipt["exit_confirmed"],true);assert_eq!(receipt["measurement_complete"],true,"{receipt}");return;}
        assert!(matches!(view.status.as_str(),"queued"|"running"|"recovering"),"{}: {:?}; log={}",id,view.reason,fs::read_to_string(root.join("logs").join(p).join(format!("{id}.log"))).unwrap_or_default());std::thread::sleep(Duration::from_millis(50));}
    panic!("pipeline deadline");
}
#[test]
#[ignore="explicit pinned scientific Docker image required; no host fallback"]
fn scheduled_features_calibration_ensemble_and_verified_oof(){
    let temp=tempfile::tempdir().unwrap();let root=temp.path().join("state");Runtime::init(&root,policy()).unwrap();
    execution::register_container_environment(&root,"science",&std::env::var("MLRESCURIX_BUNDLE_IMAGE").unwrap()).unwrap();
    let mut supervisor=Supervisor::start_with_executable(&root,2,Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"))).unwrap();
    for (p,task,metric,split,cal,ensemble) in [
        ("binary","binary_classification","balanced_accuracy","stratified",Some("sigmoid"),"voting"),
        ("multi","multiclass_classification","log_loss","stratified",Some("isotonic"),"voting"),
        ("reg","regression","rmse","random",None,"stacking"),
        ("group","binary_classification","balanced_accuracy","grouped",Some("sigmoid"),"voting"),
        ("time","regression","mae","chronological",None,"stacking"),
        ("revoked","binary_classification","balanced_accuracy","stratified",Some("sigmoid"),"voting")
    ] {
        setup(&root,p);let csv=temp.path().join(format!("{p}.csv"));let mut rows=String::from("x,z,category,group,target\n");
        for i in 0..150 {let target=if task=="regression"{format!("{}",(i%17) as f64*0.6+(i%3) as f64)}else{format!("{}",i%if p=="multi"{3}else{2})};
            rows.push_str(&format!("{},{},c{},{},{}\n",if i%13==0{String::new()}else{format!("{}",i%17)},i%9,i%5,i/5,target));}
        fs::write(&csv,rows).unwrap();execution::register_dataset(&root,p,"d",&csv).unwrap();
        let mut spec=json!({"project_id":p,"target":"target","task":task,"metric":metric,"split":split,"inference_repetitions":3,"inference_batch_size":8});
        if split=="grouped"{spec["group_column"]=json!("group");}
        job(&root,p,"audit","audit",Some(spec));wait(&root,p,"audit",&mut supervisor);
        let protocol=json!({"format":"mlrescurix.pipeline.v1","experiment_id":"full","parents":[{"recipe":"linear","parameters":{"C":1.0}},{"recipe":"linear","parameters":{"C":0.5}}],"features":{"transform_id":"features","interactions":true,"missing_indicators":true,"target_encoding":true},"calibration":cal,"ensemble":ensemble,"folds":3,"seed":104729});
        // Regression linear params differ; use default normalized recipe values.
        let mut protocol=protocol;if task=="regression"{protocol["parents"][0]["parameters"]=json!({"alpha":1.0});protocol["parents"][1]["parameters"]=json!({"alpha":2.0});}
        job(&root,p,"pipeline","pipeline",Some(protocol));wait(&root,p,"pipeline",&mut supervisor);
        let db=rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
        let receipt:String=db.query_row("SELECT receipt_json FROM pipeline_receipts WHERE project=?1 AND candidate='full'",[p],|r|r.get(0)).unwrap();
        let receipt:Value=serde_json::from_str(&receipt).unwrap();assert!(receipt["fit_count"].as_u64().unwrap()>=8);
        assert_eq!(receipt["status"],"evaluated_development");
        let binding=execution::finalization_binding(&root,p,"full").unwrap();
        if let Ok(path)=std::env::var("MLRESCURIX_EVIDENCE_DIR") {
            let path=Path::new(&path);fs::create_dir_all(path).unwrap();
            fs::write(path.join(format!("{p}-pipeline-receipt.json")),serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
            fs::write(path.join(format!("{p}-finalization-binding.json")),serde_json::to_vec_pretty(&binding).unwrap()).unwrap();
        }
        assert_eq!(binding["status"],"reviewable_development_candidate");
        assert!(execution::require_finalization_decisions(&root,p,"full").unwrap_err().contains("content-bound user_approval"));
        let model=root.join("workflows").join(p).join("trials/full/parent-0-full.joblib");
        let original=fs::read(&model).unwrap();fs::write(&model,"tampered managed parent").unwrap();
        assert!(execution::finalization_binding(&root,p,"full").unwrap_err().contains("parent artifact bytes"));
        fs::write(&model,original).unwrap();
        let gates:i64=db.query_row("SELECT count(*) FROM final_test_gates WHERE project=?1",[p],|r|r.get(0)).unwrap();assert_eq!(gates,0);
        let binding_sha=binding["binding_sha256"].as_str().unwrap().to_string();
        // Producer task ("t") cannot audit its own pipeline
        assert!(execution::record_content_decision(&root,p,"full",&binding_sha,"critic_audit","t",true,"self-audit").unwrap_err().contains("cannot independently audit"));
        // Stale binding digest is rejected
        assert!(execution::record_content_decision(&root,p,"full","0000000000000000000000000000000000000000000000000000000000000000","user_approval","human",true,"wrong digest").unwrap_err().contains("content binding changed"));
        // Record valid content-bound user approval and independent critic audit
        execution::record_content_decision(&root,p,"full",&binding_sha,"user_approval","local-human",true,"Approved composite pipeline receipt").unwrap();
        assert!(execution::require_finalization_decisions(&root,p,"full").unwrap_err().contains("content-bound critic_audit"));
        execution::record_content_decision(&root,p,"full",&binding_sha,"critic_audit","independent-critic-task",true,"Independent critic verified leak-free OOF and constraints").unwrap();
        let verified_decisions=execution::require_finalization_decisions(&root,p,"full").unwrap();
        assert_eq!(verified_decisions["binding_sha256"],binding_sha);

        let substituted=JobSpec{ resources: Default::default(),
            job_id:"substituted-finalize".into(),project_id:p.into(),task_id:"t".into(),generation:2,timeout_ms:60_000,threads:1,max_output_bytes:4_194_304,
            environment_id:Some("substitute".into()),
            probe:Probe::Scientific{operation:"finalize".into(),dataset_id:"d".into(),spec:None,candidate_id:Some("full".into()),test_access_id:Some(format!("substituted-gate-{p}"))}
        };
        assert!(execution::submit(&root,substituted).unwrap_err().contains("environment differs"));

        // Submit and run composite pipeline finalization in the pinned scientific Docker image
        execution::submit(&root,JobSpec{ resources: Default::default(),
            job_id:"finalize".into(),project_id:p.into(),task_id:"t".into(),generation:2,timeout_ms:60_000,threads:1,max_output_bytes:4_194_304,
            environment_id:Some("science".into()),
            probe:Probe::Scientific{operation:"finalize".into(),dataset_id:"d".into(),spec:None,candidate_id:Some("full".into()),test_access_id:Some(format!("test-gate-{p}"))}
        }).unwrap();
        if p=="revoked" {
            db.execute("UPDATE content_decisions SET approved=0 WHERE project=?1 AND kind='critic_audit'",[p]).unwrap();
            supervisor.tick().unwrap();
            let rejected=execution::inspect(&root,p,"finalize").unwrap();
            assert_eq!(rejected.status,"rejected");
            assert!(rejected.reason.unwrap().contains("content-bound critic_audit"));
            assert!(!root.join("workflows").join(p).join("test_access.json").exists());
            println!("revoked: queued finalization rejected before scientific dispatch and sealed-test access");
            continue;
        }
        wait(&root,p,"finalize",&mut supervisor);

        let gate_status:String=db.query_row("SELECT status FROM final_test_gates WHERE project=?1 AND access_id=?2",rusqlite::params![p,format!("test-gate-{p}")],|r|r.get(0)).unwrap();
        assert_eq!(gate_status,"consumed");
        let final_json:Value=serde_json::from_slice(&fs::read(root.join("workflows").join(p).join("final.json")).unwrap()).unwrap();
        assert_eq!(final_json["status"],"verified_development_only");
        assert_eq!(final_json["candidate_id"],"full");
        assert_eq!(final_json["bundle"],"bundle");
        let lineage_json:Value=serde_json::from_slice(&fs::read(root.join("workflows").join(p).join("lineage.json")).unwrap()).unwrap();
        assert_eq!(lineage_json["format"],"mlrescurix.pipeline-lineage.v1");
        assert_eq!(lineage_json["test_row_intersection"],0);

        let bundle_dir=root.join("workflows").join(p).join("bundle");
        let bundle_report=mlrescurix_harness::bundle::verify_bundle_with_state(&bundle_dir,true,Some(&root)).unwrap();
        assert_eq!(bundle_report.status,"sandbox_examples_verified");
        assert_eq!(bundle_report.candidate_id,"full");
        assert_eq!(bundle_report.details["authenticity"],"verified_against_authoritative_state");
        assert_eq!(bundle_report.details["feasibility"],"sandbox_measured_and_constraints_verified");
        assert!(bundle_report.details["sandbox_telemetry"]["peak_rss_kb"].as_u64().unwrap()>0);

        // Verify that a self-rehashed bundle (where a file is modified and manifest.json is recomputed) is rejected by state-bound verification
        let forged_dir=temp.path().join(format!("forged-{p}"));
        mlrescurix_harness::agents::workspace::BranchWorkspaceManager::copy_dir_recursive(&bundle_dir,&forged_dir).unwrap();
        fs::write(forged_dir.join("README.md"),"forged readme with recomputed self-manifest").unwrap();
        let mut forged_manifest:mlrescurix_harness::bundle::BundleManifest=serde_json::from_slice(&fs::read(forged_dir.join("manifest.json")).unwrap()).unwrap();
        use sha2::{Digest,Sha256};
        let forged_readme=fs::read(forged_dir.join("README.md")).unwrap();
        forged_manifest.files.insert("README.md".into(),mlrescurix_harness::bundle::BundleManifestFileEntry{bytes:forged_readme.len() as u64,sha256:format!("{:x}",Sha256::digest(&forged_readme))});
        fs::write(forged_dir.join("manifest.json"),serde_json::to_vec(&forged_manifest).unwrap()).unwrap();
        // Self-only integrity passes, but external state-bound verification rejects the self-rehashed bundle
        assert_eq!(mlrescurix_harness::bundle::verify_bundle(&forged_dir,false).unwrap().details["authenticity"],"not_established_without_external_trust");
        let forged_err=mlrescurix_harness::bundle::verify_bundle_with_state(&forged_dir,false,Some(&root)).unwrap_err();
        assert!(forged_err.contains("self-rehashed bundle rejected"),"unexpected error: {forged_err}");

        db.execute("DELETE FROM managed_artifacts WHERE project=?1 AND name='bundle/manifest.json'",[p]).unwrap();
        let missing=mlrescurix_harness::bundle::verify_bundle_with_state(&bundle_dir,false,Some(&root)).unwrap_err();
        assert!(missing.contains("authoritative bundle manifest receipt is missing"),"{missing}");

        if let Ok(path)=std::env::var("MLRESCURIX_EVIDENCE_DIR") {
            let path=Path::new(&path);
            fs::write(path.join(format!("{p}-final.json")),serde_json::to_vec_pretty(&final_json).unwrap()).unwrap();
            fs::write(path.join(format!("{p}-lineage.json")),serde_json::to_vec_pretty(&lineage_json).unwrap()).unwrap();
            fs::write(path.join(format!("{p}-bundle-verify.json")),serde_json::to_vec_pretty(&bundle_report).unwrap()).unwrap();
        }

        // Second finalization or new pipeline after test gate consumption is permanently rejected
        let dup_err=execution::submit(&root,JobSpec{ resources: Default::default(),
            job_id:"finalize-dup".into(),project_id:p.into(),task_id:"t".into(),generation:2,timeout_ms:60_000,threads:1,max_output_bytes:4_194_304,
            environment_id:Some("science".into()),
            probe:Probe::Scientific{operation:"finalize".into(),dataset_id:"d".into(),spec:None,candidate_id:Some("full".into()),test_access_id:Some(format!("test-gate-{p}-2"))}
        }).unwrap_err();
        assert!(dup_err.contains("scientific research is closed by the durable final-test gate"));
        println!("{}: fits={} cold_load_ms={} full_raw_input_p95_ms={} delivered_bytes={} OOF={} final_test={}",p,receipt["fit_count"],receipt["cold_load_ms"],receipt["warm_p95_ms"],receipt["delivered_model_bytes"],receipt["oof_sha256"],final_json["test_metrics"]);
    }

    // Also verify pre-test feasibility failure releases the final_test_gates reservation without touching sealed test rows
    let p="infeasible_pretest";
    setup(&root,p);
    let csv=temp.path().join("infeasible.csv");
    let mut rows=String::from("x,z,category,target\n");
    for i in 0..90 {rows.push_str(&format!("{},{},c{},{}\n",i%11,i%7,i%3,i%2));}
    fs::write(&csv,rows).unwrap();
    execution::register_dataset(&root,p,"d",&csv).unwrap();
    let spec=json!({"project_id":p,"target":"target","task":"binary_classification","metric":"balanced_accuracy","split":"stratified","inference_repetitions":3,"inference_batch_size":8,"max_bundle_bytes":50000});
    job(&root,p,"audit","audit",Some(spec));wait(&root,p,"audit",&mut supervisor);
    let protocol=json!({"format":"mlrescurix.pipeline.v1","experiment_id":"full","parents":[{"recipe":"linear","parameters":{"C":1.0}},{"recipe":"linear","parameters":{"C":0.5}}],"features":{"transform_id":"features","interactions":false,"missing_indicators":true,"target_encoding":false},"calibration":"sigmoid","ensemble":"voting","folds":3,"seed":104729});
    job(&root,p,"pipeline","pipeline",Some(protocol));wait(&root,p,"pipeline",&mut supervisor);
    let binding=execution::finalization_binding(&root,p,"full").unwrap();
    let b_sha=binding["binding_sha256"].as_str().unwrap();
    execution::record_content_decision(&root,p,"full",b_sha,"user_approval","human",true,"approve").unwrap();
    execution::record_content_decision(&root,p,"full",b_sha,"critic_audit","critic-task",true,"audit").unwrap();
    execution::submit(&root,JobSpec{ resources: Default::default(),
        job_id:"finalize-infeasible".into(),project_id:p.into(),task_id:"t".into(),generation:2,timeout_ms:60_000,threads:1,max_output_bytes:4_194_304,
        environment_id:Some("science".into()),
        probe:Probe::Scientific{operation:"finalize".into(),dataset_id:"d".into(),spec:None,candidate_id:Some("full".into()),test_access_id:Some("test-gate-infeasible".into())}
    }).unwrap();
    for _ in 0..600 {
        supervisor.tick().unwrap();
        let v=execution::inspect(&root,p,"finalize-infeasible").unwrap();
        if v.status=="failed"{break;}
        std::thread::sleep(Duration::from_millis(50));
    }
    let v=execution::inspect(&root,p,"finalize-infeasible").unwrap();
    assert_eq!(v.status,"failed");
    assert!(!root.join("workflows").join(p).join("test_access.json").exists(),"Sealed test gate must not be written when pre-test feasibility fails");
    let db=rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let held_gates:i64=db.query_row("SELECT count(*) FROM final_test_gates WHERE project=?1",[p],|r|r.get(0)).unwrap();
    assert_eq!(held_gates,0,"Pre-test feasibility failure must release final_test_gates reservation");

    drop(supervisor);
    let supervisor=Supervisor::start_with_executable(&root,1,Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"))).unwrap();
    drop(supervisor);
}
