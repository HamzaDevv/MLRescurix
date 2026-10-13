//! Live inference evidence: fixture fitting/deserialization occurs only in Docker.
use std::{fs, path::Path, process::Command, collections::BTreeMap};
use sha2::{Digest, Sha256};
use serde_json::{Value,json};
use mlrescurix_harness::bundle::verify_bundle;

fn seal(root: &Path) {
    let mut files=BTreeMap::new();
    for entry in fs::read_dir(root).unwrap() {
        let p=entry.unwrap().path(); let name=p.file_name().unwrap().to_str().unwrap();
        if name=="manifest.json" {continue;}
        let bytes=fs::read(&p).unwrap();
        files.insert(name.to_string(),json!({"bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(bytes))}));
    }
    fs::write(root.join("manifest.json"),json!({"format":"mlrescurix.bundle.v1","project_id":"fixture","candidate_id":"linear","files":files}).to_string()).unwrap();
}

#[test]
#[ignore = "requires explicit local digest-pinned scientific Docker image"]
fn scientific_bundle_parity_schema_and_hostile_containment() {
    let image=std::env::var("MLRESCURIX_BUNDLE_IMAGE").expect("pinned scientific image required");
    let temp=tempfile::tempdir().unwrap(); let root=temp.path();
    let protected=tempfile::tempdir().unwrap();
    fs::write(protected.path().join("labels.csv"),"protected-fixture-labels\n").unwrap();
    fs::write(protected.path().join("secret"),"disposable-secret-fixture\n").unwrap();
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(root,fs::Permissions::from_mode(0o777)).unwrap();
    let script=r#"
import json, hashlib, joblib, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
p=Path('/out'); x=pd.DataFrame({'x':[0.,1.,2.,3.,4.,5.]}); y=[0,0,0,1,1,1]
m=LogisticRegression().fit(x,y); joblib.dump(m,p/'frozen_model.joblib')
def write(n,v): (p/n).write_text(json.dumps(v))
write('input_schema.json',{'features':['x'],'task':'binary_classification'})
write('scorecard.json',{'candidate_id':'linear','constraints':{}})
write('verification.json',{'frozen_model_sha256':hashlib.sha256((p/'frozen_model.joblib').read_bytes()).hexdigest()})
write('final.json',{'candidate_id':'linear'})
write('examples_input.json',x.to_dict(orient='records'))
write('examples_expected.json',{'prediction':m.predict(x).tolist(),'probabilities':m.predict_proba(x).tolist(),'classes':m.classes_.tolist()})
(p/'inference.lock').write_text(Path('/opt/mlrescurix/requirements.lock').read_text())
(p/'README.md').write_text('Docker-only trusted logistic fixture; no release or generalization claim.')
"#;
    let output=Command::new("docker").args(["run","--rm","--pull=never","--network","none","--read-only","--user","1000:1000","--cap-drop","ALL","--security-opt","no-new-privileges:true","--memory","512m","--cpus","1","--pids-limit","64","-v"])
        .arg(format!("{}:/out:rw",root.display())).args([&image,"python","-c",script]).output().unwrap();
    assert!(output.status.success(),"{}",String::from_utf8_lossy(&output.stderr));
    let inference=r#"
import json, joblib, pandas as pd
from pathlib import Path
p=Path(__file__).parent
m=joblib.load(p/'frozen_model.joblib')
def predict(rows):
    x=pd.DataFrame(rows)
    if list(x.columns)!=['x']: raise ValueError('schema')
    return {'prediction':m.predict(x).tolist(),'probabilities':m.predict_proba(x).tolist(),'classes':m.classes_.tolist()}
"#;
    fs::write(root.join("inference.py"),inference).unwrap(); seal(root);
    let original=fs::read(root.join("manifest.json")).unwrap();
    let report=verify_bundle(root,true).unwrap();
    assert_eq!(report.status,"sandbox_examples_verified");
    assert_eq!(fs::read(root.join("manifest.json")).unwrap(),original);
    let expected:Value=serde_json::from_slice(&fs::read(root.join("examples_expected.json")).unwrap()).unwrap();
    let mut wrong=expected.clone(); wrong["probabilities"][0]=json!([0.5,0.5]);
    fs::write(root.join("examples_expected.json"),wrong.to_string()).unwrap(); seal(root);
    assert!(verify_bundle(root,true).unwrap_err().contains("PROBABILITY_PARITY_MISMATCH"));
    wrong=expected.clone();wrong["classes"]=json!([1,0]);
    fs::write(root.join("examples_expected.json"),wrong.to_string()).unwrap();seal(root);
    assert!(verify_bundle(root,true).unwrap_err().contains("CLASS_ORDER_MISMATCH"));
    fs::write(root.join("examples_expected.json"),expected.to_string()).unwrap();
    for (replacement,message) in [
        ("if list(x.columns)!=['x']: raise ValueError('schema')",""),
        ("if 'x' not in x: x['x']=0","missing column was not rejected"),
        ("if 'x' not in x: raise ValueError('schema')\n    x=x[['x']]","unexpected column was not rejected"),
    ] {
        if message.is_empty(){continue;}
        fs::write(root.join("inference.py"),inference.replace("if list(x.columns)!=['x']: raise ValueError('schema')",replacement)).unwrap(); seal(root);
        assert!(verify_bundle(root,true).unwrap_err().contains(message));
    }
    // Bundle code is actually imported, but gets only the disposable bundle mount.
    let hostile=r#"
import os,socket
from pathlib import Path
assert os.getuid()==1000
assert 'M6_SENTINEL_SECRET' not in os.environ
assert not Path('/workspace/labels.csv').exists()
assert not Path('/host').exists()
assert not Path('__PROTECTED_PATH__/labels.csv').exists()
assert not Path('__PROTECTED_PATH__/secret').exists()
for name in ['/workspace/HOST_WRITE','/etc/HOST_WRITE','__PROTECTED_PATH__/HOST_WRITE']:
    try: Path(name).write_text('escape')
    except OSError: pass
    else: raise RuntimeError('write escaped containment')
s=socket.socket();s.settimeout(.2)
try: s.connect(('1.1.1.1',443))
except OSError: pass
else: raise RuntimeError('network escaped containment')
finally: s.close()
"#;
    let hostile=hostile.replace("__PROTECTED_PATH__",protected.path().to_str().unwrap());
    fs::write(root.join("inference.py"),format!("{hostile}\n{inference}")).unwrap(); seal(root);
    verify_bundle(root,true).unwrap();
    assert!(!root.join("HOST_WRITE").exists());
    assert!(!protected.path().join("HOST_WRITE").exists());
    fs::write(root.join("inference.lock"),"numpy==0.0.0\n").unwrap();seal(root);
    assert!(verify_bundle(root,true).unwrap_err().contains("bundle dependency version mismatch"));
    println!("M6: fitted Docker logistic model; label/probability/class parity; missing/extra schema rejection; hostile import containment; lock mismatch denial");
}
