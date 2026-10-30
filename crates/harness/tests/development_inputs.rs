use mlrescurix_harness::development_inputs::separate;
use sha2::{Digest, Sha256};

fn digest(data: &[u8]) -> String { format!("{:x}", Sha256::digest(data)) }

#[test]
fn training_and_inference_inputs_never_include_validation_labels() {
    let data = b"__mlr_row_id,feature,target\n0,train,1\n1,\"multi\nline,feature\",83749\n2,val2,42\n";
    let prepared = separate(data, &digest(data), "target", &[0], &[1, 2]).unwrap();
    let training = String::from_utf8(prepared.training_csv).unwrap();
    let validation = String::from_utf8(prepared.validation_features_csv).unwrap();
    assert_eq!(training, "__mlr_row_id,feature,target\n0,train,1\n");
    assert_eq!(validation, "__mlr_row_id,feature\n1,\"multi\nline,feature\"\n2,val2\n");
    assert_eq!(prepared.labels, vec![(1, 83749.0), (2, 42.0)]);
}

#[test]
fn rejects_tampered_missing_duplicate_or_foreign_rows() {
    let data = b"__mlr_row_id,x,y\n0,1,0\n1,2,1\n";
    assert!(separate(data, &"f".repeat(64), "y", &[0], &[1]).is_err());
    assert!(separate(data, &digest(data), "y", &[0], &[0]).is_err());
    assert!(separate(data, &digest(data), "y", &[0], &[2]).is_err());
    assert!(separate(data, &digest(data), "y", &[0], &[1, 2]).is_err());
    let dup = b"__mlr_row_id,x,y\n0,1,0\n1,2,1\n1,3,0\n";
    assert!(separate(dup, &digest(dup), "y", &[0], &[1]).is_err());
    let dup_column = b"__mlr_row_id,y,y\n0,1,0\n1,2,1\n";
    assert!(separate(dup_column, &digest(dup_column), "y", &[0], &[1]).is_err());
    let nonfinite = b"__mlr_row_id,x,y\n0,1,0\n1,2,NaN\n";
    assert!(separate(nonfinite, &digest(nonfinite), "y", &[0], &[1]).is_err());
}
