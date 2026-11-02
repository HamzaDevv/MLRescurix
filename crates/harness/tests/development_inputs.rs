use mlrescurix_harness::development_inputs::{separate, separate_stream};
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

#[test]
fn streamed_separation_checks_hash_limits_and_writer_errors() {
    let data = b"__mlr_row_id,x,y\n0,train,1\n1,validation,7\n";
    let mut train = Vec::new();
    let mut validation = Vec::new();
    let labels = separate_stream(&data[..], &digest(data), "y", &[0], &[1],
        &mut train, &mut validation, 1024).unwrap();
    assert_eq!(labels, vec![(1, 7.0)]);
    assert_eq!(validation, b"__mlr_row_id,x\n1,validation\n");
    assert!(separate_stream(&data[..], &"0".repeat(64), "y", &[0], &[1],
        Vec::new(), Vec::new(), 1024).is_err());
    assert!(separate_stream(&data[..], &digest(data), "y", &[0], &[1],
        Vec::new(), Vec::new(), 20).is_err());
    struct FailingWriter;
    impl std::io::Write for FailingWriter {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> { Err(std::io::Error::other("disk full")) }
        fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
    }
    assert!(separate_stream(&data[..], &digest(data), "y", &[0], &[1],
        FailingWriter, Vec::new(), 1024).is_err());
}

#[test]
fn stream_supports_data_above_legacy_memory_limit() {
    let mut data = b"__mlr_row_id,x,y\n".to_vec();
    let feature = "a".repeat(32_768);
    for id in 0..1025 { data.extend_from_slice(format!("{id},{feature},1\n").as_bytes()); }
    assert!(data.len() > 32 * 1024 * 1024);
    let train: Vec<u64> = (0..1024).collect();
    let labels = separate_stream(&data[..], &digest(&data), "y", &train, &[1024],
        std::io::sink(), std::io::sink(), 64 * 1024 * 1024).unwrap();
    assert_eq!(labels, vec![(1024, 1.0)]);
    assert!(separate(&data, &digest(&data), "y", &train, &[1024]).is_err());
}
