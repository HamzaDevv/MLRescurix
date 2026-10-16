//! Multi-Fidelity Iteration & Hypothesis Loop (Phase P10.3).
//!
//! Provides:
//! - Domain models and serialization for `IterationHypothesis` and `ParetoFrontier`
//!   adhering to Draft 2020-12 schemas.
//! - Fast subsampled proxy screening (Tier 0, N <= 500) and full cross-validation (Tier 1).
//! - Deterministic hypothesis settlement (confirmed, refuted, inconclusive).
//! - Multi-objective Pareto frontier tracking (validation score, latency ms, peak memory bytes).
//! - Additive SQLite persistence (`iteration_hypotheses`, `pareto_frontier_records`)
//!   under `PRAGMA user_version = 10`.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Result;

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(i64::MAX as u128) as i64
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpectedMetricDelta {
    pub metric: String,
    pub direction: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_threshold: Option<f64>,
    #[serde(default = "default_min_improvement")]
    pub minimum_improvement: f64,
}

fn default_min_improvement() -> f64 {
    0.005
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProxyTierBudget {
    #[serde(default = "default_max_samples")]
    pub max_samples: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fidelity_level: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub screening_threshold: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
}

fn default_max_samples() -> usize {
    500
}

impl Default for ProxyTierBudget {
    fn default() -> Self {
        Self {
            max_samples: 500,
            fidelity_level: None,
            screening_threshold: None,
            timeout_ms: Some(15000),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FullTierBudget {
    #[serde(default = "default_folds")]
    pub folds: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
}

fn default_folds() -> usize {
    3
}

impl Default for FullTierBudget {
    fn default() -> Self {
        Self {
            folds: 3,
            seed: Some(2026),
            timeout_ms: Some(45000),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiFidelityBudget {
    #[serde(default)]
    pub proxy_tier: ProxyTierBudget,
    #[serde(default)]
    pub full_tier: FullTierBudget,
}

impl Default for MultiFidelityBudget {
    fn default() -> Self {
        Self {
            proxy_tier: ProxyTierBudget::default(),
            full_tier: FullTierBudget::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HypothesisOutcome {
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_metric: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy_metric_observed: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_metric_observed: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metric_delta: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub settled_at_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IterationHypothesis {
    pub schema_version: String,
    pub hypothesis_id: String,
    pub candidate_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_candidate_id: Option<String>,
    pub mutation_type: String,
    pub hypothesis_statement: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
    pub expected_metric_delta: ExpectedMetricDelta,
    pub multi_fidelity_budget: MultiFidelityBudget,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<HypothesisOutcome>,
    #[serde(default)]
    pub created_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParetoRecord {
    pub candidate_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_candidate_id: Option<String>,
    pub validation_score: f64,
    pub latency_ms: f64,
    pub peak_memory_bytes: i64,
    pub domination_status: String,
    pub dominated_by: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParetoFrontier {
    pub schema_version: String,
    pub frontier_id: String,
    pub project_id: String,
    pub primary_metric_name: String,
    pub optimization_direction: String,
    pub records: Vec<ParetoRecord>,
    pub frontier_candidate_ids: Vec<String>,
    pub non_dominated_count: usize,
    pub dominated_count: usize,
    pub evaluated_at_ms: i64,
}

/// Compute non-dominated Pareto frontier across (score, latency, memory).
pub fn compute_pareto_frontier(
    records: &[ParetoRecord],
    primary_metric: &str,
    direction: &str,
    project: &str,
    frontier_id: Option<&str>,
) -> ParetoFrontier {
    let now = now_ms();
    let f_id = frontier_id
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("frontier-{project}-{now}"));

    let mut out_records = records.to_vec();
    for r in &mut out_records {
        r.domination_status = "non_dominated".into();
        r.dominated_by.clear();
    }

    let n = out_records.len();
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let cand_a = &out_records[i];
            let cand_b = &out_records[j];

            let score_better_or_equal = if direction == "maximize" {
                cand_b.validation_score >= cand_a.validation_score
            } else {
                cand_b.validation_score <= cand_a.validation_score
            };

            let score_strictly_better = if direction == "maximize" {
                cand_b.validation_score > cand_a.validation_score
            } else {
                cand_b.validation_score < cand_a.validation_score
            };

            let lat_better_or_equal = cand_b.latency_ms <= cand_a.latency_ms;
            let lat_strictly_better = cand_b.latency_ms < cand_a.latency_ms;

            let mem_better_or_equal = cand_b.peak_memory_bytes <= cand_a.peak_memory_bytes;
            let mem_strictly_better = cand_b.peak_memory_bytes < cand_a.peak_memory_bytes;

            if score_better_or_equal
                && lat_better_or_equal
                && mem_better_or_equal
                && (score_strictly_better || lat_strictly_better || mem_strictly_better)
            {
                let dominating_id = cand_b.candidate_id.clone();
                out_records[i].dominated_by.push(dominating_id);
                out_records[i].domination_status = "dominated".into();
            }
        }
    }

    let frontier_ids: Vec<String> = out_records
        .iter()
        .filter(|r| r.domination_status == "non_dominated")
        .map(|r| r.candidate_id.clone())
        .collect();

    let non_dom_count = frontier_ids.len();
    let dom_count = n - non_dom_count;

    ParetoFrontier {
        schema_version: "v2".into(),
        frontier_id: f_id,
        project_id: project.into(),
        primary_metric_name: primary_metric.into(),
        optimization_direction: direction.into(),
        records: out_records,
        frontier_candidate_ids: frontier_ids,
        non_dominated_count: non_dom_count,
        dominated_count: dom_count,
        evaluated_at_ms: now,
    }
}

/// Settle an iteration hypothesis.
pub fn settle_hypothesis(
    hyp: &mut IterationHypothesis,
    observed_metric: Option<f64>,
    baseline_score: Option<f64>,
    screening_rejected: bool,
    rejection_reason: Option<&str>,
) {
    let now = now_ms();
    let direction = &hyp.expected_metric_delta.direction;
    let target = hyp.expected_metric_delta.target_threshold;
    let min_improvement = hyp.expected_metric_delta.minimum_improvement;

    if screening_rejected {
        hyp.outcome = Some(HypothesisOutcome {
            status: "refuted".into(),
            baseline_metric: baseline_score,
            proxy_metric_observed: observed_metric,
            full_metric_observed: None,
            metric_delta: match (observed_metric, baseline_score) {
                (Some(o), Some(b)) => Some(o - b),
                _ => None,
            },
            reason: Some(
                rejection_reason
                    .unwrap_or("Rejected during proxy screening; mutation not viable.")
                    .into(),
            ),
            settled_at_ms: Some(now),
        });
        return;
    }

    let observed = match observed_metric {
        Some(o) => o,
        None => {
            hyp.outcome = Some(HypothesisOutcome {
                status: "refuted".into(),
                baseline_metric: baseline_score,
                proxy_metric_observed: None,
                full_metric_observed: None,
                metric_delta: None,
                reason: Some(
                    rejection_reason
                        .unwrap_or("Evaluation produced no valid metric score.")
                        .into(),
                ),
                settled_at_ms: Some(now),
            });
            return;
        }
    };

    let delta = baseline_score.map(|b| observed - b).unwrap_or(0.0);

    let (status, reason) = if direction == "maximize" {
        let improvement = delta;
        let is_better = improvement >= min_improvement;
        let is_worse = improvement < 0.0;
        let meets_target = target.map(|t| observed >= t).unwrap_or(true);

        if baseline_score.is_none() {
            if meets_target {
                ("confirmed", format!("Observed score {observed:.4} without baseline meets target"))
            } else {
                ("refuted", format!("Observed score {observed:.4} without baseline failed target {target:?}"))
            }
        } else if is_better && meets_target {
            ("confirmed", format!("Confirmed: improvement +{improvement:.4} >= {min_improvement:.4} and reached target"))
        } else if is_worse {
            ("refuted", format!("Refuted: performance degraded ({delta:.4}) relative to baseline"))
        } else if !meets_target {
            ("inconclusive", format!("Inconclusive: improved by +{improvement:.4} but did not reach target {target:?}"))
        } else {
            ("inconclusive", format!("Inconclusive: improvement +{improvement:.4} was below required {min_improvement:.4}"))
        }
    } else {
        let improvement = -delta;
        let is_better = improvement >= min_improvement;
        let is_worse = improvement < 0.0;
        let meets_target = target.map(|t| observed <= t).unwrap_or(true);

        if baseline_score.is_none() {
            if meets_target {
                ("confirmed", format!("Observed metric {observed:.4} without baseline meets target"))
            } else {
                ("refuted", format!("Observed metric {observed:.4} without baseline failed target {target:?}"))
            }
        } else if is_better && meets_target {
            ("confirmed", format!("Confirmed: metric reduced by {improvement:.4} >= {min_improvement:.4} and reached target"))
        } else if is_worse {
            ("refuted", format!("Refuted: metric increased by {delta:.4} relative to baseline"))
        } else if !meets_target {
            ("inconclusive", format!("Inconclusive: reduced by {improvement:.4} but did not reach target {target:?}"))
        } else {
            ("inconclusive", format!("Inconclusive: improvement {improvement:.4} was below required {min_improvement:.4}"))
        }
    };

    hyp.outcome = Some(HypothesisOutcome {
        status: status.into(),
        baseline_metric: baseline_score,
        proxy_metric_observed: None,
        full_metric_observed: Some(observed),
        metric_delta: Some(delta),
        reason: Some(reason),
        settled_at_ms: Some(now),
    });
}

/// Record an iteration hypothesis into SQLite table `iteration_hypotheses`.
pub fn record_hypothesis(
    root: &Path,
    project: &str,
    hyp: &IterationHypothesis,
) -> Result<()> {
    crate::execution::migrate(root)?;
    let db = Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;

    let hyp_json = serde_json::to_string_pretty(hyp).map_err(|e| e.to_string())?;
    let status = hyp
        .outcome
        .as_ref()
        .map(|o| o.status.as_str())
        .unwrap_or("pending");
    let baseline = hyp.outcome.as_ref().and_then(|o| o.baseline_metric);
    let observed = hyp.outcome.as_ref().and_then(|o| o.full_metric_observed.or(o.proxy_metric_observed));
    let delta = hyp.outcome.as_ref().and_then(|o| o.metric_delta);
    let reason = hyp.outcome.as_ref().and_then(|o| o.reason.as_deref());
    let settled_ms = hyp.outcome.as_ref().and_then(|o| o.settled_at_ms);

    db.execute(
        "INSERT INTO iteration_hypotheses (
            project, hypothesis_id, candidate_id, parent_candidate_id, mutation_type,
            hypothesis_statement, expected_metric, direction, target_threshold, minimum_improvement,
            status, baseline_metric, observed_metric, metric_delta, settlement_reason,
            hypothesis_json, created_ms, settled_ms
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)
        ON CONFLICT(project, hypothesis_id) DO UPDATE SET
            status = excluded.status,
            baseline_metric = excluded.baseline_metric,
            observed_metric = excluded.observed_metric,
            metric_delta = excluded.metric_delta,
            settlement_reason = excluded.settlement_reason,
            hypothesis_json = excluded.hypothesis_json,
            settled_ms = excluded.settled_ms",
        params![
            project,
            hyp.hypothesis_id,
            hyp.candidate_id,
            hyp.parent_candidate_id,
            hyp.mutation_type,
            hyp.hypothesis_statement,
            hyp.expected_metric_delta.metric,
            hyp.expected_metric_delta.direction,
            hyp.expected_metric_delta.target_threshold,
            hyp.expected_metric_delta.minimum_improvement,
            status,
            baseline,
            observed,
            delta,
            reason,
            hyp_json,
            hyp.created_ms,
            settled_ms,
        ],
    ).map_err(|e| format!("Failed to record iteration hypothesis: {e}"))?;

    Ok(())
}

/// Retrieve an iteration hypothesis by hypothesis_id.
pub fn get_hypothesis(
    root: &Path,
    project: &str,
    hypothesis_id: &str,
) -> Result<IterationHypothesis> {
    crate::execution::migrate(root)?;
    let db = Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;

    let json_str: String = db.query_row(
        "SELECT hypothesis_json FROM iteration_hypotheses WHERE project = ?1 AND hypothesis_id = ?2",
        params![project, hypothesis_id],
        |r| r.get(0),
    ).map_err(|_| format!("Iteration hypothesis '{hypothesis_id}' not found in project '{project}'"))?;

    serde_json::from_str(&json_str).map_err(|e| format!("Corrupt hypothesis JSON: {e}"))
}

/// List hypotheses for a project, optionally filtered by candidate_id or status.
pub fn list_hypotheses(
    root: &Path,
    project: &str,
    candidate_id: Option<&str>,
    status: Option<&str>,
    limit: usize,
) -> Result<Vec<IterationHypothesis>> {
    crate::execution::migrate(root)?;
    let db = Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;

    let json_rows: Vec<String> = match (candidate_id, status) {
        (Some(cid), Some(st)) => {
            let mut stmt = db.prepare(
                "SELECT hypothesis_json FROM iteration_hypotheses
                 WHERE project = ?1 AND candidate_id = ?2 AND status = ?3
                 ORDER BY created_ms DESC LIMIT ?4"
            ).map_err(|e| e.to_string())?;
            let rows = stmt.query_map(params![project, cid, st, limit as i64], |r| r.get::<_, String>(0)).map_err(|e| e.to_string())?;
            rows.filter_map(|r| r.ok()).collect()
        }
        (Some(cid), None) => {
            let mut stmt = db.prepare(
                "SELECT hypothesis_json FROM iteration_hypotheses
                 WHERE project = ?1 AND candidate_id = ?2
                 ORDER BY created_ms DESC LIMIT ?3"
            ).map_err(|e| e.to_string())?;
            let rows = stmt.query_map(params![project, cid, limit as i64], |r| r.get::<_, String>(0)).map_err(|e| e.to_string())?;
            rows.filter_map(|r| r.ok()).collect()
        }
        (None, Some(st)) => {
            let mut stmt = db.prepare(
                "SELECT hypothesis_json FROM iteration_hypotheses
                 WHERE project = ?1 AND status = ?2
                 ORDER BY created_ms DESC LIMIT ?3"
            ).map_err(|e| e.to_string())?;
            let rows = stmt.query_map(params![project, st, limit as i64], |r| r.get::<_, String>(0)).map_err(|e| e.to_string())?;
            rows.filter_map(|r| r.ok()).collect()
        }
        (None, None) => {
            let mut stmt = db.prepare(
                "SELECT hypothesis_json FROM iteration_hypotheses
                 WHERE project = ?1
                 ORDER BY created_ms DESC LIMIT ?2"
            ).map_err(|e| e.to_string())?;
            let rows = stmt.query_map(params![project, limit as i64], |r| r.get::<_, String>(0)).map_err(|e| e.to_string())?;
            rows.filter_map(|r| r.ok()).collect()
        }
    };

    let mut result = Vec::with_capacity(json_rows.len());
    for s in json_rows {
        if let Ok(h) = serde_json::from_str::<IterationHypothesis>(&s) {
            result.push(h);
        }
    }
    Ok(result)
}

/// Record a Pareto frontier record into SQLite table `pareto_frontier_records`.
pub fn record_pareto_frontier(
    root: &Path,
    project: &str,
    frontier: &ParetoFrontier,
) -> Result<()> {
    crate::execution::migrate(root)?;
    let db = Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;

    let records_json = serde_json::to_string(&frontier.records).map_err(|e| e.to_string())?;
    let frontier_ids_json = serde_json::to_string(&frontier.frontier_candidate_ids).map_err(|e| e.to_string())?;

    db.execute(
        "INSERT INTO pareto_frontier_records (
            project, frontier_id, primary_metric_name, optimization_direction,
            records_json, frontier_candidate_ids_json, non_dominated_count, dominated_count, evaluated_at_ms
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
        ON CONFLICT(project, frontier_id) DO UPDATE SET
            records_json = excluded.records_json,
            frontier_candidate_ids_json = excluded.frontier_candidate_ids_json,
            non_dominated_count = excluded.non_dominated_count,
            dominated_count = excluded.dominated_count,
            evaluated_at_ms = excluded.evaluated_at_ms",
        params![
            project,
            frontier.frontier_id,
            frontier.primary_metric_name,
            frontier.optimization_direction,
            records_json,
            frontier_ids_json,
            frontier.non_dominated_count as i64,
            frontier.dominated_count as i64,
            frontier.evaluated_at_ms,
        ],
    ).map_err(|e| format!("Failed to record Pareto frontier: {e}"))?;

    Ok(())
}

/// Retrieve the latest Pareto frontier for a project.
pub fn get_latest_pareto_frontier(
    root: &Path,
    project: &str,
) -> Result<Option<ParetoFrontier>> {
    crate::execution::migrate(root)?;
    let db = Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;

    let row = db.query_row(
        "SELECT frontier_id, primary_metric_name, optimization_direction,
                records_json, frontier_candidate_ids_json, non_dominated_count, dominated_count, evaluated_at_ms
         FROM pareto_frontier_records
         WHERE project = ?1
         ORDER BY evaluated_at_ms DESC LIMIT 1",
        params![project],
        |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, i64>(5)?,
                r.get::<_, i64>(6)?,
                r.get::<_, i64>(7)?,
            ))
        },
    );

    match row {
        Ok((fid, pmn, od, rec_json, fids_json, ndc, dc, ems)) => {
            let records: Vec<ParetoRecord> = serde_json::from_str(&rec_json).unwrap_or_default();
            let frontier_ids: Vec<String> = serde_json::from_str(&fids_json).unwrap_or_default();
            Ok(Some(ParetoFrontier {
                schema_version: "v2".into(),
                frontier_id: fid,
                project_id: project.into(),
                primary_metric_name: pmn,
                optimization_direction: od,
                records,
                frontier_candidate_ids: frontier_ids,
                non_dominated_count: ndc as usize,
                dominated_count: dc as usize,
                evaluated_at_ms: ems,
            }))
        }
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(format!("Failed to query latest Pareto frontier: {e}")),
    }
}

/// Retrieve a specific Pareto frontier by frontier_id.
pub fn get_pareto_frontier(
    root: &Path,
    project: &str,
    frontier_id: &str,
) -> Result<ParetoFrontier> {
    crate::execution::migrate(root)?;
    let db = Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;

    let row = db.query_row(
        "SELECT frontier_id, primary_metric_name, optimization_direction,
                records_json, frontier_candidate_ids_json, non_dominated_count, dominated_count, evaluated_at_ms
         FROM pareto_frontier_records
         WHERE project = ?1 AND frontier_id = ?2",
        params![project, frontier_id],
        |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, i64>(5)?,
                r.get::<_, i64>(6)?,
                r.get::<_, i64>(7)?,
            ))
        },
    ).map_err(|_| format!("Pareto frontier '{frontier_id}' not found in project '{project}'"))?;

    let (fid, pmn, od, rec_json, fids_json, ndc, dc, ems) = row;
    let records: Vec<ParetoRecord> = serde_json::from_str(&rec_json).unwrap_or_default();
    let frontier_ids: Vec<String> = serde_json::from_str(&fids_json).unwrap_or_default();
    Ok(ParetoFrontier {
        schema_version: "v2".into(),
        frontier_id: fid,
        project_id: project.into(),
        primary_metric_name: pmn,
        optimization_direction: od,
        records,
        frontier_candidate_ids: frontier_ids,
        non_dominated_count: ndc as usize,
        dominated_count: dc as usize,
        evaluated_at_ms: ems,
    })
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_pareto_frontier_computation_maximize() {
        let records = vec![
            ParetoRecord {
                candidate_id: "c1".into(),
                parent_candidate_id: None,
                validation_score: 0.85,
                latency_ms: 100.0,
                peak_memory_bytes: 1_000_000,
                domination_status: "non_dominated".into(),
                dominated_by: vec![],
                metadata: None,
            },
            ParetoRecord {
                candidate_id: "c2".into(),
                parent_candidate_id: Some("c1".into()),
                validation_score: 0.90,
                latency_ms: 200.0,
                peak_memory_bytes: 1_500_000,
                domination_status: "non_dominated".into(),
                dominated_by: vec![],
                metadata: None,
            },
            ParetoRecord {
                candidate_id: "c3".into(),
                parent_candidate_id: Some("c1".into()),
                validation_score: 0.80,
                latency_ms: 150.0,
                peak_memory_bytes: 1_200_000,
                domination_status: "non_dominated".into(),
                dominated_by: vec![],
                metadata: None,
            },
        ];

        let frontier = compute_pareto_frontier(&records, "balanced_accuracy", "maximize", "p_test", None);
        assert_eq!(frontier.non_dominated_count, 2);
        assert_eq!(frontier.dominated_count, 1);
        assert!(frontier.frontier_candidate_ids.contains(&"c1".to_string()));
        assert!(frontier.frontier_candidate_ids.contains(&"c2".to_string()));
        assert!(!frontier.frontier_candidate_ids.contains(&"c3".to_string()));

        let c3_rec = frontier.records.iter().find(|r| r.candidate_id == "c3").unwrap();
        assert_eq!(c3_rec.domination_status, "dominated");
        assert!(c3_rec.dominated_by.contains(&"c1".to_string()));
    }

    #[test]
    fn test_pareto_frontier_computation_minimize() {
        let records = vec![
            ParetoRecord {
                candidate_id: "c1".into(),
                parent_candidate_id: None,
                validation_score: 0.40, // lower RMSE is better
                latency_ms: 100.0,
                peak_memory_bytes: 1_000_000,
                domination_status: "non_dominated".into(),
                dominated_by: vec![],
                metadata: None,
            },
            ParetoRecord {
                candidate_id: "c2".into(),
                parent_candidate_id: Some("c1".into()),
                validation_score: 0.50, // higher RMSE is worse, higher latency, higher memory
                latency_ms: 200.0,
                peak_memory_bytes: 2_000_000,
                domination_status: "non_dominated".into(),
                dominated_by: vec![],
                metadata: None,
            },
        ];

        let frontier = compute_pareto_frontier(&records, "rmse", "minimize", "p_test", None);
        assert_eq!(frontier.non_dominated_count, 1);
        assert_eq!(frontier.dominated_count, 1);
        assert_eq!(frontier.frontier_candidate_ids, vec!["c1".to_string()]);
    }

    #[test]
    fn test_hypothesis_settlement_rules() {
        let mut hyp = IterationHypothesis {
            schema_version: "v2".into(),
            hypothesis_id: "h1".into(),
            candidate_id: "c2".into(),
            parent_candidate_id: Some("c1".into()),
            mutation_type: "feature_engineering".into(),
            hypothesis_statement: "Adding polynomial features will improve balanced accuracy by at least 0.01".into(),
            rationale: None,
            expected_metric_delta: ExpectedMetricDelta {
                metric: "balanced_accuracy".into(),
                direction: "maximize".into(),
                target_threshold: Some(0.88),
                minimum_improvement: 0.01,
            },
            multi_fidelity_budget: MultiFidelityBudget::default(),
            outcome: None,
            created_ms: 1000,
        };

        // 1. Confirmed
        settle_hypothesis(&mut hyp, Some(0.89), Some(0.85), false, None);
        let out = hyp.outcome.as_ref().unwrap();
        assert_eq!(out.status, "confirmed");
        assert_eq!(out.baseline_metric, Some(0.85));
        assert_eq!(out.full_metric_observed, Some(0.89));

        // 2. Refuted due to degradation
        settle_hypothesis(&mut hyp, Some(0.82), Some(0.85), false, None);
        assert_eq!(hyp.outcome.as_ref().unwrap().status, "refuted");

        // 3. Inconclusive due to sub-threshold improvement
        settle_hypothesis(&mut hyp, Some(0.855), Some(0.85), false, None);
        assert_eq!(hyp.outcome.as_ref().unwrap().status, "inconclusive");

        // 4. Refuted due to proxy screening rejection
        settle_hypothesis(&mut hyp, Some(0.60), Some(0.85), true, Some("Proxy screening failed threshold 0.70"));
        assert_eq!(hyp.outcome.as_ref().unwrap().status, "refuted");
    }

    fn test_policy() -> crate::Policy {
        let mut b = std::collections::BTreeMap::new();
        b.insert("calls".into(), 100);
        b.insert("tokens".into(), 100_000);
        b.insert("wall_ms".into(), 1_000_000);
        let mut c = std::collections::BTreeMap::new();
        c.insert("workers".into(), 4);
        c.insert("cpu_threads".into(), 4);
        crate::Policy { max_tasks: 1024, budget: b, capacity: c }
    }

    #[test]
    fn test_hypothesis_and_pareto_sqlite_persistence() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("state");
        let proj = "p_iteration_test";

        // Initialize state DB
        crate::storage::Runtime::init(&root, test_policy()).unwrap();
        crate::execution::migrate(&root).unwrap();
        let r = crate::storage::Runtime::open(&root).unwrap();
        r.command(crate::Request {
            command_id: format!("init-{proj}"),
            project_id: proj.into(),
            expected_revision: None,
            operation: crate::Operation::CreateProject {
                intent: "P10.3 iteration test".into(),
                finalization_id: format!("final-{proj}"),
                policy: test_policy(),
            },
        }).unwrap();

        let hyp = IterationHypothesis {
            schema_version: "v2".into(),
            hypothesis_id: "h_test_1".into(),
            candidate_id: "c_cand_1".into(),
            parent_candidate_id: None,
            mutation_type: "hyperparameter_tuning".into(),
            hypothesis_statement: "Tuning LightGBM num_leaves will improve score".into(),
            rationale: Some("Higher capacity".into()),
            expected_metric_delta: ExpectedMetricDelta {
                metric: "balanced_accuracy".into(),
                direction: "maximize".into(),
                target_threshold: Some(0.85),
                minimum_improvement: 0.005,
            },
            multi_fidelity_budget: MultiFidelityBudget::default(),
            outcome: Some(HypothesisOutcome {
                status: "confirmed".into(),
                baseline_metric: Some(0.80),
                proxy_metric_observed: Some(0.82),
                full_metric_observed: Some(0.86),
                metric_delta: Some(0.06),
                reason: Some("Verified".into()),
                settled_at_ms: Some(2000),
            }),
            created_ms: 1000,
        };

        record_hypothesis(&root, proj, &hyp).unwrap();
        let fetched = get_hypothesis(&root, proj, "h_test_1").unwrap();
        assert_eq!(fetched.hypothesis_id, "h_test_1");
        assert_eq!(fetched.outcome.as_ref().unwrap().status, "confirmed");

        let listed = list_hypotheses(&root, proj, Some("c_cand_1"), None, 10).unwrap();
        assert_eq!(listed.len(), 1);

        // Record Pareto frontier
        let frontier = compute_pareto_frontier(
            &[ParetoRecord {
                candidate_id: "c_cand_1".into(),
                parent_candidate_id: None,
                validation_score: 0.86,
                latency_ms: 120.0,
                peak_memory_bytes: 1_200_000,
                domination_status: "non_dominated".into(),
                dominated_by: vec![],
                metadata: None,
            }],
            "balanced_accuracy",
            "maximize",
            proj,
            None,
        );
        record_pareto_frontier(&root, proj, &frontier).unwrap();

        let latest = get_latest_pareto_frontier(&root, proj).unwrap().unwrap();
        assert_eq!(latest.non_dominated_count, 1);
        assert_eq!(latest.frontier_candidate_ids, vec!["c_cand_1".to_string()]);
    }
}
