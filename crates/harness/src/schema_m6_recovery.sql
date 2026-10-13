-- Never prune these identities while their provider turn can be retried.
ALTER TABLE provider_turn_receipts ADD COLUMN tool_protocol INTEGER NOT NULL DEFAULT 0 CHECK(tool_protocol IN(0,1));
CREATE TABLE tool_outcomes(
 project TEXT NOT NULL, task TEXT NOT NULL, turn INTEGER NOT NULL, call_id TEXT NOT NULL,
 action TEXT NOT NULL, branch TEXT NOT NULL, name TEXT NOT NULL, arguments TEXT NOT NULL,
 intent_json TEXT NOT NULL, outcome TEXT, outcome_sha256 TEXT,
 PRIMARY KEY(project,task,turn,call_id),
 FOREIGN KEY(project,task,turn) REFERENCES provider_turn_receipts(project,task,turn),
 FOREIGN KEY(project,action) REFERENCES actions(project,id),
 CHECK((outcome IS NULL) = (outcome_sha256 IS NULL))
);
PRAGMA user_version=9;
