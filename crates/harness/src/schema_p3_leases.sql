CREATE TABLE IF NOT EXISTS resource_leases (
    lease_id TEXT NOT NULL PRIMARY KEY,
    project TEXT NOT NULL REFERENCES projects(id),
    task TEXT NOT NULL,
    job TEXT NOT NULL,
    generation INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('reserved', 'active', 'heartbeated', 'recovering', 'reaped', 'released')),
    backend TEXT NOT NULL,
    device TEXT NOT NULL,
    cpu_millicores INTEGER NOT NULL,
    library_threads INTEGER NOT NULL,
    memory_bytes INTEGER NOT NULL,
    accelerator_type TEXT,
    accelerator_units INTEGER NOT NULL DEFAULT 0,
    scratch_bytes INTEGER NOT NULL,
    output_bytes INTEGER NOT NULL,
    pid_cap INTEGER NOT NULL DEFAULT 64,
    deadline_ms INTEGER NOT NULL,
    heartbeat_ms INTEGER NOT NULL,
    environment_hash TEXT,
    created_ms INTEGER NOT NULL,
    updated_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS resource_leases_project_status ON resource_leases(project, status);
CREATE INDEX IF NOT EXISTS resource_leases_job ON resource_leases(project, job);

CREATE TABLE IF NOT EXISTS resource_lease_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lease_id TEXT NOT NULL REFERENCES resource_leases(lease_id),
    project TEXT NOT NULL REFERENCES projects(id),
    job TEXT NOT NULL,
    from_status TEXT,
    to_status TEXT NOT NULL,
    reason TEXT,
    transitioned_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS lease_transitions_lease ON resource_lease_transitions(lease_id, transitioned_ms);
CREATE INDEX IF NOT EXISTS lease_transitions_project ON resource_lease_transitions(project, transitioned_ms);
