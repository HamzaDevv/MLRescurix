//! Short-lived WAL reads. A stream owns a cursor, never a database lease.
use crate::Result;
use rusqlite::{Connection, OpenFlags};
use std::{collections::HashMap, ops::{Deref, DerefMut}, path::{Path, PathBuf}, sync::{Arc, Mutex, OnceLock, Weak}, time::Duration};

pub const READ_CONNECTIONS: usize = 4;
#[derive(Clone)]
pub struct ReadPool { idle: Arc<Mutex<Vec<Connection>>> }
pub struct ReadLease { connection: Option<Connection>, pool: ReadPool }
impl ReadPool {
    fn open(root: &Path) -> Result<Self> {
        // Validate ownership/migrate once before opening dedicated readers.
        drop(crate::execution::connect(root)?);
        let mut connections = Vec::with_capacity(READ_CONNECTIONS);
        for _ in 0..READ_CONNECTIONS {
            let c = Connection::open_with_flags(root.join("state.sqlite"), OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX).map_err(|e| e.to_string())?;
            c.busy_timeout(Duration::ZERO).map_err(|e| e.to_string())?;
            c.execute_batch("PRAGMA query_only=ON; PRAGMA foreign_keys=ON;").map_err(|e| e.to_string())?;
            connections.push(c);
        }
        Ok(Self { idle: Arc::new(Mutex::new(connections)) })
    }
    /// No queue and no wait: callers retry on a later streaming tick.
    pub fn try_read(&self) -> Result<ReadLease> {
        let connection = self.idle.try_lock().map_err(|_| "read pool busy")?.pop().ok_or("read pool exhausted")?;
        Ok(ReadLease { connection: Some(connection), pool: self.clone() })
    }
}
impl Deref for ReadLease {
    type Target = Connection;
    fn deref(&self) -> &Connection { self.connection.as_ref().expect("live read lease") }
}
impl DerefMut for ReadLease {
    fn deref_mut(&mut self) -> &mut Connection { self.connection.as_mut().expect("live read lease") }
}
impl Drop for ReadLease {
    fn drop(&mut self) {
        if let Some(c) = self.connection.take() {
            // Defensive rollback: returned connections never keep a WAL snapshot.
            let _ = c.execute_batch("ROLLBACK");
            if let Ok(mut idle) = self.pool.idle.lock() { idle.push(c); }
        }
    }
}

static POOLS: OnceLock<Mutex<HashMap<PathBuf, Weak<ReadPool>>>> = OnceLock::new();
pub(crate) fn shared(root: &Path) -> Result<Arc<ReadPool>> {
    let canonical = std::fs::canonicalize(root).map_err(|e| e.to_string())?;
    let pools = POOLS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut pools = pools.lock().map_err(|_| "read pool registry poisoned")?;
    if let Some(pool) = pools.get(&canonical).and_then(Weak::upgrade) { return Ok(pool); }
    let pool = Arc::new(ReadPool::open(&canonical)?);
    pools.insert(canonical, Arc::downgrade(&pool));
    Ok(pool)
}
