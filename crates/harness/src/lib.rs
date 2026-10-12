pub mod contracts;
pub mod core;
pub mod storage;
pub use core::*;
pub use storage::Runtime;
pub type Result<T> = std::result::Result<T, String>;
