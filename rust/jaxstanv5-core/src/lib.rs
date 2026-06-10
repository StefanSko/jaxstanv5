//! Zero-dependency sampling core for jaxstanv5.
//!
//! Parses the jaxstanv5 IR v1 wire format (`docs/ir-format-v1.md`), evaluates
//! the model log density and its gradient with built-in reverse-mode AD, and
//! samples with multinomial NUTS using Stan-style warmup adaptation.
//!
//! The library is a pure function of its arguments: no threads, no
//! filesystem, no clock, no OS entropy. Seeds are explicit arguments and
//! parallelism belongs to callers (CLI threads, web workers).

#![deny(unsafe_code)]

pub mod adapt;
pub mod density;
pub mod diagnostics;
pub mod error;
pub mod ir;
pub mod json;
pub mod linalg;
pub mod model;
pub mod nuts;
pub mod rng;
pub mod sampler;
pub mod special;
pub mod tape;
pub mod tensor;

pub use error::Error;
