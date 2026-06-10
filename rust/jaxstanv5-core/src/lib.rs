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

pub mod error;
pub mod json;

pub use error::Error;
