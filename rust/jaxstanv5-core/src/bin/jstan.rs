//! `jstan` — subprocess sampling protocol over the jaxstanv5 IR.
//!
//! Usage:
//!   jstan sample --model <ir.json|-> --data <data.json> [--seed N]
//!       [--chains C] [--warmup W] [--draws D] [--max-treedepth T]
//!       [--target-accept A]
//!
//! Output is NDJSON on stdout: a header object (draws_format
//! "v0-provisional"), one object per draw with constrained values keyed by
//! parameter, and a final trailer object with per-chain diagnostics plus
//! cross-chain R-hat/ESS. Errors are a single JSON object on stderr with a
//! nonzero exit code; messages state what to change.
//!
//! Parallelism lives here, not in the library: one thread per chain.

use std::io::Read;
use std::io::Write;

use jaxstanv5_core::diagnostics;
use jaxstanv5_core::error::{Error, ErrorKind};
use jaxstanv5_core::ir::decode_model;
use jaxstanv5_core::json::{self, Value};
use jaxstanv5_core::model::{data_from_json, Posterior};
use jaxstanv5_core::sampler::{sample, ChainDraws, Settings};

/// One constrained draw: (name, shape, values) per parameter.
type ConstrainedDraw = Vec<(String, Vec<usize>, Vec<f64>)>;

fn usage_error(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidSettings, message)
}

struct Args {
    model_path: String,
    data_path: String,
    seed: u64,
    chains: u64,
    settings: Settings,
}

fn parse_args(argv: &[String]) -> Result<Args, Error> {
    if argv.is_empty() || argv[0] != "sample" {
        return Err(usage_error(
            "usage: jstan sample --model <ir.json|-> --data <data.json> [--seed N] \
             [--chains C] [--warmup W] [--draws D] [--max-treedepth T] [--target-accept A]",
        ));
    }
    let mut model_path: Option<String> = None;
    let mut data_path: Option<String> = None;
    let mut seed = 0u64;
    let mut chains = 4u64;
    let mut settings = Settings::default();

    let mut iter = argv[1..].iter();
    while let Some(flag) = iter.next() {
        let mut value_for = |name: &str| -> Result<&String, Error> {
            iter.next()
                .ok_or_else(|| usage_error(format!("flag {name} needs a value")))
        };
        match flag.as_str() {
            "--model" => model_path = Some(value_for("--model")?.clone()),
            "--data" => data_path = Some(value_for("--data")?.clone()),
            "--seed" => {
                seed = value_for("--seed")?
                    .parse()
                    .map_err(|_| usage_error("--seed must be an unsigned integer"))?
            }
            "--chains" => {
                chains = value_for("--chains")?
                    .parse()
                    .map_err(|_| usage_error("--chains must be a positive integer"))?
            }
            "--warmup" => {
                settings.num_warmup = value_for("--warmup")?
                    .parse()
                    .map_err(|_| usage_error("--warmup must be a non-negative integer"))?
            }
            "--draws" => {
                settings.num_draws = value_for("--draws")?
                    .parse()
                    .map_err(|_| usage_error("--draws must be a positive integer"))?
            }
            "--max-treedepth" => {
                settings.max_treedepth = value_for("--max-treedepth")?
                    .parse()
                    .map_err(|_| usage_error("--max-treedepth must be a positive integer"))?
            }
            "--target-accept" => {
                settings.target_accept = value_for("--target-accept")?
                    .parse()
                    .map_err(|_| usage_error("--target-accept must be a number in (0, 1)"))?
            }
            other => {
                return Err(usage_error(format!(
                    "unknown flag {other}; see `jstan sample` usage"
                )))
            }
        }
    }
    let model_path =
        model_path.ok_or_else(|| usage_error("--model is required (a path or - for stdin)"))?;
    let data_path = data_path.ok_or_else(|| usage_error("--data is required (a path)"))?;
    if chains == 0 {
        return Err(usage_error("--chains must be at least 1"));
    }
    Ok(Args {
        model_path,
        data_path,
        seed,
        chains,
        settings,
    })
}

fn read_input(path: &str) -> Result<String, Error> {
    if path == "-" {
        let mut buffer = String::new();
        std::io::stdin()
            .read_to_string(&mut buffer)
            .map_err(|e| usage_error(format!("cannot read stdin: {e}")))?;
        Ok(buffer)
    } else {
        std::fs::read_to_string(path)
            .map_err(|e| usage_error(format!("cannot read \"{path}\": {e}")))
    }
}

/// Constrained parameter value as JSON: scalar or rank-1 array (the core
/// profile has no higher-rank parameters).
fn tensor_to_value(shape: &[usize], data: &[f64]) -> Value {
    if shape.is_empty() {
        Value::Float(data[0])
    } else {
        Value::Array(data.iter().map(|&v| Value::Float(v)).collect())
    }
}

fn header_line(args: &Args, packing: &[(String, Vec<usize>)]) -> Value {
    Value::Object(vec![
        (
            "draws_format".to_string(),
            Value::Str("v0-provisional".to_string()),
        ),
        (
            "params".to_string(),
            Value::Array(
                packing
                    .iter()
                    .map(|(name, shape)| {
                        Value::Object(vec![
                            ("name".to_string(), Value::Str(name.clone())),
                            (
                                "shape".to_string(),
                                Value::Array(shape.iter().map(|&d| Value::Int(d as i64)).collect()),
                            ),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "packing".to_string(),
            Value::Array(
                packing
                    .iter()
                    .map(|(name, _)| Value::Str(name.clone()))
                    .collect(),
            ),
        ),
        (
            "settings".to_string(),
            Value::Object(vec![
                (
                    "num_warmup".to_string(),
                    Value::Int(args.settings.num_warmup as i64),
                ),
                (
                    "num_draws".to_string(),
                    Value::Int(args.settings.num_draws as i64),
                ),
                (
                    "max_treedepth".to_string(),
                    Value::Int(args.settings.max_treedepth as i64),
                ),
                (
                    "target_accept".to_string(),
                    Value::Float(args.settings.target_accept),
                ),
            ]),
        ),
        ("seed".to_string(), Value::Int(args.seed as i64)),
        ("chains".to_string(), Value::Int(args.chains as i64)),
    ])
}

fn run() -> Result<(), Error> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = parse_args(&argv)?;

    let model_doc = json::parse(&read_input(&args.model_path)?)?;
    let meta = decode_model(&model_doc)?;
    let data_doc = json::parse(&read_input(&args.data_path)?)?;
    let data = data_from_json(&data_doc)?;
    let posterior = Posterior::new(meta, data)?;

    let packing = posterior.packing();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    writeln!(out, "{}", json::write(&header_line(&args, &packing))?).map_err(io_error)?;

    // One thread per chain; the library itself stays single-threaded.
    let results: Vec<Result<ChainDraws, Error>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..args.chains)
            .map(|chain_id| {
                let posterior = &posterior;
                let settings = &args.settings;
                let seed = args.seed;
                scope.spawn(move || sample(posterior, settings, seed, chain_id))
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("chain thread panicked"))
            .collect()
    });
    let mut chains: Vec<ChainDraws> = Vec::with_capacity(results.len());
    for result in results {
        chains.push(result?);
    }

    // Constrained draws, precomputed once per draw (also reused for the
    // trailer's cross-chain diagnostics).
    let mut constrained_chains: Vec<Vec<ConstrainedDraw>> = Vec::new();
    for chain in &chains {
        let mut constrained_draws = Vec::with_capacity(chain.draws.len());
        for q in &chain.draws {
            let constrained = posterior.constrain(q)?;
            constrained_draws.push(
                constrained
                    .into_iter()
                    .map(|(name, tensor)| (name, tensor.shape().to_vec(), tensor.data().to_vec()))
                    .collect::<ConstrainedDraw>(),
            );
        }
        constrained_chains.push(constrained_draws);
    }

    for (chain_id, draws) in constrained_chains.iter().enumerate() {
        for (draw_id, constrained) in draws.iter().enumerate() {
            let values = Value::Object(
                constrained
                    .iter()
                    .map(|(name, shape, data)| (name.clone(), tensor_to_value(shape, data)))
                    .collect(),
            );
            let line = Value::Object(vec![
                ("chain".to_string(), Value::Int(chain_id as i64)),
                ("draw".to_string(), Value::Int(draw_id as i64)),
                ("values".to_string(), values),
            ]);
            writeln!(out, "{}", json::write(&line)?).map_err(io_error)?;
        }
    }

    // Trailer: per-chain stats plus cross-chain R-hat / ESS per parameter
    // (max over coordinates for R-hat, min for ESS, matching
    // jaxstanv5.diagnostics conventions).
    let mut rhat_entries = Vec::new();
    let mut ess_entries = Vec::new();
    for (param_idx, (name, shape)) in packing.iter().enumerate() {
        let size: usize = shape.iter().product::<usize>().max(1);
        let mut worst_rhat = f64::NEG_INFINITY;
        let mut worst_ess = f64::INFINITY;
        for coord in 0..size {
            let series: Vec<Vec<f64>> = constrained_chains
                .iter()
                .map(|draws| {
                    draws
                        .iter()
                        .map(|constrained| constrained[param_idx].2[coord])
                        .collect()
                })
                .collect();
            worst_rhat = worst_rhat.max(diagnostics::split_rhat(&series));
            worst_ess = worst_ess.min(diagnostics::effective_sample_size(&series));
        }
        rhat_entries.push((name.clone(), Value::Float(worst_rhat)));
        ess_entries.push((name.clone(), Value::Float(worst_ess)));
    }
    let chain_stats = Value::Array(
        chains
            .iter()
            .enumerate()
            .map(|(chain_id, chain)| {
                Value::Object(vec![
                    ("chain".to_string(), Value::Int(chain_id as i64)),
                    (
                        "divergences".to_string(),
                        Value::Int(chain.divergences as i64),
                    ),
                    (
                        "treedepth_histogram".to_string(),
                        Value::Array(
                            chain
                                .treedepth_histogram
                                .iter()
                                .map(|&c| Value::Int(c as i64))
                                .collect(),
                        ),
                    ),
                    ("step_size".to_string(), Value::Float(chain.step_size)),
                    ("mean_accept".to_string(), Value::Float(chain.mean_accept)),
                ])
            })
            .collect(),
    );
    let trailer = Value::Object(vec![(
        "trailer".to_string(),
        Value::Object(vec![
            ("chains".to_string(), chain_stats),
            ("rhat".to_string(), Value::Object(rhat_entries)),
            ("ess".to_string(), Value::Object(ess_entries)),
        ]),
    )]);
    writeln!(out, "{}", json::write(&trailer)?).map_err(io_error)?;
    Ok(())
}

fn io_error(e: std::io::Error) -> Error {
    Error::new(
        ErrorKind::InvalidSettings,
        format!("cannot write output: {e}"),
    )
}

fn main() {
    if let Err(error) = run() {
        let payload = Value::Object(vec![
            (
                "error".to_string(),
                Value::Str(error.kind.name().to_string()),
            ),
            ("message".to_string(), Value::Str(error.message.clone())),
        ]);
        let text =
            json::write(&payload).unwrap_or_else(|_| "{\"error\":\"InvalidSettings\"}".to_string());
        eprintln!("{text}");
        std::process::exit(1);
    }
}
