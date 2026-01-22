use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand::seq::SliceRandom;
use rayon::prelude::*;
use regex::Regex;
use serde::Serialize;
use once_cell::sync::Lazy;
use std::{
    collections::{HashSet},
    env,
    error::Error,
    ffi::{OsStr, OsString},
    fmt,
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    process::{self, Command},
    sync::{Arc, Mutex},
};
use tempfile::NamedTempFile;
use walkdir::WalkDir;
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum Severity {
    Low,
    Medium,
    High,
}
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum MutatorKind {
    Syntax,
    Semantic,
    Ast,
    Noise,
}
type MutFn = fn(&str, &mut StdRng) -> Option<Mutate>;
#[derive(Debug, Clone)]
struct MutatorSpec {
    id: &'static str,
    label: &'static str,
    kind: MutatorKind,
    severity: Severity,
    func: MutFn,
    default_weight: f32,
}
#[derive(Serialize)]
struct Sample {
    broken: String,
    fixed: String,
    explanation: String,
    mutator_name: String,
    mutator_kind: MutatorKind,
    severity: Severity,
    file_path: String,
    seed: u64,
    chain_len: usize,
    diff: String,
}
#[derive(Debug, Clone)]
struct Config {
    output_prefix: PathBuf,
    inputs: Vec<PathBuf>,
    seed: Option<u64>,
    per_file: usize,
    max_chain: usize,
    severity_filter: Option<Severity>,
    mutator_include: Option<HashSet<String>>,
    mutator_exclude: HashSet<String>,
    validate: bool,
    dry_run: bool,
    shard_size: usize,
    noisy: bool,
}
#[derive(Debug, Clone)]
enum AppError {
    Usage(String),
    Io { ctx: String, err: String },
    Json { ctx: String, err: String },
    #[allow(dead_code)]
    Regex(String),
    Internal(String),
}
#[allow(dead_code)]
impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::Usage(msg) => write!(f, "{msg}"),
            AppError::Io { ctx, err } => write!(f, "{ctx}: {err}"),
            AppError::Json { ctx, err } => write!(f, "{ctx}: {err}"),
            AppError::Regex(msg) => write!(f, "regex error: {msg}"),
            AppError::Internal(msg) => write!(f, "internal error: {msg}"),
        }
    }
}
impl Error for AppError {}
fn main() {
    if let Err(e) = run() {
        eprintln!("fatal: {e}");
        process::exit(1);
    }
 }
 fn run() -> Result<(), AppError> {
    let cfg = parse_args(env::args_os().collect())?;
    let base_seed = init_seed(cfg.seed)?;
    eprintln!("info: output prefix = '{}'", cfg.output_prefix.to_string_lossy());
    eprintln!("info: per-file samples = '{}'", cfg.per_file);
    eprintln!("info: max chain length = {}", cfg.max_chain);
    eprintln!("info: shard size = {}", cfg.shard_size);
    eprintln!("info: validation = {}", cfg.validate);
    eprintln!("info: dry_run = {}", cfg.dry_run);
    eprintln!("info: noisy mode = {}", cfg.noisy);
    let files = collect_rs_files(&cfg.inputs)?;
    if files.is_empty() {
        return Err(AppError::Usage(
            "no .rs files found in the provided inputs.".to_string(),
        ));
    }
if !cfg.dry_run {
    let first = output_shard_path(&cfg.output_prefix, 1);
    if first.exists() {
        return Err(AppError::Usage(format!(
            "refusing to overwrite existing output shard '{}'",
            first.to_string_lossy()
        )));
    }
}
let mutators = build_mutators(&cfg)?;
let writer_state = Arc::new(Mutex::new(ShardWriter::new(&cfg)?));
files.par_iter().enumerate().try_for_each(|(idx, path)| {
    process_file(
        &cfg,
        path,
        &mutators,
        base_seed.wrapping_add(idx as u64),
        &writer_state,
    )
})?;
Ok(())
}
fn usage() -> String {
    "Usage: datasetgenv2 [options] <output_prefix> <input> [input2 ...]
    Options:
    --seed <u64>  RNG seed (overridden by datasetgenv2 env var).
    --per-file <N>  How samples emit per source file (default 6).
    --max-chain <N>  Max number of chained mutations per sample (default 3).
    --severity <low|medium|high> Restrict to mutations of this severity or lower.
    --mutator include=<a,b> Only use these mutators (comma-separated-names).
    --mutator exclude=<a,b> Exclude these mutators.
    --validate Only keep samples that still parse and typecheck (rustc).
    --dry_run Do not write output, just log what would happen.
    --shard-size <N> Max Samples per output shard (default 50000).
    --noisy Enable noisy mode mutations.
    - <input> may be a .rs file or a directory (search recursively).\n\
    - Output will be '<output_prefix>_00001.jsonl', '<output_prefix>_00002.jsonl', etc."
        .to_string()
}
fn parse_args(args_os: Vec<OsString>) -> Result<Config, AppError> {
    let mut it = args_os.into_iter();
    let _exe = it.next();
    let mut seed: Option<u64> = None;
    let mut per_file: usize = 6;
    let mut max_chain: usize = 3;
    let mut severity_filter: Option<Severity> = None;
    let mut mutator_include: Option<HashSet<String>> = None;
    let mut mutator_exclude: HashSet<String> = HashSet::new();
    let mut validate = false;
    let mut dry_run = false;
    let mut shard_size: usize = 50_000;
    let mut noisy = false;
    let mut positionals: Vec<PathBuf> = Vec::new();
    while let Some(arg) = it.next() {
        let s = arg.to_string_lossy();
        if s == "--seed" {
            let val = it
            .next()
            .ok_or_else(|| AppError::Usage("missing value for --seed".to_string()))?;
        let parsed: u64 = val
            .to_string_lossy()
            .parse()
            .map_err(|_| AppError::Usage("invalid value for --seed".to_string()))?;
    seed = Some(parsed);
        } else if let Some(rest) = s.strip_prefix("--seed=") {
            let parsed: u64 = rest
            .parse()
            .map_err(|_| AppError::Usage("invalid value for --seed".to_string()))?;
    seed = Some(parsed);
        } else if s == "--per-file" {
            let val = it
            .next()
            .ok_or_else(|| AppError::Usage("missing value for --per-file".to_string()))?;
    let parsed: usize = val
    .to_string_lossy()
    .parse()
    .map_err(|_| AppError::Usage("invalid value for --per-file".to_string()))?;
    per_file = parsed.max(1);
        } else if let Some(rest) = s.strip_prefix("--per-file=") {
            let parsed: usize = rest
            .parse()
             .map_err(|_| AppError::Usage("invalid value for --per-file".to_string()))?;
        per_file = parsed.max(1);
        } else if s == "--max-chain" {
            let val = it
            .next()
            .ok_or_else(|| AppError::Usage("missing value for --max-chain".to_string()))?;
        let parsed: usize = val
        .to_string_lossy()
        .parse()
        .map_err(|_| AppError::Usage("invalid value for max-chain".to_string()))?;
    max_chain = parsed.clamp(1, 16);
        } else if let Some(rest) = s.strip_prefix("--severity=") {
        severity_filter = Some(parse_severity(rest)?);
    }
         else if s == "--severity" {
            let val = it
            .next()
            .ok_or_else(|| AppError::Usage("missing value for --severity".to_string()))?;
        severity_filter = Some(parse_severity(val.to_string_lossy().as_ref())?);
        } else if s.starts_with("--mutator") {
            let rest = if s == "--mutator" {
            it.next()
            .ok_or_else(|| AppError::Usage("missing value for mutator".to_string()))?
        .to_string_lossy()
        .to_string()
        } else {
            s["--mutator".len()..].trim_start_matches('=').to_string()
        };
    if let Some(rest) = rest.strip_prefix("include=") {
        let set = mutator_include.get_or_insert_with(HashSet::new);
        for name in rest.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
            set.insert(name.to_string());
        }
    }  else if let Some(rest) = rest.strip_prefix("exclude=") {
        for name in rest.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
            mutator_exclude.insert(name.to_string());
        }
    }
        } else if s == "--validate" {
            validate = true;
        } else if s == "--dry_run" || s == "--dry-run" {
            dry_run = true;
        } else if s == "--shard-size" {
            let val = it.next().ok_or_else(|| {
                AppError::Usage("missing value for --shard-size".to_string())
            })?;
            let parsed: usize = val
            .to_string_lossy()
            .parse()
            .map_err(|_| AppError::Usage("invalid value for --shard-size".to_string()))?;
        shard_size = parsed.max(1);
        } else if s == "--noisy" {
            noisy = true;
        } else if s.starts_with("--") {
            return Err(AppError::Usage(format!("unknown flag {}", s)));
        } else {
            positionals.push(PathBuf::from(arg));
        } 
    }
    if positionals.len() < 2 {
        return Err(AppError::Usage(usage()));
    }
    let output_prefix = positionals.remove(0);
    let prefix_ext_ok = output_prefix
        .extension()
        .and_then(OsStr::to_str)
        .map(|ext| ext.eq_ignore_ascii_case("jsonl"))
        .unwrap_or(false);
if prefix_ext_ok {
    return Err(AppError::Usage(format!(
        "policy: output prefix must not have .jsonl extension. got '{}'",
        output_prefix.to_string_lossy()
    )));
}
let cfg = Config {
    output_prefix,
    inputs: positionals,
    seed,
    per_file,
    max_chain,
    severity_filter,
    mutator_include,
    mutator_exclude,
    validate,
    dry_run,
    shard_size,
    noisy
    };
    Ok(cfg)
        }
fn parse_severity(s: &str) -> Result<Severity, AppError> {
    match s.to_ascii_lowercase().as_str() {
        "low" => Ok(Severity::Low),
        "medium" => Ok(Severity::Medium),
        "high" => Ok(Severity::High),
        other => Err(AppError::Usage(format!(
            "invalid severity '{other}' (expected low|medium|high)"
        ))),
}
}
fn init_seed(seed: Option<u64>) -> Result<u64, AppError> {
    if let Some(s) = seed {
        eprintln!("info: using provided seed {}", s);
    return Ok(s)
    }
    if let Ok(s) = env::var("datasetgenv2") {
        let parsed: u64 = s.parse().map_err(|_| {
            AppError::Usage("invalid datasetgenv2 env value (expected u64)".to_string()) 
        })?;
        eprintln!("info: using env seed {}", parsed);
        return Ok(parsed);
    }
    Ok(StdRng::from_entropy().r#gen())
}
fn collect_rs_files(inputs: &[PathBuf]) -> Result<Vec<PathBuf>, AppError> {
    let mut out = Vec::new();
    for p in inputs {
           let meta = fs::metadata(p).map_err(|e| AppError::Io {
            ctx: format!("failed to stat input '{}'", p.to_string_lossy()),
            err: e.to_string(),
        })?;
        if meta.is_dir() {
            for entry in WalkDir::new(p)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
            {
        let path = entry.path();
        if path.is_file()
        && path.extension().and_then(OsStr::to_str) == Some("rs")
        {
    out.push(path.to_path_buf());
    }
}
}
    else if meta.is_file() {
    if p.extension().and_then(OsStr::to_str) == Some("rs") {
        out.push(p.clone());
    } else {
        return Err(AppError::Usage(format!(
            "only .rs inputs are allowed as files (or provide directories) invalid: '{}'",
            p.to_string_lossy()
        )));
    }
}
    }       
Ok(out)
}

struct Mutate {
    broken: String,
    explanation: String,
}
fn build_mutators(cfg: &Config) -> Result<Vec<MutatorSpec>, AppError> {
    let mut all: Vec<MutatorSpec> = vec![
        MutatorSpec {
            id: "remove_semicolon",
            label: "remove semicolon",
            kind: MutatorKind::Syntax,
            severity: Severity::Low,
            func: mutate_remove_random_semicolon,
            default_weight: 1.0,
        },
        MutatorSpec {
            id: "remove_last_brace",
            label: "remove last brace",
            kind: MutatorKind::Syntax,
            severity: Severity::Medium,
            func: mutate_remove_last_closing_brace,
            default_weight: 0.6,
        },
        MutatorSpec {
            id: "keyword_fn_to_fun",
            label: "keyword fn to fun",
            kind: MutatorKind::Syntax,
            severity: Severity::Low,
            func: mutate_keyword_fn_to_fun_random,
            default_weight: 0.8,
        },
        MutatorSpec {
            id: "insert_stray_token",
            label: "insert stray token",
            kind: MutatorKind::Syntax,
            severity: Severity::Medium,
            func: mutate_insert_stray_token_random,
            default_weight: 0.7,
        },
        MutatorSpec {
            id: "duplicate_line",
            label: "duplicate line",
            kind: MutatorKind::Syntax,
            severity: Severity::Low,
            func: mutate_duplicate_line_random,
            default_weight: 1.0,
        },
        MutatorSpec {
            id: "flip_boolean_literal",
            label: "flip boolean literal",
            kind: MutatorKind::Semantic,
            severity: Severity::Low,
            func: mutate_flip_boolean_literal_random,
            default_weight: 1.0,
        },
        MutatorSpec {
            id: "flip_equality",
            label: "flip equality",
            kind: MutatorKind::Semantic,
            severity: Severity::Low,
            func: mutate_flip_equality_random,
            default_weight: 0.9,
        },
        MutatorSpec {
            id: "comment_out_return",
            label: "comment out return",
            kind: MutatorKind::Semantic,
            severity: Severity::Medium,
            func: mutate_comment_out_return_random,
            default_weight: 0.9,
        },
        MutatorSpec {
            id: "flip_plus_minus",
            label: "flip plus minus",
            kind: MutatorKind::Semantic,
            severity: Severity::Medium,
            func: mutate_flip_plus_minus_random,
            default_weight: 0.9,
        },
        MutatorSpec{
            id: "change_numeric_literal",
            label: "change numeric literal",
            kind: MutatorKind::Semantic,
            severity: Severity::Low,
            func: mutate_change_numeric_literal_random,
            default_weight: 1.0,
        },
        MutatorSpec {
            id: "ast_swap_first_two_fns",
            label: "ast swap first two fns",
            kind: MutatorKind::Ast,
            severity: Severity::Medium,
            func: mutate_ast_swap_first_two_fns,
            default_weight: 0.5,
        },
    ];
    if cfg.noisy {
        all.push(MutatorSpec {
            id: "noise_insert_blank_lines",
            label: "noise insert blank lines",
            kind: MutatorKind::Noise,
            severity: Severity::High,
            func: mutate_noise_insert_blank_lines,
            default_weight: 0.7,
        });
    }
    if let Some(limit) = cfg.severity_filter {
        all.retain(|m| severity_len(m.severity, limit));
    }
    if let Some(include) = &cfg.mutator_include {
    all.retain(|m| include.iter().any(|s| s.eq_ignore_ascii_case(m.id)));
    }
    if !cfg.mutator_exclude.is_empty() {
        all.retain(|m| !cfg.mutator_exclude.iter().any(|s| s.eq_ignore_ascii_case(m.id)));
    }
    if all.is_empty() {
        return Err(AppError::Usage(
            "no mutators left after applying filters."
            .to_string()
        ));
    }
    Ok(all)
}
fn severity_len(a: Severity, b: Severity) -> bool {
    use Severity::*;
    match (a, b) {
        (Low, _) => true,
        (Medium, Medium | High) => true,
        (High, High) => true,
        _ => false,
    }
}
fn process_file(
    cfg: &Config,
    path: &Path,
    mutators: &[MutatorSpec],
    file_seed: u64,
    writer_state: &Arc<Mutex<ShardWriter>>,
) -> Result<(), AppError> {
    let display_path = path.to_string_lossy().to_string();
    eprintln!("info: processing '{}'", display_path);
    let code = fs::read_to_string(path).map_err(|e| AppError::Io {
        ctx: format!("failed to read '{}'", display_path),
        err: e.to_string(),
    })?;

let mut rng = StdRng::seed_from_u64(file_seed);
let mut samples: Vec<Sample> = Vec::with_capacity(cfg.per_file);
let mut seen_broken: HashSet<u64> = HashSet::new();
let max_attempts = cfg.per_file.saturating_mul(40).max(80);
let mut attempts = 0;
while samples.len() < cfg.per_file && attempts < max_attempts {
    attempts += 1;
    if let Some(sample) =
    generate_one_sample(cfg, &code, &display_path, mutators, &mut rng, file_seed)
    {
        if seen_broken.insert(hash_str(&sample.broken)) {
            samples.push(sample);
        }
        }
    }
if samples.is_empty() {
    eprintln!(
        "warning: could not generate samples for '{}' (falling back to trivial injection)",
        display_path
    );
    let trivial_broken = format!("???;\n{}", (code));
    let diff = compute_diff(&code, &trivial_broken);
    samples.push(Sample {
        broken: trivial_broken,
        fixed: code.clone(),
        explanation: "remove the stray token '???' from the source.".to_string(),
        mutator_name: "fallback_trivial".to_string(),
        mutator_kind: MutatorKind::Syntax,
        severity: Severity::Low,
        file_path: display_path.clone(),
        seed: file_seed,
        chain_len: 1,
        diff,
    });
}
if cfg.dry_run {
    for s in &samples {
        eprintln!(
            "dry-run: file='{}' mutator='{} severity={:?} chain_len={} explanation={}",
            s.file_path, s.mutator_name, s.severity, s.chain_len, s.explanation
        );
    }
    return Ok(());
}
let mut guard = writer_state
    .lock()
    .map_err(|_| AppError::Internal("writer mutex poisoned".to_string()))?; 
for s in samples {
    guard.write_sample(&s)?;
    }
Ok(())
}
fn generate_one_sample(
    cfg: &Config,
    code: &str,
    file_path: &str,
    mutators: &[MutatorSpec],
    rng: &mut StdRng,
    file_seed: u64
) -> Option<Sample> {
    if mutators.is_empty() {
        return None;
    }
    let mut total_weight = 0.0f32;
    let mut cumulative: Vec<(f32, usize)> = Vec::with_capacity(mutators.len());
    for (i, m) in mutators.iter().enumerate() {
        total_weight += m.default_weight.max(0.0);
        cumulative.push((total_weight, i));
    }
    if total_weight <= 0.0 {
        return None;
    }
    let target_chain_len = if cfg.max_chain <= 1 {
        1
    }
    else {
       rng.gen_range(1..=cfg.max_chain) 
    }; 
    let mut current_code = code.to_string();
    let mut explanation: Vec<String> = Vec::new();
    let mut last_mutator: Option<&MutatorSpec> = None;
    let mut applied_chain_len: usize = 0;
    for _ in 0..target_chain_len {
        let r: f32 = rng.r#gen::<f32>() * total_weight;
        let mut picked_idx: usize = 0;
        for (cum_w, idx) in &cumulative {
            if r <= *cum_w {
                picked_idx = *idx;
                break;
            }
        }
        let spec = &mutators[picked_idx];
        if let Some(m) = (spec.func)(&current_code, rng) {
            current_code = m.broken;
            explanation.push(format!("[{}] {}", spec.id, m.explanation));
            last_mutator = Some(spec);
            applied_chain_len += 1;
        }
    }
    let final_mutator = last_mutator?;
    if applied_chain_len == 0 {
        return None;
    }

    let explanation = explanation.join(" | ");
    let diff = compute_diff(code, &current_code);
    if cfg.validate && !validate_rust(&current_code) {
        return None;
    }
    Some(Sample {
        broken: current_code,
        fixed: code.to_string(),
        explanation,
        mutator_name: final_mutator.label.to_string(),
        mutator_kind: final_mutator.kind,
        severity: final_mutator.severity,
        file_path: file_path.to_string(),
        seed: file_seed,
        chain_len: applied_chain_len,
        diff,
    }) 
}
fn compute_diff(original: &str, mutated: &str) -> String {
    use similar::{ChangeTag, TextDiff};
    let diff = TextDiff::from_lines(original, mutated);
    let mut out = String::new();
    for change in diff.iter_all_changes() {
        let sign = match change.tag() {
            ChangeTag::Delete => '-',
            ChangeTag::Insert => '+',
            ChangeTag::Equal => ' ',
        };
        out.push(sign);
        out.push(' ');
        out.push_str(change.value());
    }
    out
}
fn hash_str(s: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}
fn validate_rust(code: &str) -> bool {
    let mut tmp = match NamedTempFile::new() {
        Ok(f) => f,
        Err(_) => return false
    };
    if tmp.write_all(code.as_bytes()).is_err() {
        return false;
    }
    let path = tmp.path();
    let status = Command::new("rustc")
    .arg("--edition")
    .arg("2021")
    .arg("--crate-type")
    .arg("lib")
    .arg("--emit")
    .arg("metadata")
    .arg(path)
    .status();
match status {
    Ok(s) => s.success(),
    Err(_) => false,
}
}
struct ShardWriter {
    cfg: Config,
    current_index: usize,
    current_count: usize,
    writer: Option<BufWriter<std::fs::File>>,
}
impl ShardWriter {
    fn new(cfg: &Config) -> Result<Self, AppError> {
        let mut s = Self {
            cfg: cfg.clone(),
            current_index: 0,
            current_count: 0,
            writer: None,
        };
        s.rotate_shard()?;
        Ok(s)
    }
fn rotate_shard(&mut self) -> Result<(), AppError> {
    if let Some(w) = self.writer.take() {
        let mut w = w;
        w.flush().map_err(|e| AppError::Io {
            ctx: "flush error on shard rotation".to_string(),
            err: e.to_string(),
        })?;
    }
    self.current_index += 1;
    self.current_count = 0;
    let shard_path = output_shard_path(&self.cfg.output_prefix, self.current_index);
    if shard_path.exists() {
        return Err(AppError::Usage(format!(
            "refusing to overwrite shard '{}'",
            shard_path.to_string_lossy()
        )));
    }
    let file = OpenOptions::new()
    .write(true)
    .create_new(true)
    .open(&shard_path)
    .map_err(|e| AppError::Io {
        ctx: format!("failed to open output shard '{}'", shard_path.to_string_lossy()),
        err: e.to_string(),
    })?;
    eprintln!("info: writing shard '{}'", shard_path.to_string_lossy());
    self.writer = Some(BufWriter::new(file));
    Ok(())
}
fn write_sample(&mut self, sample: &Sample) -> Result<(), AppError> {
    if self.current_count >= self.cfg.shard_size {
        self.rotate_shard()?;
    }
    let w = self
    .writer
    .as_mut()
    .ok_or_else(|| AppError::Internal("no active writer".to_string()))?;
    serde_json::to_writer(&mut *w, sample).map_err(|e| AppError::Json {
        ctx: "json serialization failed".to_string(),
        err: e.to_string()
    })?;
    w.write_all(b"\n").map_err(|e| AppError::Io {
        ctx: "failed to write newline".to_string(),
        err: e.to_string(),
    })?;
    self.current_count += 1;
    Ok(())
    }
}
fn output_shard_path(prefix: &Path, index: usize) -> PathBuf {
    let stem = prefix
    .file_stem()
    .and_then(OsStr::to_str)
    .unwrap_or("output");
let ext = prefix.extension().and_then(OsStr::to_str).unwrap_or("jsonl");
let shard_name = format!("{stem}_{:05}.{ext}", index);
prefix.with_file_name(shard_name)
    }
fn mutate_remove_random_semicolon(code: &str, rng: &mut StdRng) -> Option<Mutate> {
        let positionals: Vec<usize> = code
        .match_indices(';')
    .map(|(i, _)| i)
    .collect();
if positionals.is_empty() {
    return None;
}
let idx = positionals[rng.gen_range(0..positionals.len())];
let mut broken = String::with_capacity(code.len().saturating_sub(1));
broken.push_str(&code[..idx]);
broken.push_str(&code[idx + 1..]);
Some(Mutate {
    broken, 
    explanation: "Reinsert the missing ';' that was removed from the source."
    .to_string(),
})
    }
fn mutate_remove_last_closing_brace(code: &str, _rng: &mut StdRng) -> Option<Mutate> {
    let pos = code.rfind('}')?;
    let mut broken = String::with_capacity(code.len().saturating_sub(1));
    broken.push_str(&code[..pos]);
    broken.push_str(&code[pos + 1..]);
    Some(Mutate {
        broken,
        explanation: "Add back the missing closing '}' brace near the end of the file."
        .to_string(),
    })
} 
fn mutate_keyword_fn_to_fun_random(code: &str, rng: &mut StdRng) -> Option<Mutate> {
static FN_RE: Lazy<Regex> = 
Lazy::new(|| Regex::new(r"\bfn\b").unwrap());
let matches: Vec<_> = FN_RE.find_iter(code).collect();
 if matches.is_empty() {
        return None;
    }
    let m = matches[rng.gen_range(0..matches.len())];
    let start = m.start();
    let end = m.end();
    let mut broken = String::with_capacity(code.len() + 1);
    broken.push_str(&code[..start]);
    broken.push_str("fun");
    broken.push_str(&code[end..]);
    Some(Mutate {
        broken,
        explanation: "Replace the incorrect keyword 'fun' with 'fn'."
        .to_string(),
    })
}
fn mutate_insert_stray_token_random(code: &str, _rng: &mut StdRng) -> Option<Mutate> {
    if code.starts_with("???") {
        return None;
    }
    let broken = format!("???;\n{code}");
    Some(Mutate {
        broken,
        explanation: "Remove stray token '???;' that was injected at the top of the file."
        .to_string(),
    })
}
fn mutate_duplicate_line_random(code: &str, rng: &mut StdRng) -> Option<Mutate> {
    let lines: Vec<&str> = code.lines().collect();
    if lines.len() < 2 {
        return None;
    }
    let ends_with_newline = code.as_bytes().last() == Some(&b'\n');
    let idx = rng.gen_range(0..lines.len());
    let mut broken = String::with_capacity(code.len() + lines[idx].len() + 1);
    for (i, line) in lines.iter().enumerate() {
        broken.push_str(line);
        broken.push('\n');
        if i == idx {
            broken.push_str(line);
            broken.push('\n');
        }
    }
          if !ends_with_newline && broken.ends_with('\n') {
          broken.pop();
    }
    Some(Mutate {
        broken,
        explanation: "Remove the duplicated line introduced by mutation."
        .to_string(),
    })
}
fn mutate_flip_boolean_literal_random(code: &str, rng: &mut StdRng) -> Option<Mutate> {
    let re = Regex::new(r"\b(true|false)\b").ok()?;
    let matches: Vec<_> = re.find_iter(code).collect();
    if matches.is_empty() {
        return None;
    }
    let m = matches[rng.gen_range(0..matches.len())];
    let lit = &code[m.start()..m.end()];
    let other = if lit == "true" {  "false" } else { "true"};
    let mut broken = String::with_capacity(code.len());
    broken.push_str(&code[..m.start()]);
    broken.push_str(other);
    broken.push_str(&code[m.end()..]);
    Some(Mutate {
        broken,
        explanation: "Restore the original literal value."
        .to_string(),
    })
}
fn mutate_flip_equality_random(code: &str, rng: &mut StdRng) -> Option<Mutate> {
    let re = Regex::new(r"(==|!=)").ok()?;
    let matches: Vec<_> = re.find_iter(code).collect();
    if matches.is_empty() {
        return None;
    }
    let m = matches[rng.gen_range(0..matches.len())];
    let op = &code[m.start()..m.end()];
    let replacement = if op == "==" { "!=" } else {"=="};
    let mut broken = String::with_capacity(code.len());
    broken.push_str(&code[..m.start()]);
    broken.push_str(replacement);
    broken.push_str(&code[m.end()..]);
    Some(Mutate {
        broken,
        explanation: "Restore the correct equality operator. (== or !=)."
        .to_string(),
    })
}
fn mutate_comment_out_return_random(code: &str, rng: &mut StdRng) -> Option<Mutate> {
    let lines: Vec<&str> = code.lines().collect();
    let candidate_idxs: Vec<usize> = lines
    .iter()
    .enumerate()
    .filter_map(|(i, line)| {
        let trimmed = line.trim_start();
        if trimmed.starts_with("return ") || trimmed.starts_with("return;") {
            Some(i)
        } else {
            None
        }
    })
    .collect();
if candidate_idxs.is_empty() {
    return None;
    }
    let idx = candidate_idxs[rng.gen_range(0..candidate_idxs.len())];
    let mut broken = String::with_capacity(code.len() + 3);
    for (i, line) in lines.iter().enumerate() {
        if i == idx {
            broken.push_str("//");
            broken.push_str(line);
        } else {
            broken.push_str(line);
        }
        broken.push('\n');
    }
    Some(Mutate {
        broken, explanation:
        "Uncomment the return statement that was commented out."
        .to_string(),
    })
}
fn mutate_flip_plus_minus_random(code: &str, rng: &mut StdRng) -> Option<Mutate> {
    static RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?m)\b([A-Za-z_]\w*)\s*([+-])\s*([A-Za-z_]\w*)\b").unwrap()
    });
    let matches: Vec<_> = RE.captures_iter(code).collect();
    if matches.is_empty() {
        return None;
    }
    let cap = &matches[rng.gen_range(0..matches.len())];
    let op = cap.get(2).unwrap();
    let old = op.as_str();
    let new = if old == "+" { "-" } else { "+" };
    let mut broken = String::with_capacity(code.len());
    broken.push_str(&code[..op.start()]);
    broken.push_str(new);
    broken.push_str(&code[op.end()..]);
    Some(Mutate {
          broken, 
          explanation: format!("flip '{new}' back to '{old}' in the arithmetic expression."),
    })
}
fn mutate_change_numeric_literal_random(code: &str, rng: &mut StdRng) -> Option<Mutate> {
                let re = Regex::new(r"\b\d+\b").ok()?;
                let nums: Vec<_> = re.find_iter(code).collect();
                if nums.is_empty() {
                    return None;
                }
                let m = nums[rng.gen_range(0..nums.len())];
                let lit = &code[m.start()..m.end()];
                let val: i128 = lit.parse().ok()?;
                let delta: i128 = if rng.gen_bool(0.5) { 1 } else { -1 };
                let new_val = val.saturating_add(delta);
                let new_str = new_val.to_string();
                let mut broken = String::with_capacity(code.len() - lit.len() + new_str.len());
                broken.push_str(&code[..m.start()]);
                broken.push_str(&new_str);
                broken.push_str(&code[m.end()..]);
                Some(Mutate {
                    broken,
                    explanation: format!("Revert numeric literal {new_val} back to {val}."),
                })
            }
fn mutate_ast_swap_first_two_fns(code: &str, _rng: &mut StdRng) -> Option<Mutate> {
    use syn::{File, Item};
    let file: File = syn::parse_file(code).ok()?;
    let mut fns: Vec<(usize, &syn::ItemFn)> = Vec::new();
    for (idx, item) in file.items.iter().enumerate() {
        if let Item::Fn(func) = item {
            fns.push((idx, func));
        }
    }
    if fns.len() < 2 {
        return None;
    }
    let (i1, _f1) = fns[0];
    let (i2, _f2) = fns[1];
    let mut new_items = file.items.clone();
    new_items.swap(i1, i2);
    let new_file = File {
        shebang: file.shebang.clone(),
        attrs: file.attrs.clone(),
        items: new_items,
    };
  let broken = prettyplease::unparse(&new_file);
    Some(Mutate {
        broken,
        explanation: "Restore the original order of the first two function definitions."
        .to_string(),
    })
}
fn mutate_noise_insert_blank_lines(code: &str, rng: &mut StdRng) -> Option<Mutate> {
    let lines: Vec<&str> = code.lines().collect();
    if lines.is_empty() {
        return None;
    }
    let inserts = rng.gen_range(1..=lines.len());
    let mut positions: Vec<usize> = (0..lines.len()).collect();
    positions.shuffle(rng);
    positions.truncate(inserts);
    let pos_set: HashSet<usize> = positions.into_iter().collect();
    let mut broken = String::with_capacity(code.len() + inserts * 2);
    for (idx, line) in lines.iter().enumerate() {
        broken.push_str(line);
        broken.push('\n');
        if pos_set.contains(&idx) {
            broken.push('\n');
        }
    }
    Some(Mutate {
        broken,
        explanation: "Remove the extra blank lines from the source."
        .to_string()
    })
}