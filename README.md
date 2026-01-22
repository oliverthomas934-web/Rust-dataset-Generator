DatasetGen v2 — Mini Rust Dataset Generator

DatasetGen v2.exe is a lightweight dataset generator written in and for Rust that produces small, high-quality code-repair datasets by applying weighted, controlled mutations to Rust source code.

It works by taking valid Rust files, deliberately introducing realistic errors, and emitting paired (broken → fixed) samples as JSONL shards. These datasets are designed for tools and models that need to understand or explain code at a small scale.

The generator can run deterministically using a fixed seed, or non-deterministically by allowing it to generate its own seed on each run.

The tool processes Rust source files and outputs JSONL shards, where each line represents one training or testing sample. Each sample includes:

 A broken version of the code
 The corrected version
 Metadata describing how and why the mutation occurred

This format is well-suited for:

 Compiler and static-analysis tooling
 Small language model (SLM) training
 Code-repair and explanation datasets
 Mutation testing and fuzzing research

Mutation Types

DatasetGen v2 supports four categories of mutations:

1. Syntax errors

    Missing semicolons
    Unbalanced braces
    Stray or invalid tokens

2. Semantic changes

    Boolean literal flips
    Operator substitutions
    Numeric value edits

3. AST-level transformations

    Reordering of syntactic structures

4. Optional noise mutations (disabled by default)

   Formatting damage
   Cosmetic or non-semantic corruption

Each line in the output file is a single JSON object:
{
  "broken": "...",
  "fixed": "...",
  "explanation": "...",
  "mutator_name": "...",
  "mutator_kind": "syntax|semantic|ast|noise",
  "severity": "low|medium|high",
  "file_path": "...",
  "seed": 123456,
  "chain_len": 3,
  "diff": "..."
}

This structure makes the output easy to stream, shard, and consume in training or analysis pipelines.

Platform Support

DatasetGen v2 requires PowerShell and therefore works on:

 Windows
 Linux
 macOS

As long as PowerShell is available on the system, the tool will run correctly.

Example Usage

Standard run

powershell
datasetgenv2.exe `
  --noisy `
  --per-file 100 `
  --max-chain 6 `
  --shard-size 100 `
  out `
  .\src

Flags explained:

 `--noisy` – Enables noise-only mutations (disabled by default)
 `--per-file` – Target number of mutation attempts per input file
 `--max-chain` – Maximum number of mutations applied per sample
 `--shard-size` – Number of samples per output shard



Dry run (no output written)

powershell
datasetgenv2.exe --dry-run out .\src

Restricting Mutators

You can control which mutations are allowed by filtering on severity or specific mutators.

Example

powershell
datasetgenv2.exe `
  --severity=medium `
  --mutator include=remove_semicolon,flip_boolean_literal `
  --mutator exclude=flip_boolean_literal `
  out `
  .\src