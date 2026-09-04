# AGENTS.md

This file provides guidance to AI coding agents working with code in this
repository.

## Project

**sparse-ir-rs** is the Rust core of the sparse-ir software stack: the
`sparse-ir` crate implements the sparse intermediate representation (IR) for
quantum many-body physics (basis construction, sampling, DLR), and the
`sparse-ir-capi` crate exposes that core through a C ABI distributed as the
`libsparseir` C library. Python, Julia, and Fortran bindings in other
repositories, and in this repository's `python/`, `julia/`, and `fortran/`
directories, wrap that C API.

## Shared Rules

Read the SpM-lab shared agent rules before making changes:
<https://github.com/SpM-lab/spm-agent-rules> — start at
[`rules/index.md`](https://github.com/SpM-lab/spm-agent-rules/blob/main/rules/index.md)
and load only the rule files the current task needs. If internet access is
unavailable, look for a sibling checkout at `../spm-agent-rules`.

For this repository, the routing table in `rules/index.md` typically resolves
to:

- `common.md` — for any implementation work.
- `ffi-boundary.md` — whenever a change touches `sparse-ir-capi`, an
  `extern "C"` function, a raw pointer, a dtype conversion, memory order, or a
  status code.
- `numerical-conventions.md` — whenever a change touches basis construction,
  sampling, DLR, statistics (fermionic/bosonic), `tau`/Matsubara domains, or
  real/complex handling.
- `testing.md` — for any new test or change to existing test coverage.
- `rust.md` — the sparse-ir-rs- and libsparseir-specific rules.

## Repository-Specific Rules

See [`REPOSITORY_RULES.md`](REPOSITORY_RULES.md) for the durable,
sparse-ir-rs-specific facts and contracts (workspace layout, build and test
commands, the C API surface, CI entry points).

## Precedence

Repository-local rules in `REPOSITORY_RULES.md` override the shared rules in
`spm-agent-rules` when they are more specific. Note any such override in the
pull request description when it affects review.

## Other Repository-Specific Pointers

These are not agent rules, only orientation to existing repository assets:

- Release/version-management workflow: the "Version management" section of
  [`README.md`](README.md), [`check_version.py`](check_version.py), and
  [`bump_version_downstream.md`](bump_version_downstream.md).
- Repo-local skills: [`agent-skills/semantic-version-suggestion/SKILL.md`](agent-skills/semantic-version-suggestion/SKILL.md)
  and [`agent-skills/manual-rust-release/SKILL.md`](agent-skills/manual-rust-release/SKILL.md).
- Supplemental crate implementation notes: [`sparse-ir/CODING_RULES.md`](sparse-ir/CODING_RULES.md)
  (read only after the rules above; not an independent policy source).
