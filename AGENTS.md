# Agent Instructions

Before acting, read the latest shared tensor4all agent rules from the
[`tensor4all-agent-rules`](https://github.com/tensor4all/tensor4all-agent-rules)
repository. Start from:

- `https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md`

If internet access is unavailable or the remote cannot be resolved, use the
sibling checkout:

- `../tensor4all-agent-rules/rules/index.md`

Load only the common, Rust, performance, numerical, documentation, or
provenance rule files relevant to the task. In particular,
`rules/common/provenance.md` applies whenever code is written while referencing
third-party code.

Then read [`REPOSITORY_RULES.md`](REPOSITORY_RULES.md), which contains the
durable sparse-ir-rs-specific contracts. Repository-local rules override shared
rules when they are more specific. Existing code that predates a rule is a
migration target, not a pattern to copy.

[`sparse-ir/CODING_RULES.md`](sparse-ir/CODING_RULES.md) contains narrower
implementation notes. It may provide context, but it is not
authoritative when it conflicts with `REPOSITORY_RULES.md`, current source,
public documentation, or generated binding contracts.

Before changing release automation or version metadata, read the
version-management section in [README.md](README.md). For downstream wrapper
version bumps, also read
[`bump_version_downstream.md`](bump_version_downstream.md).

Use these repo-local skills when the task matches:

- `agent-skills/semantic-version-suggestion/SKILL.md`
  Use when deciding the next sparse-ir-rs version under Semantic Versioning.
- `agent-skills/manual-rust-release/SKILL.md`
  Use when preparing or triggering the manual GitHub Actions workflow that publishes crates and pushes the release tag.

Release invariants for this repository:

- Keep `[workspace.package].version` and `[workspace.dependencies].sparse-ir.version` in [`Cargo.toml`](Cargo.toml) in sync.
- Keep [`python/pyproject.toml`](python/pyproject.toml) `[project].version` aligned with the workspace version before a Rust release.
- Run `python3 check_version.py` before any release or release PR.
- Update Julia version metadata only after the crates are published to crates.io.
- Push `vX.Y.Z` tags only after successful crates.io publication.
- The manual Rust release workflow file is `.github/workflows/manual-release.yml`.
