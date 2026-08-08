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
rules when they are more specific. Existing code that predates a rule is not
precedent; remediation outside the current task requires the scope and approval
defined in `REPOSITORY_RULES.md`.

[`sparse-ir/CODING_RULES.md`](sparse-ir/CODING_RULES.md) contains supplemental
crate implementation notes. Read it only after the root rules, and do not treat
it as an independent policy source. `REPOSITORY_RULES.md`, current source,
public documentation, and generated binding contracts are authoritative.

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

- Follow the [Release And Version Integrity](REPOSITORY_RULES.md#release-and-version-integrity)
  rules in `REPOSITORY_RULES.md`.
