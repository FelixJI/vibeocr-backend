# Contributing

Open an issue before large changes. Keep generated files reproducible, run the repository CI commands, and never commit local path or editable dependencies.

## Commits and changelog

Use Conventional Commit titles and squash merge pull requests so the final
commit is the release-note source of truth.

- Release notes include `feat`, `fix`, `perf`, `security`, `deps`,
  `build`, and `revert`.
- Repository-only maintenance types `docs`, `refactor`, `test`, `ci`,
  `style`, and `chore` are intentionally hidden.
- Choose the type that describes the actual change. Do not relabel maintenance
  work merely to make it appear in the changelog.
