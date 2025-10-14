# GitHub Actions Artifact Migration Notes

The project uses `actions/upload-artifact@v4`, which introduces immutable artifacts. Key behaviors:

- **Overwrite uploads**: set `overwrite: true` to replace an existing artifact (the workflow does this for `qlib-results`).
- **Hidden files**: `include-hidden-files: true` is enabled so cached state or dotfiles in `results/` are captured when needed.
- **Combining artifacts**: if future jobs need to merge multiple artifacts, prefer unique names per job and either download with `pattern` + `merge-multiple` or use `actions/upload-artifact/merge@v4`.
- **Matrix uploads**: give each job a unique artifact name (`name: my-artifact-${{ matrix.os }}`) and download with `pattern` to collect them.

Refer to GitHub's migration guide for further scenarios.
