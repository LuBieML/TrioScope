# Releasing TrioScope

TrioScope uses Semantic Versioning, Conventional Commit pull-request titles,
and Release Please. `src/version.py` is the application-facing version; do not
edit it manually during normal development.

## Version rules

- `fix:` increments the patch version.
- `feat:` increments the minor version.
- A `!` after the type/scope or a `BREAKING CHANGE:` footer increments the
  major version. Before 1.0, Release Please keeps breaking releases in the
  0.x series unless explicitly overridden.
- `docs:`, `test:`, `build:`, `ci:`, `chore:`, and `refactor:` are included in
  release notes where applicable but do not normally trigger a release.

Examples:

```text
fix(scope): correct trigger timing
feat(ui): add 3D path hover
feat!: change the saved profile format
```

Use squash merging so the pull-request title becomes the commit Release Please
analyses on `main`. The `PR Title` workflow validates the title format.

## Normal release flow

1. Merge conventional feature/fix pull requests into `main`.
2. Release Please opens or updates one release pull request. It updates
   `CHANGELOG.md`, `.release-please-manifest.json`, and `src/version.py`.
3. Review and merge that pull request when the release is ready.
4. Release Please creates an immutable `vMAJOR.MINOR.PATCH` tag and GitHub
   Release. Never move or reuse a release tag.
5. If automated packaging is enabled, the tagged source is tested, packaged,
   checksummed, and attached to the GitHub Release.

To force a particular next version, include `Release-As: 1.2.3` in the body of
a conventional commit. Use this sparingly, for example when declaring 1.0.0.

## Repository setup

The release workflow works with `GITHUB_TOKEN`. For CI to run on Release
Please's generated pull request, configure `RELEASE_PLEASE_TOKEN` as a fine-
grained personal-access token or GitHub App token with repository contents and
pull-request write access.

For automatic Windows artifacts, configure a restricted self-hosted runner:

1. Use Windows x64 with Python 3.13, GitHub CLI, and the licensed Trio Unified
   API files installed.
2. Add the custom runner label `trioscope-release`.
3. Set the repository Actions variable `TRIOSCOPE_SDK_DIR` to the directory
   containing `Trio_UnifiedApi*.pyd`, `Trio_UnifiedApi_PCMCAT.dll`, and
   `Trio_UnifiedApi_TCP.dll`.
4. Set `TRIOSCOPE_ENABLE_RELEASE_BUILD` to `true`.

Until that variable is enabled, versioning, changelog, tags, and GitHub Releases
remain automatic, while the proprietary binary build is skipped. A release can
be rebuilt from the Actions page with **Build Windows Release → Run workflow**
and an existing tag such as `v0.1.0`.

## Local checks

```powershell
python scripts/check_release_version.py v0.0.2
$env:QT_QPA_PLATFORM = "offscreen"
python -m pytest tests/ -q
python build_exe.py
```

`build_exe.py` reads the same source version, discovers the SDK from
`TRIOSCOPE_SDK_DIR` or the active Python environment, and embeds Windows
`FileVersion` and `ProductVersion` metadata into the executable.
