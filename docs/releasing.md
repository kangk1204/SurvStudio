# Releasing SurvStudio

A release publishes the Python package to PyPI as `survstudio`, a container image to
`ghcr.io/kangk1204/survstudio`, and an archived copy with a DOI on Zenodo. The
`Release` workflow (`.github/workflows/release.yml`) does the first two when a GitHub
release is published; Zenodo archives the release through its GitHub integration.

## One-time setup (maintainer accounts)

1. **PyPI trusted publisher.** Sign in at pypi.org, open *Your projects → Publishing*,
   and add a pending publisher: project `survstudio`, owner `kangk1204`, repository
   `SurvStudio`, workflow `release.yml`, environment `pypi`. No API token is stored in
   GitHub.
2. **GitHub environment.** In the repository settings, create an environment named
   `pypi` (optionally requiring an approval before each upload).
3. **Zenodo.** Sign in at zenodo.org with GitHub, open *GitHub* in the account menu and
   switch the `SurvStudio` repository on. Zenodo reads `.zenodo.json` for the record's
   title, description, creators and keywords; add your ORCID and affiliation there
   before the first release if you want them on the record.
4. **Container image visibility.** After the first release, set the `survstudio`
   package on GitHub to public if the image should be pullable without signing in.

## Each release

1. Update the version in `src/survival_toolkit/__init__.py`, and `version` and
   `date-released` in `CITATION.cff`.
2. In `RELEASE_NOTES.md`, rename the *Unreleased* heading to the version and date.
3. Merge to `main` and wait for CI.
4. Create a GitHub release with the tag `vX.Y.Z` (for example `v0.3.0`) and paste the
   release notes. Publishing it starts the `Release` workflow:
   - builds the sdist and wheel, checks them with `twine`, installs the wheel and runs
     `survstudio inspect` on a small file;
   - uploads to PyPI;
   - builds the container image for amd64 and arm64 and pushes it with the version tag
     and `latest`.
5. Zenodo mints a DOI for the release. Add the DOI badge to the README and a
   `doi` field to `CITATION.cff` (the concept DOI, which always resolves to the latest
   version, is the one to cite in papers).

## Checking a release locally

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
python -m pip install dist/survstudio-*.whl
survstudio inspect examples/gbsg2_jco1994_upload_ready.csv

docker build -t survstudio .
docker run --rm -p 127.0.0.1:8000:8000 survstudio
```

## After the first release

- Register the tool at bio.tools and request an RRID (SciCrunch) so papers can cite it
  by identifier.
- Add GitHub topics (survival-analysis, biomarkers, bioinformatics, cox-regression,
  kaplan-meier) so the repository is found by search.
