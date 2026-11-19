# CI test policy

The compatibility workflow runs automatically for pushes to `dev` and `main`,
and pull requests **targeting** those branches. A push to another branch with no
such PR does not start a run. Updating a feature branch with an open PR into
`dev` still runs PR validation, irrespective of the source branch name.

PRs run one `data-resolution` validation job: the baseline unit/feature suite,
real wheel/sdist provenance tests, hydrated local LFS files, representative live
remote downloads, and a package build. The 18 OS/Python/OpenCV jobs run only on
pushes/merges to `dev` or `main`, after that validation job succeeds. Matrix jobs
run at most three at once and cancel remaining queued/running combinations on a
failure. New commits cancel
older runs for the same event and PR/branch. Main pushes retain the production
remote smoke test. Job permissions remain read-only.

To check a feature branch manually, use Actions → Python compatibility → Run
workflow and choose that branch. Leave `full_matrix` off for validation alone,
or enable it for all 18 combinations after validation. Manual dispatch requires
the workflow to be available on the repository's default branch.

## Build-test dependencies

Install `requirements-test.txt` before running tests locally. It includes pinned
`build`, `setuptools`, `wheel`, and `pyproject-hooks` as well as runtime/test
requirements. The shared constraints carry the same versions. Package installation
via pip uses an isolated build environment; its build dependencies do not become
available to the Python interpreter that subsequently runs pytest. The packaging
tests invoke their backend in the test interpreter, so it needs these explicit
dependencies. A local `build/` output directory is not the PyPA `build` frontend.
Missing tools fail tests instead of silently skipping packaging validation.

```sh
python -m pip install -r requirements-test.txt
python -m pip install -c requirements-constraints.txt .
python -m pip check
MPLBACKEND=Agg python -m pytest -q -k "not batch"
```

The depth-resolution regressions for issue #39 run in both baseline validation
and the compatibility matrix. Only batch examples remain deferred. Live remote
and all-download tests remain opt-in; see the instructions at the top of
`tests/test_data_integration.py`.

## Applying workflow changes

Commit and push the workflow, requirements, and tests together to the feature PR.
The updated PR run uses the revised workflow and runs only validation. Re-running
an old failed run reuses its old workflow definition and does not apply this fix.
After merging into `dev`, subsequent pushes to `dev` run the full gated matrix.
Merge the same change into `main` through the normal release process. There is
no need to pause Actions or make repository settings changes.
