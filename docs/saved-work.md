# Saved work

SurvStudio automatically saves uploaded tables, attached marker matrices and completed analysis results
on the computer running the analysis server. **Saved work** reopens a saved analysis after a page reload,
an idle period or a server restart. Reopening uses the saved numbers and plots; it does not train a model again.

The in-memory cache remains limited. Removing an item from that cache does not delete its saved copy.
Only an explicit delete removes saved work. Changing the inputs or settings still makes an older result
out of date; reopening it does not remove input warnings or release a withheld conclusion.
Deleting an external validation table also removes saved validation results and saved screens that
refer to that table. Other completed analyses in the development workspace remain available.

## Choose a storage location

The default location is a private directory in your user profile. To choose a permanent location:

```bash
survstudio serve --state-dir /path/to/private-survstudio-data
```

Server operators can set `SURVSTUDIO_STATE_DIR` instead. Use one running server per directory.
Keep the directory private and on a local disk. It contains patient data and may contain fitted models.
Do not put it in a public Git repository or a shared website folder.

For Docker, keep a named volume across container replacements:

```bash
docker run --rm -p 127.0.0.1:8000:8000 -v survstudio-data:/home/survstudio/saved-work survstudio
```

## What is preserved

- Uploaded tables, column types, patient identifiers, metadata and attached marker matrices.
- Completed result values, plots, settings and analysis status, including unavailable values and reasons for review.
- The browser workspace saved after a completed analysis, including model comparison results.
- Supported fitted tree-model artifacts used by follow-up prediction views. Completed DL results are saved;
  a reusable trained neural network is not currently stored.

An interrupted calculation has no completed result to restore. Results that existed only in an older
browser session before this feature was installed cannot be recovered from the server automatically.
Keep that page open until the results have been exported or saved with the updated interface.

## Failures and backups

A failed disk write must show an error rather than a saved confirmation. Free disk space and retry the
save; keep the current page open meanwhile. Do not delete the storage directory to bypass a corruption,
owner or dependency-version error. Preserve a copy and inspect the cause first.

Automatic saving protects against cache expiry and ordinary process restarts. It does not replace an
independent backup or protect against disk failure. Stop the server before copying the entire private
storage directory; preserve the signing key and database together. Restore under the same account and
compatible Python/scientific package versions. Test the copy before removing the original.

For a deliberately temporary session, set `SURVSTUDIO_AUTOSAVE=off` before starting the server.
That mode does not preserve new work after the process stops.
