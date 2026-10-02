# Audited case-I interface capture

`markers_tab_case_study_i.png` was captured from the actual application with the real
TCGA matrix attached. It displays the saved, verified case-I result computed at
`3e0c4af142e7afc005f3a9bfb28e23c99c203ff7`, rendered with the interpretation and
plotting code at `6d1bb277482b4ba5c0f79d450fed381469b88095`. The analysis was not
recomputed in the browser. The screenshot has not been retouched.

`markers_tab_case_study_i.provenance.json` records the numerical result's hash and
original stamp, the rendering commit and source hashes, the actual attached matrix
and request configuration, the displayed text, and the viewport and device scale.
It does not stamp the old numerical analysis as a result of the newer display code.

`capture_case_i_20261002.py` preserves this audit's capture procedure. Run it with
Playwright installed, the rendering commit on `PYTHONPATH`, and two arguments: a
checkout containing the verified case-I result and real matrix, and an output
directory. Its assertions deliberately require the audited result and defaults.
It is not a replacement for rerunning the numerical paper pipeline.

The original release-candidate screenshot remains available in Git at `f7d8baa1`.
Raster screenshot text needs visual inspection at the proposed printed size; the
matplotlib figure guard cannot verify fonts inside the screenshot.
