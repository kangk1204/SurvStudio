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

## Native summary component for Figure 1

`marker_summary_case_study_i.png` is a separate native capture of the application's
summary panel at a 900-pixel viewport and device scale 3, rendered with
`01d8956f2a83ceade9eaf9a67a5b020d6a394901`. It uses the same saved numerical analysis
and freshly attached real matrix. Numbers and labels were visually checked for
clipping and overlap. It has not been retouched or cropped from the full screenshot.

`marker_summary_case_study_i.provenance.json` additionally records the PNG's hash,
the executed capture helper's hash, the component's CSS width (786 px), and its
smallest rendered SVG font (12 px). At the figure script's final 168.5-mm component
width this corresponds to 7.29 pt, above our project's 7-pt readability floor.
That floor is a project decision, not an explicit journal font-size requirement.
The figure script refuses a changed PNG, a smaller effective font, or a figure
over 200 mm high, reserving 25 mm of the 225-mm combined limit for its legend.

Use `capture_case_i_print_20261002.py` with the same two arguments and rendering
source on `PYTHONPATH` to reproduce this component. Its default viewport is 900 px.
The capture output `case_i_summary.png` and `provenance.json` are stored here under
the descriptive names above. Figure 1 uses the component; the full interface
capture remains available for examining the rest of the workflow.
