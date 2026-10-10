# Basic workflow

Upload a file or choose a sample cohort. Each row should describe one patient.

1. **Data** — check the uploaded rows, follow-up time, event column and value meaning the event occurred.
2. **Survival curves** — choose a group if needed, then click **Draw curves**. Curve options are available in a collapsed panel.
3. **Risk factors** — choose the variables for Cox regression and review estimates and model checks.
4. **Compare models** — evaluate statistical, machine-learning and deep-learning models using the existing shared evaluation settings.
5. **Save results** — open the file choices for the current analysis. This button does not rerun an analysis or download an older result.

Use **More analyses** to open marker analysis or a baseline-characteristics table. Closing this menu retains the dataset, settings and results. History restoration reveals the selected additional tab.

Broader time and event column lists are available under **Other columns**. Verify the event mapping before analysis; the interface does not select a model from its observed results.

This interface change preserves the numerical methods, diagnostic policies and export checks. It has been tested as software behavior, not as a human usability study.

Each analysis has a short next-step hint. Result badges say **Assumptions apply**, **Needs review** or **Caution**; they do not certify a model. A hosted page identifies its remote processing arrangement and shows a recovery message when its private server cannot be reached.
