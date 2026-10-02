# Direct user-tool comparison: evidence and access limits

Checked on 2026-10-02. These observations concern this audit's browser and network,
not a claim that a service is unavailable everywhere. No user dataset was uploaded,
no account was created, and no participant study was performed.

| Tool | Primary source establishes | Current direct observation | Unverified comparison |
| --- | --- | --- | --- |
| [ESurv](https://www.jmir.org/2020/5/e16084/) | User datasets, TCGA data, cut-point analysis and regularized Cox selection | Opening the published `https://easysurv.net` in the audit browser returned `net::ERR_NAME_NOT_RESOLVED` | Current functions, defaults, numerical results and task completion time |
| [surviveR](https://www.nature.com/articles/s41598-023-48894-9) | Endpoint coding, KM curves, patient filtering, continuous-variable categorization, group Cox HR and downloads | The published [service](https://generatr.qub.ac.uk/app/surviveR) redirected to a generatR Auth0 sign-in page with Email/Password and Log In | Logged-in functionality, multivariable task compatibility, defaults and task results |
| [KM Plotter custom data](https://www.jmir.org/2021/7/e27633/) | Custom-data univariate/multivariate Cox, cut-off selection and FDR | The [current form](https://kmplot.com/analysis/index.php?cancer=custom_plot&p=service) opened without sign-in. Its public sample populated time/event selectors; selecting multivariate enabled three covariate checkboxes. PH check, risk table, median, manual cutoff and optional percentile/all-value searches are visible. Median was the selected split default | Numerical output, reference coding and the proposed task's continuous/categorical covariates; no analysis was submitted. The form requires explicit acceptance of its Terms of Use before drawing a plot |
| SurvStudio | Code and browser regressions in this checkout; case-I display reproduced from verified saved analysis | Real TCGA matrix attached and saved analysis rendered with provenance in `paper/interface/` | Human usability, comparative speed and error reduction |

Published descriptions do not establish current feature absence. The comparison
matrix must distinguish **directly exercised**, **documented in the paper**, and
**not checked**. Features not reported in a paper must not be marked unsupported.

Before the human study, verify that each comparator can perform each proposed
common task and record its accessible version, options and event coding. A group
Cox hazard ratio must not be treated as a multivariable age/sex/stage Cox fit.
Restrict paired task comparisons to verified common capabilities; evaluate the
selection-procedure and locked-validation tasks separately if necessary.

P1–P3's numerical experiments address particular analysis pipelines and publication
claims. They do not replace a direct user-tool comparison or participant evidence.
The synthetic R-verified task fixtures are preparation only.

The current [KM Plotter description](https://kmplot.com/analysis/) documents BH FDR for best-cutoff selection;
the [update history](https://kmplot.com/analysis/index.php?p=updates) records it since 2018. P3 computes uncorrected
minimum log-rank p-values and a gene-count-only Bonferroni sensitivity threshold. It is a stylised pipeline and
must not be named the current KM Plotter default. This observation does not establish the service's exact numerical
implementation or calibration. The audit corrected that attribution without changing the frozen numerical run.
