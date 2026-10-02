# SurvStudio user study: execution materials, protocol v1

Status: prepared for ethics review and pilot execution; no participants recruited and no observations collected.

## Question and comparison

Does SurvStudio reduce clinically consequential interpretation errors in common survival-analysis tasks after equal introductory training? Distinguish usability from availability of additional functionality. ESurv is the primary graphical comparator for common KM/Cox tasks (https://www.jmir.org/2020/5/e16084/). A standardized worksheet is the comparator for procedure-audit and frozen-validation tasks when the graphical comparator has no corresponding feature; report these separately rather than treating missing features as usability failures.

Unit of inference: participant. Use synthetic patient data only, with identifiers generated for the study. Freeze both tool versions, browser, task datasets, independent answer keys, instruction time, and recording rules before recruitment. Capture the comparator's actual availability and functionality during setup. Do not substitute a different comparator after viewing participant outcomes.

## Design

Engineering pilot: 6 participants, excluded from the main study. Main study: initially 48 participants, with a recruitment target of 16 novices, 16 users familiar with basic survival analysis, and 16 experienced analysts. Stratify order randomization by experience. Fix any sample-size revision after the pilot, before main enrollment, using pilot discordance and a transparent paired-error precision or power calculation. This initial target is a feasibility target, not a claim of sufficient statistical power.

Each participant receives the same 15-minute introduction per tool and four paired task families. Matched synthetic variants A/B have different patient identities and values but the same task difficulty and traps. Randomize the tool order and variant allocation in balanced blocks; conceal answers until all tasks finish. Two analysts adjudicate answers using the frozen key while blinded to tool assignment; disagreements are reconciled and retained in the audit trail. Record assistance, deviations, timeouts and incomplete tasks.

## Task cards and scoring rubric

| Family | Participant instruction | Critical errors | Independent answer key |
| --- | --- | --- | --- |
| Outcome and KM | Identify which status value represents death, estimate 24-month survival, and explain median survival and risk-table counts | Reverses event coding; treats non-estimable median as zero; interprets survival beyond supported follow-up as measured | Generate KM values and event mapping independently in R survival; record full input hashes |
| Cox interpretation | Fit age, sex and stage; explain an HR and CI, identify the reference, and assess a PH warning | Reverses HR reference; claims a CI spanning 1 proves no association; interprets a PH warning as no effect; gives a causal interpretation without justification | R coxph coefficient/reference and interval key, plus prespecified text rubric |
| Procedure audit | Compare apparent and held-out discrimination after gene selection; identify repeated patients and an outcome-derived predictor | Calls apparent C independent validation; splits repeated patient copies across partitions; accepts direct outcome leakage | Synthetic duplicates and leakage labels set by generator; held-out patient ID list sealed |
| Frozen external validation | Apply a supplied development recipe to another synthetic cohort and explain gain, calibration and uncertainty | Refits using external outcomes then calls it validation; reads an interval including zero as proof of equivalence; calls a modified implementation canonical | Frozen recipe hash, independent linear predictor calculation, and paired per-patient evaluation key |

A family is failed if any prespecified critical error occurs. An incomplete or timed-out family counts as failed in the primary intention-to-test analysis. A secondary analysis may report completed tasks separately, with the denominator explicit. Minor transcription mistakes are recorded separately. Do not add errors to the rubric after observing tool differences.

## Outcomes and analysis

Primary: participant-level difference between tools in the share of common KM/Cox families containing a critical error. Report the paired mean difference with a participant-resampling 95% interval and a participant-level permutation test swapping the two tool assignments. The additional procedure/validation families are separate secondary outcomes. They test workflow support and interpretation under the specified worksheet comparison.

Secondary: successful completion, time to correct completion, number of hints, and a validated usability questionnaire administered unchanged in its validated language after each tool. Record a task time cap of 15 minutes; report timeout share separately. Avoid analyzing successful-task times alone as if failures were random. Analyze experience interaction descriptively unless adequately powered and prespecified. Multiplicity: one primary outcome; secondary analyses receive explicit exploratory labels and effect sizes, with Holm adjustment for the four secondary error contrasts.

Use participant-level resampling, not individual task rows as independent observations. Report period/order effects and outcomes by experience group. Check whether the pilot changed instructions or interface; freeze the final task pack and hash its files before the first main participant.

## Ethics and operational checklist

Institutional review must determine the applicable approval, exemption and consent process. The consent draft should state that participation is voluntary, withdrawal is possible, synthetic data are used, screen/audio recording is optional, and results are reported in aggregate. Compensation, investigator contact, retention period and institutional privacy details must be filled in by the responsible investigator. Recruitment and participant communications have not been performed by this audit.

Use study IDs rather than names in analysis files. Store the contact list separately. Before main enrollment, complete the following: ethics determination; comparator version and accessibility check; independently computed answer keys; allocation file; consent text; pilot report; final sample-size rationale; frozen protocol/task hashes. Never fill empty observation forms with simulated participant results.

## Observation form

Required fields: study ID, experience stratum, randomized order, variant allocation, tool version, task family, start/end time, completed, timeout, hints, submitted answer, critical error codes, adjudicator decisions, protocol deviation, and optional recording consent. Questionnaire item wording and attribution must come from the validated source rather than reconstructed from memory.

## Reporting boundary

Until real participants complete the approved study, manuscript text must say that a study protocol was prepared. It cannot claim reduced errors, saved time, improved usability, or any participant-derived score.
