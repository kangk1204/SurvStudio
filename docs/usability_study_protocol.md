# SurvStudio usability study: protocol (draft)

A small formative usability study for the software paper. It measures whether researchers
who analyse survival data can finish typical analyses in SurvStudio without help, how long
they take, and how usable they find the tool. The findings feed a list of fixes before
release, and the summary statistics go into the paper.

## Objectives

1. Effectiveness: the share of participants who complete each core task correctly without
   assistance.
2. Efficiency: time on task.
3. Satisfaction: System Usability Scale (SUS) score, and the Single Ease Question after each
   task.
4. Problems: usability problems observed or reported, rated by severity.

## Design

Single-session, moderated, think-aloud study with a within-participant task set. No
comparison tool: the SUS score is interpreted against its published benchmark (average
68; Sauro & Lewis 2016), and task success against a target of 80% for each core task.
Sessions run in person or by video call and take about 60 minutes.

## Participants

- 8 to 12 adults who analyse or interpret survival data at least occasionally. Five users
  find most usability problems of a design (Nielsen & Landauer 1993), and 10 to 12 give
  a usable SUS estimate.
- Mix of roles, about equal: clinicians or clinical researchers; biostatisticians or
  bioinformaticians; graduate students in a biomedical field.
- Exclusions: SurvStudio developers and anyone who has used SurvStudio before.
- Recruitment by invitation within the department and collaborating groups; participation is
  voluntary and has no bearing on employment or grades.

## Ethics

Obtain the institution's IRB determination (exempt or approved) before recruiting.
Participants give informed consent, including separate consent for screen and audio
recording. Only public data sets are used; no participant data are analysed. Recordings
and notes carry a study ID, stay on an encrypted institutional drive, and are deleted after
analysis; the consent forms are stored separately.

## Materials

- SurvStudio at a fixed version (recorded in the results), running locally on the
  participant's or a provided computer, opened on the start page.
- Task sheets with the scenario for each task, and a data file for task 4: the GBSG2 cohort
  with the event coded as text ("recurrence" or "censored") and one extra ID column.
- A facilitator script (introduction, think-aloud practice, neutral prompts).
- The SUS questionnaire (10 items), the Single Ease Question (7-point), and a short
  background form (role, years analysing survival data, tools used).

## Tasks

| # | Task | Correct when the participant |
|---|------|--------------------------|
| 1 | Load the breast cancer sample and compare recurrence-free survival by hormone therapy | reports the log-rank p-value shown for `horTh` |
| 2 | Build a Table 1 by hormone therapy and export it to Excel | produces the XLSX file |
| 3 | Fit a Cox model with age, tumour size, grade and positive nodes | reports the hazard ratio of `pnodes` with its 95% CI, and says whether the proportional-hazards check flags a term |
| 4 | Upload the provided file and set the outcome | sets time = `rfs_days` and the event value to "recurrence" |
| 5 | Evaluate `pnodes`, `progrec` and `estrec` as prognostic markers adjusted for age and grade, and export the REMARK checklist | names the robust markers and saves the checklist |
| 6 | Compare all prediction models and say which one ranks first | names the top model and its C-index |
| 7 | On the design-check page, enter a study that picked the best of 101 models on one cohort of 150 patients, with training C-index in the choice | reports the expected optimism and names at least two flagged problems |

Tasks 1 to 5 are the core tasks; 6 and 7 are secondary. Task order is fixed (later
tasks reuse loaded data); a participant who cannot finish a task in 10 minutes is shown
the solution and moves on (counted as a failure).

## Measures

- Task outcome: success, partial success (correct with one hint or a minor error), or
  failure; hints given; time from reading the task to the answer.
- Single Ease Question after each task.
- SUS after the last task.
- Problems: each observed or reported problem with the task, the screen, what happened, and
  a severity rating (0 = not a problem to 4 = catastrophe; Nielsen 1994) agreed by two
  raters.
- A 5-minute closing interview: what was confusing, what was missing, whether they would
  use SurvStudio for their own work.

## Analysis

- Success rate per task with a Wilson 95% interval; median and range of time on task.
- SUS: mean with a 95% t-interval, compared with 68 descriptively; adjective rating
  (Bangor et al. 2009).
- Problems grouped by screen and theme, counted by severity; every severity 3 or 4 problem
  goes on the fix list with an owner.
- Results are reported by role group only descriptively (the groups are too small to test).
- The paper reports the participants' background, the task table, success rates, times,
  SUS, and the main problems with what was changed in response.

## Procedure (60 minutes)

1. Welcome, consent, background form (5 min).
2. Think-aloud practice on an unrelated website (3 min).
3. Tasks 1 to 7 with the Single Ease Question after each (40 min).
4. SUS and closing interview (10 min).
5. Debrief (2 min).

## Pilot

Run the whole protocol with one or two colleagues first to check the timing, the task
wording and the data file; pilot sessions are not analysed.

## References

- Bangor A, Kortum P, Miller J. Determining what individual SUS scores mean: adding an
  adjective rating scale. J Usability Stud 2009;4:114-23.
- Nielsen J. Severity ratings for usability problems. Nielsen Norman Group, 1994.
- Nielsen J, Landauer TK. A mathematical model of the finding of usability problems.
  Proc INTERCHI 1993:206-13.
- Sauro J, Lewis JR. Quantifying the User Experience. 2nd ed. Morgan Kaufmann, 2016.
