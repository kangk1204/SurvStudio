# Numerical agreement with R, lifelines and scikit-survival

Generated 2026-09-30 by `validation/agreement/run_agreement.py` on the bundled GBSG2 and TCGA-LUAD cohorts.
R survival is the reference; the difference columns give the absolute difference from R. Empty cells: the package
does not report that quantity (or the package is not installed).

| Component | Version |
|---|---|
| SurvStudio | 0.3.0 |
| Python | 3.11.15 |
| numpy | 2.4.6 |
| statsmodels | 0.15.0 |
| R | R version 4.5.2 (2025-10-31) ; survival 3.8.6 |
| lifelines | 0.30.0 |
| scikit-survival | 0.28.0 |
| Platform | Linux aarch64 |

## Summary

| Analysis | Quantities | Largest difference from R: SurvStudio | lifelines | scikit-survival |
|---|---|---|---|---|
| GBSG2: Kaplan-Meier by hormone therapy | 29 | 4.32e-12 | 4.77e-12 | 8.88e-14 |
| TCGA-LUAD: Kaplan-Meier by stage | 57 | 5.68e-14 | 1.49e-13 | 1.07e-13 |
| GBSG2: Cox model | 31 | 3.27e-13 | 1.81e-05 | 2.64e-14 |
| GBSG2: Cox model stratified by menopausal status | 15 | 3.64e-12 | 1.47e-05 |  |
| TCGA-LUAD: Cox model (complete cases) | 31 | 9.02e-13 | 3.93e-09 | 2.62e-15 |
| GBSG2: marker score tests (marker engine) | 8 | 1.8e-07 |  |  |
| GBSG2: Cox fit of the marker engine (stratified by menopausal status) | 18 | 2.27e-12 |  |  |

Notes:

- Kaplan-Meier intervals are pointwise log-log intervals (R `conf.type = "log-log"`); the median's interval is
  where those bands cross 0.5.
- When the curve stays exactly at 0.5 over an interval, SurvStudio and R report its midpoint as the median and
  lifelines its first time; this does not happen in these cohorts.
- The proportional-hazards statistics are the classic Grambsch-Therneau tests on the Schoenfeld residuals with log
  time, which SurvStudio reports; the R values apply those formulas to `resid(fit, "schoenfeld")` (the newer
  `cox.zph` test differs), and lifelines' per-term test uses the same formulas.
- lifelines stops its Newton-Raphson iterations at a looser tolerance, so its Cox estimates differ from R's in the
  fifth or sixth significant digit.
- The RMST standard error is compared with R only: lifelines returns the variance of the restricted survival time
  itself, not of the RMST estimate.
- The marker score tests are compared with R's `coxph(..., init = ..., iter.max = 0)` score statistics; the
  marker engine's Cox fit is compared for Efron and Breslow ties.

## GBSG2: Kaplan-Meier by hormone therapy

| Quantity | SurvStudio | R | difference | lifelines | difference | scikit-survival | difference |
|---|---|---|---|---|---|---|---|
| no @ 365 survival | 0.896619 | 0.896619 | 4.44e-16 | 0.896619 | 4.44e-16 | 0.896619 | 4.44e-16 |
| no @ 365 CI lower | 0.863582 | 0.863582 | 0 | 0.863582 | 0 | 0.863582 | 0 |
| no @ 365 CI upper | 0.922018 | 0.922018 | 0 | 0.922018 | 0 | 0.922018 | 0 |
| no @ 1095 survival | 0.605801 | 0.605801 | 4.44e-16 | 0.605801 | 1.11e-16 | 0.605801 | 4.44e-16 |
| no @ 1095 CI lower | 0.555423 | 0.555423 | 0 | 0.555423 | 2.22e-16 | 0.555423 | 0 |
| no @ 1095 CI upper | 0.652333 | 0.652333 | 2.22e-16 | 0.652333 | 2.22e-16 | 0.652333 | 2.22e-16 |
| no @ 1825 survival | 0.436806 | 0.436806 | 3.33e-16 | 0.436806 | 2.22e-16 | 0.436806 | 3.89e-16 |
| no @ 1825 CI lower | 0.37792 | 0.37792 | 4.44e-16 | 0.37792 | 9.44e-16 | 0.37792 | 3.89e-16 |
| no @ 1825 CI upper | 0.494104 | 0.494104 | 2.78e-16 | 0.494104 | 9.44e-16 | 0.494104 | 2.22e-16 |
| yes @ 365 survival | 0.949584 | 0.949584 | 2.22e-16 | 0.949584 | 5.55e-16 | 0.949584 | 1.11e-16 |
| yes @ 365 CI lower | 0.912924 | 0.912924 | 2.22e-16 | 0.912924 | 7.77e-16 | 0.912924 | 3.33e-16 |
| yes @ 365 CI upper | 0.971053 | 0.971053 | 3.33e-16 | 0.971053 | 9.99e-16 | 0.971053 | 4.44e-16 |
| yes @ 1095 survival | 0.707733 | 0.707733 | 2.22e-16 | 0.707733 | 4.44e-16 | 0.707733 | 3.33e-16 |
| yes @ 1095 CI lower | 0.643235 | 0.643235 | 3.33e-16 | 0.643235 | 2.22e-16 | 0.643235 | 2.22e-16 |
| yes @ 1095 CI upper | 0.76275 | 0.76275 | 1.11e-16 | 0.76275 | 1.11e-16 | 0.76275 | 0 |
| yes @ 1825 survival | 0.58121 | 0.58121 | 6.66e-16 | 0.58121 | 5.55e-16 | 0.58121 | 4.44e-16 |
| yes @ 1825 CI lower | 0.506789 | 0.506789 | 1.11e-16 | 0.506789 | 8.88e-16 | 0.506789 | 0 |
| yes @ 1825 CI upper | 0.648399 | 0.648399 | 5.55e-16 | 0.648399 | 7.77e-16 | 0.648399 | 3.33e-16 |
| no median | 1528 | 1528 | 0 | 1528 | 0 |  |  |
| no median CI lower | 1280 | 1280 | 0 | 1280 | 0 |  |  |
| no median CI upper | 1814 | 1814 | 0 | 1814 | 0 |  |  |
| no RMST | 1537.47 | 1537.47 | 1.14e-12 | 1537.47 | 9.09e-13 |  |  |
| no RMST SE | 49.253 | 49.253 | 1.42e-14 |  |  |  |  |
| yes median | 2018 | 2018 | 0 | 2018 | 0 |  |  |
| yes median CI lower | 1918 | 1918 | 0 | 1918 | 0 |  |  |
| yes median CI upper |  |  |  |  |  |  |  |
| yes RMST | 1774.45 | 1774.45 | 4.32e-12 | 1774.45 | 4.77e-12 |  |  |
| yes RMST SE | 61.6359 | 61.6359 | 3.55e-14 |  |  |  |  |
| log-rank chi-square | 8.56478 | 8.56478 | 7.28e-14 | 8.56478 | 7.82e-14 | 8.56478 | 8.88e-14 |

## TCGA-LUAD: Kaplan-Meier by stage

| Quantity | SurvStudio | R | difference | lifelines | difference | scikit-survival | difference |
|---|---|---|---|---|---|---|---|
| Stage I @ 12 survival | 0.955875 | 0.955875 | 4.44e-16 | 0.955875 | 4.44e-16 | 0.955875 | 3.33e-16 |
| Stage I @ 12 CI lower | 0.921684 | 0.921684 | 5.55e-16 | 0.921684 | 5.55e-16 | 0.921684 | 4.44e-16 |
| Stage I @ 12 CI upper | 0.975337 | 0.975337 | 1.11e-16 | 0.975337 | 1.11e-16 | 0.975337 | 0 |
| Stage I @ 36 survival | 0.763198 | 0.763198 | 6.66e-16 | 0.763198 | 1.67e-15 | 0.763198 | 4.44e-16 |
| Stage I @ 36 CI lower | 0.6863 | 0.6863 | 7.77e-16 | 0.6863 | 1.55e-15 | 0.6863 | 5.55e-16 |
| Stage I @ 36 CI upper | 0.82366 | 0.82366 | 4.44e-16 | 0.82366 | 1.44e-15 | 0.82366 | 1.11e-16 |
| Stage I @ 60 survival | 0.50639 | 0.50639 | 1.11e-16 | 0.50639 | 1.33e-15 | 0.50639 | 3.33e-16 |
| Stage I @ 60 CI lower | 0.399683 | 0.399683 | 1.67e-16 | 0.399683 | 1.22e-15 | 0.399683 | 0 |
| Stage I @ 60 CI upper | 0.603582 | 0.603582 | 1.11e-16 | 0.603582 | 1.78e-15 | 0.603582 | 2.22e-16 |
| Stage II @ 12 survival | 0.809245 | 0.809245 | 1.11e-16 | 0.809245 | 1.44e-15 | 0.809245 | 1.11e-16 |
| Stage II @ 12 CI lower | 0.722447 | 0.722447 | 1.11e-16 | 0.722447 | 1.33e-15 | 0.722447 | 3.33e-16 |
| Stage II @ 12 CI upper | 0.871281 | 0.871281 | 2.22e-16 | 0.871281 | 1.67e-15 | 0.871281 | 3.33e-16 |
| Stage II @ 36 survival | 0.519052 | 0.519052 | 2.22e-16 | 0.519052 | 1.55e-15 | 0.519052 | 2.22e-16 |
| Stage II @ 36 CI lower | 0.399752 | 0.399752 | 1.67e-16 | 0.399752 | 8.33e-16 | 0.399752 | 1.67e-16 |
| Stage II @ 36 CI upper | 0.625644 | 0.625644 | 0 | 0.625644 | 1.55e-15 | 0.625644 | 0 |
| Stage II @ 60 survival | 0.292331 | 0.292331 | 1.11e-16 | 0.292331 | 4.44e-16 | 0.292331 | 1.11e-16 |
| Stage II @ 60 CI lower | 0.156826 | 0.156826 | 3.61e-16 | 0.156826 | 1.11e-16 | 0.156826 | 3.61e-16 |
| Stage II @ 60 CI upper | 0.441997 | 0.441997 | 5.55e-17 | 0.441997 | 9.44e-16 | 0.441997 | 1.11e-16 |
| Stage III @ 12 survival | 0.732428 | 0.732428 | 0 | 0.732428 | 2.22e-16 | 0.732428 | 0 |
| Stage III @ 12 CI lower | 0.616215 | 0.616215 | 3.33e-16 | 0.616215 | 1.11e-16 | 0.616215 | 3.33e-16 |
| Stage III @ 12 CI upper | 0.818507 | 0.818507 | 3.33e-16 | 0.818507 | 4.44e-16 | 0.818507 | 2.22e-16 |
| Stage III @ 36 survival | 0.369002 | 0.369002 | 5.55e-17 | 0.369002 | 2.22e-16 | 0.369002 | 5.55e-17 |
| Stage III @ 36 CI lower | 0.237817 | 0.237817 | 5e-16 | 0.237817 | 5.55e-16 | 0.237817 | 4.44e-16 |
| Stage III @ 36 CI upper | 0.500564 | 0.500564 | 2.22e-16 | 0.500564 | 4.44e-16 | 0.500564 | 2.22e-16 |
| Stage III @ 60 survival | 0.207024 | 0.207024 | 3.89e-16 | 0.207024 | 4.16e-16 | 0.207024 | 3.61e-16 |
| Stage III @ 60 CI lower | 0.0854037 | 0.0854037 | 4.16e-17 | 0.0854037 | 4.16e-17 | 0.0854037 | 4.16e-17 |
| Stage III @ 60 CI upper | 0.364901 | 0.364901 | 1.67e-16 | 0.364901 | 1.11e-16 | 0.364901 | 1.67e-16 |
| Stage IV @ 12 survival | 0.836364 | 0.836364 | 3.33e-16 | 0.836364 | 2.22e-16 | 0.836364 | 4.44e-16 |
| Stage IV @ 12 CI lower | 0.620452 | 0.620452 | 0 | 0.620452 | 1.11e-16 | 0.620452 | 1.11e-16 |
| Stage IV @ 12 CI upper | 0.935291 | 0.935291 | 1.11e-16 | 0.935291 | 0 | 0.935291 | 1.11e-16 |
| Stage IV @ 36 survival | 0.32314 | 0.32314 | 3.89e-16 | 0.32314 | 2.22e-16 | 0.32314 | 3.33e-16 |
| Stage IV @ 36 CI lower | 0.115638 | 0.115638 | 2.78e-16 | 0.115638 | 2.22e-16 | 0.115638 | 2.78e-16 |
| Stage IV @ 36 CI upper | 0.553468 | 0.553468 | 3.33e-16 | 0.553468 | 1.11e-16 | 0.553468 | 2.22e-16 |
| Stage IV @ 60 survival | 0.107713 | 0.107713 | 4.44e-16 | 0.107713 | 4.44e-16 | 0.107713 | 4.3e-16 |
| Stage IV @ 60 CI lower | 0.00739135 | 0.00739135 | 6.07e-18 | 0.00739135 | 6.07e-18 | 0.00739135 | 8.67e-19 |
| Stage IV @ 60 CI upper | 0.363573 | 0.363573 | 1.11e-16 | 0.363573 | 1.11e-16 | 0.363573 | 1.11e-16 |
| Stage I median | 76.15 | 76.15 | 0 | 76.15 | 0 |  |  |
| Stage I median CI lower | 50.3 | 50.3 | 0 | 50.3 | 0 |  |  |
| Stage I median CI upper | 110.41 | 110.41 | 0 | 110.41 | 0 |  |  |
| Stage I RMST | 61.3004 | 61.3004 | 7.11e-15 | 61.3004 | 1.14e-13 |  |  |
| Stage I RMST SE | 2.59865 | 2.59865 | 3.55e-15 |  |  |  |  |
| Stage II median | 37.68 | 37.68 | 0 | 37.68 | 0 |  |  |
| Stage II median CI lower | 29.43 | 29.43 | 0 | 29.43 | 0 |  |  |
| Stage II median CI upper | 49.31 | 49.31 | 0 | 49.31 | 0 |  |  |
| Stage II RMST | 43.6289 | 43.6289 | 2.13e-14 | 43.6289 | 8.53e-14 |  |  |
| Stage II RMST SE | 3.83307 | 3.83307 | 1.78e-15 |  |  |  |  |
| Stage III median | 26.51 | 26.51 | 0 | 26.51 | 0 |  |  |
| Stage III median CI lower | 15.7 | 15.7 | 0 | 15.7 | 0 |  |  |
| Stage III median CI upper | 41.56 | 41.56 | 0 | 41.56 | 0 |  |  |
| Stage III RMST | 35.8646 | 35.8646 | 2.13e-14 | 35.8646 | 2.84e-14 |  |  |
| Stage III RMST SE | 4.24093 | 4.24093 | 3.55e-15 |  |  |  |  |
| Stage IV median | 27.14 | 27.14 | 0 | 27.14 | 0 |  |  |
| Stage IV median CI lower | 15.37 | 15.37 | 0 | 15.37 | 0 |  |  |
| Stage IV median CI upper | 42.48 | 42.48 | 0 | 42.48 | 0 |  |  |
| Stage IV RMST | 32.0596 | 32.0596 | 4.97e-14 | 32.0596 | 3.55e-14 |  |  |
| Stage IV RMST SE | 6.04895 | 6.04895 | 2.66e-15 |  |  |  |  |
| log-rank chi-square | 52.2549 | 52.2549 | 5.68e-14 | 52.2549 | 1.49e-13 | 52.2549 | 1.07e-13 |

## GBSG2: Cox model

| Quantity | SurvStudio | R | difference | lifelines | difference | scikit-survival | difference |
|---|---|---|---|---|---|---|---|
| age coefficient | -0.00945924 | -0.00945924 | 7.74e-16 | -0.00945923 | 7.28e-09 | -0.00945924 | 5.22e-16 |
| age SE | 0.00930059 | 0.00930059 | 3.12e-17 | 0.00930059 | 1.92e-11 |  |  |
| tsize coefficient | 0.00779608 | 0.00779608 | 6.07e-17 | 0.00779607 | 1.39e-08 | 0.00779608 | 1.22e-16 |
| tsize SE | 0.00393902 | 0.00393902 | 1.13e-17 | 0.00393902 | 1.13e-09 |  |  |
| pnodes coefficient | 0.0487886 | 0.0487886 | 3.47e-17 | 0.0487889 | 2.71e-07 | 0.0487886 | 1.39e-17 |
| pnodes SE | 0.00744709 | 0.00744709 | 8.67e-19 | 0.00744707 | 2.23e-08 |  |  |
| progrec coefficient | -0.00221724 | -0.00221724 | 9.15e-17 | -0.00221724 | 9.3e-11 | -0.00221724 | 8.54e-17 |
| progrec SE | 0.000573529 | 0.000573529 | 8.87e-17 | 0.000573529 | 2.5e-11 |  |  |
| estrec coefficient | 0.000197311 | 0.000197311 | 5.96e-19 | 0.000197311 | 5.63e-11 | 0.000197311 | 6.64e-18 |
| estrec SE | 0.000450368 | 0.000450368 | 7.83e-17 | 0.000450368 | 1.52e-10 |  |  |
| horTh: yes vs no coefficient | -0.346278 | -0.346278 | 1.11e-16 | -0.346279 | 3.19e-07 | -0.346278 | 1.22e-15 |
| horTh: yes vs no SE | 0.129075 | 0.129075 | 3.61e-16 | 0.129075 | 9.72e-09 |  |  |
| menostat: Pre vs Post coefficient | -0.258445 | -0.258445 | 1.35e-14 | -0.258445 | 2.93e-08 | -0.258445 | 1.14e-14 |
| menostat: Pre vs Post SE | 0.183476 | 0.183476 | 1.3e-15 | 0.183476 | 5.3e-09 |  |  |
| tgrade: II vs I coefficient | 0.636112 | 0.636112 | 2e-14 | 0.636112 | 1.05e-07 | 0.636112 | 2.54e-14 |
| tgrade: II vs I SE | 0.249202 | 0.249202 | 1.03e-15 | 0.249203 | 6.1e-08 |  |  |
| tgrade: III vs I coefficient | 0.779654 | 0.779654 | 2e-14 | 0.779654 | 2.72e-07 | 0.779654 | 2.64e-14 |
| tgrade: III vs I SE | 0.26848 | 0.26848 | 1.39e-15 | 0.26848 | 5.69e-08 |  |  |
| partial log-likelihood | -1735.73 | -1735.73 | 2.27e-13 | -1735.73 | 7.51e-10 |  |  |
| likelihood-ratio chi-square | 104.745 | 104.745 | 3.27e-13 | 104.745 | 1.5e-09 |  |  |
| concordance | 0.691851 | 0.691851 | 2.22e-16 | 0.691851 | 2.22e-16 | 0.691851 | 2.22e-16 |
| age PH chi-square | 2.43355 | 2.43355 | 6.22e-15 | 2.43355 | 5.41e-06 |  |  |
| tsize PH chi-square | 0.321678 | 0.321678 | 2.35e-14 | 0.32168 | 2.08e-06 |  |  |
| pnodes PH chi-square | 0.456075 | 0.456075 | 2.29e-14 | 0.456093 | 1.81e-05 |  |  |
| progrec PH chi-square | 0.696063 | 0.696063 | 1.37e-14 | 0.696063 | 4.64e-07 |  |  |
| estrec PH chi-square | 1.61847 | 1.61847 | 1.78e-15 | 1.61847 | 3.71e-06 |  |  |
| horTh: yes vs no PH chi-square | 0.119788 | 0.119788 | 4.86e-16 | 0.119788 | 1.76e-07 |  |  |
| menostat: Pre vs Post PH chi-square | 9.08672e-05 | 9.08672e-05 | 1.68e-16 | 9.0839e-05 | 2.82e-08 |  |  |
| tgrade: II vs I PH chi-square | 1.51883 | 1.51883 | 1.26e-13 | 1.51883 | 2e-06 |  |  |
| tgrade: III vs I PH chi-square | 5.27694 | 5.27694 | 2.02e-13 | 5.27694 | 2.07e-06 |  |  |
| global PH chi-square | 22.6101 | 22.6101 | 8.88e-14 |  |  |  |  |

## GBSG2: Cox model stratified by menopausal status

| Quantity | SurvStudio | R | difference | lifelines | difference | scikit-survival | difference |
|---|---|---|---|---|---|---|---|
| age coefficient | -0.0124611 | -0.0124611 | 9.71e-17 | -0.0124611 | 7.58e-09 |  |  |
| age SE | 0.00895525 | 0.00895525 | 2.43e-17 | 0.00895525 | 9.05e-11 |  |  |
| tsize coefficient | 0.00772346 | 0.00772346 | 2.08e-17 | 0.00772345 | 1.19e-08 |  |  |
| tsize SE | 0.00388299 | 0.00388299 | 4.99e-17 | 0.00388299 | 9.8e-10 |  |  |
| pnodes coefficient | 0.0526375 | 0.0526375 | 3.47e-17 | 0.0526378 | 2.34e-07 |  |  |
| pnodes SE | 0.0073934 | 0.0073934 | 7.81e-17 | 0.00739338 | 1.92e-08 |  |  |
| horTh: yes vs no coefficient | -0.376634 | -0.376634 | 3.33e-16 | -0.376634 | 2.16e-07 |  |  |
| horTh: yes vs no SE | 0.128605 | 0.128605 | 3.33e-16 | 0.128605 | 1.03e-08 |  |  |
| partial log-likelihood | -1554.94 | -1554.94 | 6.82e-13 | -1554.94 | 5.59e-10 |  |  |
| likelihood-ratio chi-square | 65.2392 | 65.2392 | 3.64e-12 | 65.2392 | 1.12e-09 |  |  |
| age PH chi-square | 3.77468 | 3.77468 | 1.86e-13 | 3.77468 | 7.01e-06 |  |  |
| tsize PH chi-square | 0.509716 | 0.509716 | 3.69e-14 | 0.509718 | 1.53e-06 |  |  |
| pnodes PH chi-square | 0.334349 | 0.334349 | 1.18e-14 | 0.334364 | 1.47e-05 |  |  |
| horTh: yes vs no PH chi-square | 0.000601034 | 0.000601034 | 3.83e-16 | 0.000601025 | 8.65e-09 |  |  |
| global PH chi-square | 4.39225 | 4.39225 | 1.63e-13 |  |  |  |  |

## TCGA-LUAD: Cox model (complete cases)

| Quantity | SurvStudio | R | difference | lifelines | difference | scikit-survival | difference |
|---|---|---|---|---|---|---|---|
| age coefficient | 0.0103262 | 0.0103262 | 1.09e-16 | 0.0103262 | 9.9e-13 | 0.0103262 | 1.02e-16 |
| age SE | 0.00834349 | 0.00834349 | 9.02e-17 | 0.00834349 | 1.02e-12 |  |  |
| sex: Male vs Female coefficient | 0.0613172 | 0.0613172 | 0 | 0.0613172 | 6.33e-12 | 0.0613172 | 2.78e-16 |
| sex: Male vs Female SE | 0.163331 | 0.163331 | 2.5e-16 | 0.163331 | 5.61e-12 |  |  |
| stage_group: Stage II vs Stage I coefficient | 0.879484 | 0.879484 | 0 | 0.879484 | 2.33e-11 | 0.879484 | 9.99e-16 |
| stage_group: Stage II vs Stage I SE | 0.196607 | 0.196607 | 2.5e-16 | 0.196607 | 3.55e-11 |  |  |
| stage_group: Stage III vs Stage I coefficient | 1.19309 | 1.19309 | 1.33e-15 | 1.19309 | 1.5e-11 | 1.19309 | 4.44e-16 |
| stage_group: Stage III vs Stage I SE | 0.204409 | 0.204409 | 8.33e-17 | 0.204409 | 3.71e-11 |  |  |
| stage_group: Stage IV vs Stage I coefficient | 1.36045 | 1.36045 | 3.11e-15 | 1.36045 | 3.93e-09 | 1.36045 | 1.55e-15 |
| stage_group: Stage IV vs Stage I SE | 0.290717 | 0.290717 | 5e-16 | 0.290717 | 3.69e-10 |  |  |
| smoking_status: Current smoker vs Lifelong Non-smoker coefficient | -0.216717 | -0.216717 | 1.94e-16 | -0.216717 | 2.98e-11 | -0.216717 | 1.89e-15 |
| smoking_status: Current smoker vs Lifelong Non-smoker SE | 0.265733 | 0.265733 | 3.89e-16 | 0.265733 | 1.41e-11 |  |  |
| smoking_status: Former smoker (duration unknown) vs Lifelong Non-smoker coefficient | 1.09409 | 1.09409 | 1.78e-15 | 1.09409 | 2.32e-11 | 1.09409 | 1.78e-15 |
| smoking_status: Former smoker (duration unknown) vs Lifelong Non-smoker SE | 1.03 | 1.03 | 2.66e-15 | 1.03 | 1.87e-10 |  |  |
| smoking_status: Former smoker <=15y vs Lifelong Non-smoker coefficient | 0.121013 | 0.121013 | 1.12e-15 | 0.121013 | 2.96e-12 | 0.121013 | 2.62e-15 |
| smoking_status: Former smoker <=15y vs Lifelong Non-smoker SE | 0.240907 | 0.240907 | 1.67e-16 | 0.240907 | 5.17e-12 |  |  |
| smoking_status: Former smoker >15y vs Lifelong Non-smoker coefficient | -0.041832 | -0.041832 | 9.16e-16 | -0.041832 | 3.05e-13 | -0.041832 | 1.58e-15 |
| smoking_status: Former smoker >15y vs Lifelong Non-smoker SE | 0.265798 | 0.265798 | 5e-16 | 0.265798 | 3.16e-13 |  |  |
| partial log-likelihood | -866.563 | -866.563 | 4.55e-13 | -866.563 | 1.14e-12 |  |  |
| likelihood-ratio chi-square | 50.0751 | 50.0751 | 9.02e-13 | 50.0751 | 2.27e-12 |  |  |
| concordance | 0.681003 | 0.681003 | 1.11e-16 | 0.681003 | 1.11e-16 | 0.681003 | 1.11e-16 |
| age PH chi-square | 0.307718 | 0.307718 | 9.88e-15 | 0.307718 | 9.02e-10 |  |  |
| sex: Male vs Female PH chi-square | 1.7945 | 1.7945 | 3.33e-15 | 1.7945 | 2.03e-09 |  |  |
| stage_group: Stage II vs Stage I PH chi-square | 2.40577 | 2.40577 | 7.99e-15 | 2.40577 | 1.56e-09 |  |  |
| stage_group: Stage III vs Stage I PH chi-square | 4.82366 | 4.82366 | 8.88e-15 | 4.82366 | 2.73e-09 |  |  |
| stage_group: Stage IV vs Stage I PH chi-square | 0.00041491 | 0.00041491 | 6.4e-17 | 0.00041491 | 1.62e-11 |  |  |
| smoking_status: Current smoker vs Lifelong Non-smoker PH chi-square | 0.178043 | 0.178043 | 9.16e-16 | 0.178043 | 9.32e-10 |  |  |
| smoking_status: Former smoker (duration unknown) vs Lifelong Non-smoker PH chi-square | 0.83896 | 0.83896 | 4.44e-16 | 0.83896 | 2.74e-10 |  |  |
| smoking_status: Former smoker <=15y vs Lifelong Non-smoker PH chi-square | 0.002892 | 0.002892 | 2.41e-16 | 0.002892 | 5e-11 |  |  |
| smoking_status: Former smoker >15y vs Lifelong Non-smoker PH chi-square | 0.0249695 | 0.0249695 | 6.31e-16 | 0.0249695 | 1.46e-10 |  |  |
| global PH chi-square | 9.91342 | 9.91342 | 3.02e-14 |  |  |  |  |

## GBSG2: marker score tests (marker engine)

| Quantity | SurvStudio | R | difference | lifelines | difference | scikit-survival | difference |
|---|---|---|---|---|---|---|---|
| pnodes marginal score chi-square | 78.4594 | 78.4594 | 1.14e-13 |  |  |  |  |
| pnodes added-value score chi-square | 68.9237 | 68.9237 | 1.8e-07 |  |  |  |  |
| progrec marginal score chi-square | 21.1936 | 21.1936 | 8.17e-14 |  |  |  |  |
| progrec added-value score chi-square | 15.6225 | 15.6225 | 1.3e-07 |  |  |  |  |
| estrec marginal score chi-square | 4.18411 | 4.18411 | 1.78e-15 |  |  |  |  |
| estrec added-value score chi-square | 2.43215 | 2.43215 | 2.34e-08 |  |  |  |  |
| tsize marginal score chi-square | 17.9053 | 17.9053 | 1.28e-13 |  |  |  |  |
| tsize added-value score chi-square | 15.2956 | 15.2956 | 3.39e-08 |  |  |  |  |

## GBSG2: Cox fit of the marker engine (stratified by menopausal status)

| Quantity | SurvStudio | R | difference | lifelines | difference | scikit-survival | difference |
|---|---|---|---|---|---|---|---|
| age coefficient (efron) | -0.0108121 | -0.0108121 | 1.58e-16 |  |  |  |  |
| age SE (efron) | 0.00912524 | 0.00912524 | 9.02e-17 |  |  |  |  |
| tsize coefficient (efron) | 0.00759116 | 0.00759116 | 2.26e-17 |  |  |  |  |
| tsize SE (efron) | 0.00386639 | 0.00386639 | 7.37e-17 |  |  |  |  |
| pnodes coefficient (efron) | 0.0515414 | 0.0515414 | 2.08e-17 |  |  |  |  |
| pnodes SE (efron) | 0.00758399 | 0.00758399 | 7.46e-17 |  |  |  |  |
| progrec coefficient (efron) | -0.00256219 | -0.00256219 | 6.59e-17 |  |  |  |  |
| progrec SE (efron) | 0.000564043 | 0.000564043 | 8.11e-17 |  |  |  |  |
| partial log-likelihood (efron) | -1544.68 | -1544.68 | 2.27e-12 |  |  |  |  |
| age coefficient (breslow) | -0.0107989 | -0.0107989 | 1.91e-17 |  |  |  |  |
| age SE (breslow) | 0.00912488 | 0.00912488 | 0 |  |  |  |  |
| tsize coefficient (breslow) | 0.0075977 | 0.0075977 | 1.13e-17 |  |  |  |  |
| tsize SE (breslow) | 0.00386634 | 0.00386634 | 7.94e-17 |  |  |  |  |
| pnodes coefficient (breslow) | 0.0515278 | 0.0515278 | 2.08e-17 |  |  |  |  |
| pnodes SE (breslow) | 0.00758493 | 0.00758493 | 6.77e-17 |  |  |  |  |
| progrec coefficient (breslow) | -0.00256213 | -0.00256213 | 4.73e-17 |  |  |  |  |
| progrec SE (breslow) | 0.000563952 | 0.000563952 | 9.52e-17 |  |  |  |  |
| partial log-likelihood (breslow) | -1544.77 | -1544.77 | 1.36e-12 |  |  |  |  |

