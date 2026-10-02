# SurvStudio 0.3.0 논문 재현과 후속 검토

**근거 갱신: 2026-10-02 15:42 KST.**

**현재 판정: 코드 수정과 확대 검증은 완료한 부분부터 근거를 확보했다. 논문 전체 재현은 실행 중이며 제출 준비 완료로 판정하지 않는다.**

사용자의 후속 진행 승인에 따라 main 0.2.0 감사에 이어, 미병합 PR #8의 0.3.0 논문 재현 자료를 발견하고 검토했다.
수정은 [draft PR #9](https://github.com/kangk1204/SurvStudio/pull/9)에 보존했다. 원본 main 감사와 release candidate는
별도의 참조 checkout에 보존했다. 새 계산의 기준은 `3e0c4af142e7afc005f3a9bfb28e23c99c203ff7`이며,
확대 검정 연구는 사전 고정한 `adfc187a` checkout과 독립 분석 환경에서 실행했다.
수치 계산을 실행하는 동안 해당 checkout은 변경하지 않는다. 해석·화면 수정은 별도 checkout에서 검증하며,
비교 구간 반복수와 표시·그림 수정에 이어 `24b4f1f7`에서 보조 tier 추론의 수렴·식별성 검사를 추가했다.
숫자 입력과 새 표시 코드의 provenance를 구분한다.

## 현재 완료·진행 상태

- 완료: Case I–IV 내부·외부 분석 중 위에 보고한 결과, 실제 9개 모델 비교, 16,000개 확대 조건 검증,
  100쌍 LASSO 비교, 독립 R 수치·scoring·pooling·bootstrap 대조, P2 고정 200개와 P3 고정 200개 실행.
- 실행 중: 전체 plasmode 2,300개 중 최소 1,800개, Case V 재발 분석과 후속 기관 sensitivity·tier 분석,
  고정 60개 seed 평가, Mime 500개 후보 sensitivity와 고정 null 설계 50개. 남은 계산을 줄이거나
  유리한 완료 결과만 골라 최종 값으로 보고하지 않는다.
- 교정된 bootstrap 2,000회 경쟁 절차 집계와 새 tier guard 후처리는 원 R fit과 기존 원 계산이 준비된 뒤
  별도 provenance로 실행한다. 현재의 전체 재현·그림 패키지를 제출 완료 상태로 부르지 않는다.
- 사람 대상 연구의 실제 참가자는 0명이다. 준비된 protocol과 합성 과제는 관찰된 사용성 결과가 아니다.

## 실행 위치와 재현 근거

- Tailscale 서버: `keunsoo-3900x`, `100.122.125.104`; 영구 checkout `/home/keunsoo/projects/SurvStudio-paper`.
- 추가 seed 연구: 현재 여유를 다시 확인한 32 CPU 서버 `100.75.73.88`의 영구 checkout
  `/home/keunsoo/projects/SurvStudio-seed-audit`; 동일한 전체 commit과 72개 명시된 패키지 pin, 전체 데이터 해시를
  확인했다. 기본 60개 평가의 표본·설정·seed를 유지하며 분산한다. 실제 OMP/BLAS/MKL thread는 각 1개이다.
- Mime null 비교의 아직 시작하지 않은 고정 설계 10–49는 여유를 확인한 `100.106.141.29`의
  `/home/keunsoo/projects/SurvStudio-paper`로 분산한다. 같은 전체 commit, 동일한 절대 경로의 R 환경과
  입력 내용을 checksum으로 확인했다. P2 설계 0을 먼저 재현하여 4개 CSV가 바이트 단위로 같고 elapsed time 외
  metadata도 같은 것을 확인했다. 기존 서버의 설계 0–9는 그대로 실행하며, 대기열만 멈춰 중복 실행을 방지한다.
  보조 서버는 16개 one-core worker, 같은 8시간 제한과 12 GB 주소공간 제한을 사용한다. 실패를 다음 설계로 대체하지 않는다.
- 변경 전 0.3.0 참조: `/home/keunsoo/projects/SurvStudio-release-reference`, Git `f7d8baa1b464a405e38968b2a544cd646b56709c`.
- 증거: `/home/keunsoo/projects/SurvStudio/audit_results/20261002_followup`.
- 로컬 증거 사본: `/Users/keunsoo/Documents/projects/00_SurvStudio_remote/audit_records/20261002_followup`.
- 원자료 재준비 후 LUAD 9개 분석 입력과 breast 79개 파일이 기존 명세의 해시와 모두 일치했다.
  GEO 플랫폼 GPL570 다운로드 압축파일 하나는 달랐지만, 그 자료로 재생성한 분석 입력은 일치했다.
  임의로 기대 해시를 바꾸어 검사를 통과시키지 않았다.
- Python 분석 환경은 `paper/requirements-lock.txt`의 72개 명시된 패키지 버전을 확인했다. 검사용 lifelines 설치가 pandas 버전을
  바꾼 것을 발견하여 즉시 되돌리고 테스트 환경을 분리했다. 재확인 결과 lock 불일치가 없었다.
  확대 실험은 복구된 환경의 불변 checkout에서 다시 실행해 처음의 48,000개 기록과 바이트 단위로 일치했다.
- 데이터 준비용 R은 4.5.2, MetaGxBreast 1.30.0, GEOquery 2.78.0, data.table 1.18.4이다.
  비교 도구는 별도 고정 R 4.4.3 환경과 지정한 Mime/CoxBoost commit을 사용한다. 실제 패키지 버전 기록도 보존했다.

## 추가로 고친 중요한 문제

1. **gain 구간의 비교 대상**: 조건부 null에서는 마커가 추가 정보를 갖지 않지만, 잡음 마커로 학습한 실제 모델의
   새 환자에서의 gain이 항상 정확히 0은 아니다. 기존 summary의 무조건적인 zero coverage를 고쳐,
   독립 생성한 3,000명에서의 고정 모델 gain과 대조하고 null-zero coverage를 따로 기록했다.
   다만 부분표본에서 다시 선택·학습한 절차와 전체 표본에서 고정한 모델은 학습 크기와 목표가 다르다.
   이 비교는 성능 추정의 진단이며, 선택 절차에 대한 정식 95% coverage 보장으로 해석하지 않는다.
2. **비교 도구의 실패 반복**: Mime 실패를 다음 반복으로 대체하는 방식을 제거했다. 정해진 처음 50개 설계를
   유지하고, 실패·미완료를 모두 포함한 결과 범위를 추가했다. 성공한 실행에 조건부인 비율을 독립적인 FWER처럼
   보고할 수 없다. P1–P3와 SurvStudio는 서로 다른 주장을 검정하므로 claim rate도 같은 FWER가 아니다.
3. **부분 실행의 그림 생성**: 다른 사례의 입력이 아직 없는 합성 그림은 pending으로 표시한다.
   이미 존재하지만 변경되거나 서로 다른 분석에서 온 결과는 계속 오류로 거부한다.
4. **연구 설계 지도**: 원 생성기와 반복별 결과를 모든 Git 이력에서 찾지 못했다. 정적 pilot 값의 생성 근거를
   재현했다고 주장하지 않고, API·화면에 역사적 미재현 pilot임을 명시했다. 개인 연구의 정확한 편향 예측 도구로
   주장할 근거가 없다.
5. 0.2.0 감사에서 고친 내부 fold별 LASSO 전처리, 정확한 risk tie 처리, Brier·calibration 입력 검사,
   subsample gap-adjusted 명칭, DeepHit 변형 구현 표시, R 수치 검증과 배포 파일 포함을 0.3.0에 통합했다.
6. **내부 성능의 대상과 화면 해석**: 부분표본마다 마커 선택과 모델 학습을 반복하므로 left-out C와 gain은
   전체 표본의 최종 고정 모델이 아니라 선택 절차에 대한 추정이다. 화면·보고서·그림의 설명을 바로잡았다.
   구간 없이 평균 gain만 남은 예전 결과에는 확실한 판정을 주지 않는다. `0.02`는 화면의 표시 기준이며
   임상 효용의 확립된 최소값이 아니다. FWER 관련 설명에는 잔차 교환가능성 가정과 독립 검증 필요성을 유지한다.
7. **실패와 그림 provenance**: 그림 입력의 commit이 unknown 또는 dirty이거나 분석 코드 해시가 없으면 거부한다.
   경쟁 도구 비교표와 그림에는 완료/예정 반복 수와 실패·미완료를 포함한 범위를 전달한다. 이 범위는
   실패 결과를 모두 0 또는 1로 놓은 범위이며 신뢰구간이 아니다. 서로 다른 도구의 서로 다른 claim을 공통 FWER로 해석하지 않는다.
8. **중복 감사의 입력 변경**: script 11이 검사 중 커밋된 `breast_duplicate_pairs.csv`를 덮어썼다.
   기본 출력은 `results/breast_duplicate_pairs_recomputed.csv`로 바꾸어 분석 입력을 보존한다. 기존 코드는
   별도 checkout에서 실행했기 때문에 현재 진행 중인 본 분석의 입력에는 영향이 없다.
9. **비교 도구의 잘못된 귀속**: P3의 미보정 최적 절단값 검색을 KM Plotter의 현재 기본 기능으로 설명했다.
   [공식 설명](https://kmplot.com/analysis/)에는 BH FDR이 명시되어 있다. P3를 별도의 미보정 절차로 수정했고,
   gene-count-only Bonferroni도 절단값 탐색의 다중검정을 해결하지 않음을 표시했다. 현재 서비스의 오류율이나
   기능 부재를 이 P3 실험으로 주장할 수 없다. 실행 중인 숫자 계산은 보존한다.
10. **그림의 인쇄 크기와 잘림**: 기존 Figure 1은 251 mm 높이여서 figure와 legend를 합한 225 mm 제한을 넘었다.
    새 guard는 legend 25 mm를 남겨 그림 높이를 200 mm 이하로 제한한다. Raster 화면의 글자도 캡처한 SVG의
    글꼴과 최종 인쇄 폭으로 검증한다. 7 pt는 프로젝트의 가독성 하한이며 저널이 명시한 최소값으로 인용하지 않는다.
    실제 900 px 화면에서 숫자가 잘리거나 긴 축 라벨이 겹치는 문제도 수정했다. 최종 native summary는
    2,358×1,377 px이며 원본 구성요소를 브라우저에서 캡처했다. 인쇄 폭에서 최소 글자는 약 7.29 pt이다.
11. **비교 구간의 반복수 불일치**: Case II는 bootstrap 2,000회였지만 비교 절차는 200회를 사용했다.
    동일한 2,000회와 seed 20260926을 명시적으로 전달하도록 고쳤고 실제 설정도 결과에 기록한다.
    9개 회귀 검사가 통과했다. 고정한 R 모델을 보존한 채 새 구간과 pooling을 별도로 재계산해야 하며,
    실행 중인 기존 숫자를 새 버전의 결과로 바꾸어 기록하지 않는다.
12. **보조 tier 추론의 미수렴 결론**: robust 8개가 모두 재현되는 고정 합성 조건에서 기존 script 15b는
    미수렴 상태인데도 OR 약 2.1e11, upper Infinity와 LR p 약 7.2e-5를 출력하고 “no more often”이라고 서술했다.
    수렴·분리·rank·유한 구간을 확인하고 식별되지 않는 fit은 진단과 null 항목을 남기며 추론을 보류하도록 고쳤다.
    13개 서버 회귀 검사가 통과했다. 수렴하더라도 유전자 의존성을 무시한 working-model 결과는 기술적·탐색적이다.
    반대 방향의 외부 연관은 이질성이나 실제 반전일 수 있어 “empirical null”로 부르지 않는다. 진행 중인 원 계산은
    변경하지 않으며, 완료 후 새 보조 집계를 별도 provenance로 생성한다.

13. **실제 사례와 원인 해석**: ER 양성 재발 사례는 특정 잠금 signature의 이득에 대한 known-truth 양성 대조가 아니다.
    표현을 실제 자료 적용 사례로 바꾸고, 기관 holdout의 내부·외부 gain 비교도 코호트 이동과 optimism의 원인을
    분리해 확정하는 분석으로 설명하지 않는다. 학습 크기, signature와 대상 집단이 다를 수 있다. Wilson 구간도
    gene dependence 때문에 반드시 너무 좁다고 단정하지 않고 nominal coverage가 무효화될 수 있음을 명시한다.

## 완료된 검증

| 검증 | 확인한 결과 | 해석 범위 |
| --- | --- | --- |
| GitHub CI | [실행 36955573243](https://github.com/kangk1204/SurvStudio/actions/runs/36955573243)의 9개 작업 모두 성공 | Linux Python 3.11–3.13, macOS, Windows, 브라우저, 배포 설치, 화면 스크립트, R 비교 |
| CI 상세 | Ubuntu 3.12·macOS·Windows 각 1,563 passed, 41 skipped | 브라우저는 별도 작업에서 실행; skipped를 통과로 합산하지 않음 |
| 독립 서버 전체 검사 | 1,565 passed, 1 skipped, 19 warnings | 수치 계산 기준 `3e0c4af1`, 별도 테스트 환경 |
| 독립 R 수치 비교 | 7개 분석의 189항목 PASS; 최대 절대 차이 약 1.8e-7 | 지정한 KM/Cox/score 항목; 모든 통계 가정의 검증은 아님 |
| 실제 브라우저 | 38 passed | 브라우저 및 초보 사용자 흐름 테스트; 실제 인간 사용성 연구는 아님 |
| 새 회귀 검사 | 4 passed | 수정 명칭, 미생성 입력, stale 결과 거부, 실패 분모 유지 |
| 확대 검정 연구 | 16,000개 데이터 × 3방법, 48,000기록; 오류 0 | 8가지 사전 고정 조건, 각 2,000회 |
| 확대 연구 재현 | 두 실행의 CSV SHA-256 `b64e351c1af79f278f115ca27ecea47c90ae192e877bccd306a2f8b0fb04f92c` 일치 | 독립 고정 환경 재실행의 기록이 동일 |
| LASSO 수정 영향 | 동일 outer split 100쌍, 실패 0, 예측 변경 17쌍 | 전처리 위치를 바로잡은 공학적 실험; 정확도 우월성 입증은 아님 |
| 합성 사용자 과제 | A/B의 KM, Cox 계수, 고정 외부 C를 독립 R과 대조해 일치 | 실제 참가자 0명, 관찰 결과 없음 |
| Case II 독립 R scoring | 14개 코호트·scaling 조합의 70개 점추정 일치, 최대 차이 약 7.2e-16 | 준비된 코호트에 대한 임상 encoding, complete-case 선택, marker scale·대치, 고정 예측식, Harrell C와 calibration slope |
| Case II 독립 R pooling | 146개 수치 일치, 최대 차이 약 1.8e-15 | 코호트별 추정치를 입력으로 한 DL 분산·가중치, HKSJ 구간·하한 보정, prediction interval와 HR 변환 |
| Case IV 독립 R 검증 | 15개 점추정과 73개 집계 수치 일치; 최대 차이 각각 6.7e-16, 8.9e-16 | 준비된 3개 코호트의 고정 예측·C·calibration과 DL/HKSJ 집계; 상류 조화·중복 제외·bootstrap을 독립 재현한 것은 아님 |
| Mime 실제 자료 출력 | 117개 모델×8개 코호트의 C 936개를 독립 R에서 확인, 최대 차이 약 5.6e-16 | 원 위험 점수로 코호트별 단변량 Cox를 다시 계산한 reported C; 원 학습 117개를 다시 실행하거나 학습 경고를 해결한 것은 아님 |
| 외부 bootstrap 독립 R | 폐암 84개·유방암 18개 구간 끝값 일치; 최대 차이 2.2e-16, 1.1e-16 | 원 NumPy seed·추출 index를 공유하며 R의 독립 C 계산과 type-7 percentile을 대조; coverage 보장이 아님 |
| 완성된 그림 출력 | 5개 그림의 PNG·PDF·SVG 15개를 두 번 생성하여 각각 바이트 일치 | 인쇄 크기·해상도·포함된 TrueType 글꼴 확인, 수치 재계산과 구분 |

추가 해석 수정 `6d1bb277`의 [CI 실행 36959666364](https://github.com/kangk1204/SurvStudio/actions/runs/36959666364)도
9개 작업 모두 성공했다. 확인한 Ubuntu 3.12와 Windows 로그는 각각 1,567 passed, 41 skipped, 11 warnings이다.
앞서 통과한 버전의 결과를 새 commit의 검증처럼 재사용하지 않는다. 논문 본문 전체 수치 재현은 별도 로그로 진행 중이다.

입력 보존 수정 `dc8422fe`의 [CI 실행 36962595749](https://github.com/kangk1204/SurvStudio/actions/runs/36962595749)도
9개 작업 모두 성공했다. 인쇄 크기와 plot 수정 `8a21020a`의 서버 회귀 검사는 54 passed, 3 warnings이었다.

`5f3ce580`의 [CI 실행 36966175096](https://github.com/kangk1204/SurvStudio/actions/runs/36966175096)은
5개 전체 검사 작업에서 같은 라벨 기대값 하나가 실패했다. 확인한 Ubuntu 3.12와 Windows는 각 1 failed,
1,568 passed, 42 skipped, 11 warnings이다. 원본 로그를 보존하고, 줄바꿈을 반영한 `740532a0`의 보고서·그래프
서버 검사 69개가 통과한 것을 확인했다. 이어 `740532a0`의
[CI 실행 36967854867](https://github.com/kangk1204/SurvStudio/actions/runs/36967854867)은 9개 작업 모두 성공했고,
Ubuntu 3.12, macOS와 Windows의 전체 로그는 각각 1,569 passed, 42 skipped, 11 warnings이었다.
새 SVG export의 10개 서버 회귀 검사도 통과했고, `ee879be4`의
[CI 실행 36969497982](https://github.com/kangk1204/SurvStudio/actions/runs/36969497982)은 9개 작업 모두 성공했다.
추가 tier 추론 guard의 `24b4f1f7`은 13개 서버 회귀 검사를 통과했다.
[새 CI 실행 36971382301](https://github.com/kangk1204/SurvStudio/actions/runs/36971382301)의 9개 작업은 모두 성공했다.
확인한 Ubuntu 3.12·macOS·Windows 전체 로그는 각각 1,572 passed, 43 skipped, 11 warnings이다.
같은 합성 입력 SHA-256 `6ff840da9cd10eb83b7e94590a78dfc8cc178bf2980e34d7531bdaead2e477a7`로 원본의
무한 구간과 수정본의 추론 보류를 재현했다. 실행한 helper, 입력 CSV, 두 진단과 해시 manifest를 보존했다.

완성된 Figure 1, Figure 1a, Figure 2, Figure 4와 Figure S2의 재현 패키지는 로컬 증거의
`completed_case_figure_package/`에 보존했다. 원 수치표와 stamp, 화면 provenance, exact source CSV,
PNG·PDF·SVG, renderer, 독립 R 검증 입력·스크립트와 legend를 포함한다. SVG의 불필요한 외부 DTD도 제거하여
publication profile의 구조 검증을 통과했다. 구조 검증을 통계적 타당성 검증으로 합산하지 않는다.
별도 점검은 대표 수치의 R 일치와 실제 그림의 숫자·라벨·잘림·겹침을 확인했다. 재현 package에서 그림을 그리는
작업은 저장 결과의 rendering이며 상류 분석 전체 재현은 아니다.

## 확대 검정 연구의 결과

각 데이터는 180명, 마커 30개, 999순열이다. 선형 잔차 순열, 원 마커 순열, 임상변수 Z²를 추가한 잔차 순열을
같은 데이터에서 비교했다. 확인용 seed는 `2026100207`, 공학적 사전 점검은 별도 seed를 사용했다.
부분 null은 마커 5개에만 직접 효과를 주고 나머지 25개가 Z를 조건으로 정확히 null이 되도록 생성했다.

| 조건 | 선형 잔차 순열의 오류 데이터 | FWER | 95% Monte Carlo Wilson 구간 |
| --- | --- | --- | --- |
| 독립 마커 | 95/2000 | 4.75% | 3.90–5.77% |
| Z와 선형 관계 | 103/2000 | 5.15% | 4.26–6.21% |
| Z에 따른 이분산 | 103/2000 | 5.15% | 4.26–6.21% |
| Z와 비선형 관계 | 120/2000 | 6.00% | 5.04–7.13% |
| 상관 마커 | 92/2000 | 4.60% | 3.77–5.61% |
| Z 의존 검열 | 89/2000 | 4.45% | 3.63–5.44% |
| 부분 null, 약한 신호 | 85/2000 | 4.25% | 3.45–5.23% |
| 부분 null, 강한 신호 | 83/2000 | 4.15% | 3.36–5.12% |

비선형 조건의 6.0%는 보편적인 5% 통제 주장을 지지하지 않는다. Z² 추가 방식은 이분산 조건에서 7.1%였으므로
관측된 결과를 보고 자동으로 기본 방법을 바꿀 근거도 없다. 구간은 조건별 pointwise Monte Carlo 구간이며
조건·방법 전체에 걸친 동시 구간이 아니다. Z 의존 검열은 Z를 조건으로 사건시간과 독립이다. 임의의 informative
censoring을 다뤘다는 뜻이 아니다. robust tier의 오류율은 이 실험의 목표가 아니며 genome-wide 조건 전체를
검증한 것도 아니다. 약한/강한 신호의 평균 검정력은 각각 8.97%/48.28%였다.

`expanded_calibration_figure/`에는 PNG·PDF·SVG, 독립 재계산된 source CSV, 입력 기록, 재생성 스크립트와 manifest가
있다. 구조 검증과 Wilson 구간의 독립 검증을 별도로 수행했으며 그림을 다시 생성한 세 형식의 bytes도 일치했다.

## LASSO 수정과 실제 모델 비교의 해석

합성 paired outer split의 complete·rare category 조건에서는 평균 C 변화가 0이었다. missing 조건은 25쌍 중
12쌍의 예측이 바뀌었고 평균 ΔC는 -0.00440이었다. missing+rare 조건은 5쌍이 바뀌었고 평균 ΔC는 -0.000167이었다.
수정은 내부 검증 행의 전처리 정보가 벌점 선택에 들어가는 것을 막는다. 성능이 상승했다는 결론은 아니다.

실제 TCGA 임상 데이터에서 수정 전후 9개 모델의 C는 동일했다. 같은 test 147명, 54사건에서 Cox C는 0.598,
RSF 0.634였다. RSF와 Cox의 paired ΔC는 약 +0.036, CI -0.017–0.085로, 1위인 모델이 유의하게 우월하다고
결론내릴 수 없다. DeepHit 변형은 0.587이었다. 단일 holdout과 기본 설정의 비교는 알고리즘의 일반적인 우열이나
임상적 효용을 입증하지 않는다. `paired_real_model_comparison/`에 원본과 수정본의 상세 표·summary·차이표와
해시 manifest를 보존했다. split fingerprint는 두 실행 모두 `b036e18eb7a962f2`이며, train 342명/124사건,
test 147명/54사건, bootstrap 1,000회이다. C뿐 아니라 구간과 Cox 대비 차이 등 모든 비교한 숫자가 정확히 같았다.

## 비교 도구 실험의 평가 대상

Mime의 reported C는 각 코호트에서 위험 점수를 다시 단변량 Cox에 넣어 정한 방향을 따른다. 실제 완료된
117×8 출력은 독립 R에서 재현됐지만, 이 수치 일치가 개발 자료에서 정한 위험 방향의 외부 성능과 같다는 뜻은
아니다. 본 비교는 TCGA에서 한 번 정한 방향을 유지한 C를 별도로 계산한다.

7개 외부 코호트 모두로 선택한 winner의 같은 7개 코호트 성능은 독립 검증으로 보고할 수 없다. 3개로 선택하고
4개를 봉인하는 35개 조합은 selection replay이며, 코호트를 반복 공유한다. 35개 결과의 범위를 35개의 독립
검증 연구나 독립된 유효 표본처럼 취급하지 않는다. 동일 유전자 집합 9,938개와 동일 환자를 쓰는 비교와
각 도구의 원 기본 작업 흐름을 구분한다. 후보 100개 cap과 500개 sensitivity, 실행 가능한 알고리즘의 변경,
원 학습 경고와 제외 모델을 모두 설명해야 공정한 비교가 된다.

외부 점수 rescaling은 outcome을 쓰지 않지만 코호트의 분포를 쓴다. 이를 새 환자 한 명에게 개발 자료의
scaling만 적용하는 배포 조건과 같은 것으로 주장하지 않는다. P2/P3의 marginal claim은 임상변수 조건부
추가 정보를 검정하는 SurvStudio claim과 다르며, conditional null에서 나타난 marginal association을 곧바로
통계적 false positive로 이름 붙이지 않는다. 우월성의 핵심 근거는 서로 다른 claim rate의 크기 비교가 아니라
독립 검증을 유지하도록 돕는 작업 흐름과 그 재현 가능한 진단이어야 한다.

P3의 고정 200개 설계는 모두 완료됐다. 9,938개 유전자에서 미보정 최적 절단값 p < 0.05 claim은
모든 설계에 있었고 평균 3,124.965개였다. gene-count-only Bonferroni claim이 하나 이상인 설계는
147/200개, median-cut의 같은 기준은 29/200개였다. 이는 P3의 procedure claim count이다. 임상변수 조건부
null이 marginal null을 뜻하지 않으므로 이를 현 KM Plotter의 false-positive rate나 두 방법의 동일 FWER
비교로 보고하지 않는다. 원 200개 입력·출력, upstream와 source 해시를 보존했다.

P2의 고정 200개는 모두 원 출력이 있었고 147개는 유전자를 선택했으며 53개는 선택하지 않았다. 선택한 실행의
계수·SE·p와 위험 점수는 유한한 값 범위 검사를 통과했다. 선택하지 않은 실행의 NA 위험 점수는 예상된 결과다.
이 검사는 모든 Cox fit의 수렴·식별성이나 원 학습 경고가 해결되었다는 증거가 아니다. metadata의 iter < 20
수렴 표시는 경고 전체를 대신하는 검증으로 사용하지 않는다.

## 폐암의 새로운 내부 결과와 고정 외부 검증

Case I의 완전한 JSON을 새로 생성했다. 484명/177사건, 마커 20,530개 중 19,112개를 검사했고,
near-constant 1,079개와 constant 339개를 제외했다. 임상변수 조정 후 nominal p < 0.05인 마커 3,404개,
BH 306개, family-wise p 기준 5개, robust 5개, suggestive 450개였다. robust 마커는 DKK1, NTSR1, TLE1,
CTCFL, FAM117A이다. 잠금 모델은 10개 마커를 포함한다.

겉보기 C는 0.7493, heuristic subsample gap-adjusted C는 0.6502였다. 선택·재학습 절차의 left-out C는
0.6576, 임상변수만 사용한 절차는 0.6555, paired gain은 +0.00208이고 근사 95% 구간은 -0.04188–0.04603이었다.
이 결과가 최종 고정 모델의 성능 구간은 아니다.

Case II의 외부 7개 코호트는 총 1,509명/573사건이다. within-cohort rescaling에서 모델의 pooled C는
0.6663(HKSJ 95% 구간 0.6297–0.7029), 임상 모델 C는 0.6605(0.6242–0.6967)였다. **paired ΔC의 pooled 값은
+0.00012(-0.04132–0.04156)**, prediction interval은 -0.07999–0.08023이었다. 각 C와 paired 차이의 가중치가
다르므로 두 pooled C를 단순히 빼서 pooled gain이라고 보고하지 않는다. 모든 코호트별 gain 구간에도 0이 포함됐다.

Calibration slope는 0.568(HKSJ 0.298–0.837), 임상 모델은 1.095(0.846–1.345)였다. 유전자 부분의 조정 HR은
1.197(HKSJ 0.996–1.437), prediction interval 0.805–1.778이었다. 추가적인 임상 예측 이득이나 동등성을
확정할 증거가 없다. As-measured 분석의 paired ΔC도 +0.00694(-0.02880–0.04268)이었다.
within-cohort rescaling은 외부 환자 집단의 분포를 사용하는 비지도 적응이며, 새 환자 한 명의 독립적 배포와 다르다.

독립 R 재계산은 실제 locked recipe와 준비된 7개 코호트의 10개 마커 값을 사용했다. 원 준비 입력에는
임상변수 누락으로 제외되는 행이 남아 있으므로, R에서 joint complete cases와 positive time을 독립적으로
선택했다. 최초 보조 검증기의 전체 원행 수 가정은 실패했으며 그대로 보존했다. 수정 후 70개 점추정과
별도의 pooling 146개 수치가 일치했다. 상류 GEO harmonisation과 코호트별 bootstrap 구간을 R에서 독립적으로
재생성한 것은 아니다. `independent_case_II_R_evidence_manifest.json`에 입력·스크립트·실패·성공 기록의 해시를 보존했다.

실제 화면을 새 코드로 다시 캡처했고 `paper/interface/`에 이미지, 분석/표시 코드 provenance와 캡처 절차를 보존했다.
검증한 저장 결과를 화면에 표시한 것이며 브라우저에서 새 수치 분석을 실행한 것으로 표현하지 않는다.

## 유방암의 완료된 내부·외부 결과

Case IV 내부 분석은 METABRIC 1,846명/714사건, 25,235개 마커를 사용했다. 임상변수 조정 nominal p < 0.05는
3,692개, BH q ≤ 0.05는 444개, family-wise p 기준과 robust tier는 각각 20개였다. 잠금 모델은 10개 마커이다.
선택·재학습 절차의 left-out C는 0.6742, 임상 절차는 0.6673, paired gain은 +0.00692이고 근사 구간은
−0.01075–0.02460이었다. 겉보기 C 0.7118과 gap-adjusted C 0.6681을 고정 모델의 독립 성능으로 해석하지 않는다.

외부 사전 screen은 CAL, NKI, TRANSBIG 3개 코호트, 598명/173사건을 포함했다. pooled C는 0.6203
(HKSJ 0.4538–0.7868), paired gain은 +0.00444(HKSJ −0.07292–0.08179)였다. 추가 예측 이득을 확정하지 못한다.
3개 코호트에서 정규 근사 random-effects prediction interval은 C의 범위 밖까지 넓어졌다. 이는 매우 적은 코호트와
분산 추정의 불확실성을 보여주는 근사 구간이며 물리적으로 가능한 C 범위나 보장된 예측 범위로 제시하지 않는다.
가독성을 위해 구간을 임의로 잘라 확실한 것처럼 보이게 하지 않는다.

독립 R은 잠금 encoding·마커 scaling·대치와 고정 예측식에서 15개 C·gain·calibration 점추정을 확인했고,
73개 DL/HKSJ/PI·HR 집계 값도 확인했다. 입력 선택은 기존 MetaGxBreast loader와 duplicate 규칙에 조건부이며,
독립적인 상류 환자 조화·제외나 bootstrap coverage 검증으로 합산하지 않는다.

## 중복 환자 감사의 재현과 한계

유방암 감사의 109쌍은 환자 식별자, 순서와 모든 중복 판정이 기존 입력과 같았다. 상관계수 r와 gap의
최대 차이는 각각 3.33e-16이었다. CSV bytes와 해시는 다르므로 바이트 재현으로 주장하지 않는다.
분석 입력은 교체하지 않았으며 `duplicate_audit/canonical_comparison.json`에 비교 근거를 보존했다.
입력 보존 수정 후 실제 감사를 다시 실행해 입력 해시가 전후 같고 독립 생성 출력도 앞선 출력과 바이트가 같음을
확인했다. 최초 외부 검증기는 동시에 생성한 그림을 source 변경에 포함해 실패했으므로 그 로그는 유지했다.
후속 검증은 당시 status에서 생성 그림만 제외하여 source/input 보존과 실제 script exit 0을 확인했다.

폐암 감사는 QC가 이미 제외한 기술 중복 2쌍을 expression screen에서 모두 찾았다. 그 2쌍은 임상 주석이
서로 모순되어 임상 일치 검증의 양성 대조가 아니다. 현재 포함한 자료에서 expression 기준 5쌍이 flag됐으나,
임상 기준까지 만족한 쌍은 0이었다. 이는 모든 환자의 독립성이나 다른 플랫폼에서 중복 검출의 완전한 민감도를
입증하지 않는다. 유방암의 개발-검증 gap 0.15 같은 규칙에는 관찰한 자료를 참고한 선택이 포함되므로
독립적인 사전 고정 알고리즘 검증으로 해석하지 않는다.

## 최신 선행 연구를 반영한 논문 전략

가장 방어 가능한 기여는 **기존 방법을 연결해 선택 편향·독립 검증·보고를 점검하는 로컬 생존분석 작업 흐름**이다.
새 순열 이론, 새 DL 알고리즘, 최고의 예측 성능, 임상적 유용성으로 표현하면 현재 증거보다 큰 주장이 된다.

| 가까운 선행 연구 | 확인된 겹침 | 남길 수 있는 차별화 주장 |
| --- | --- | --- |
| [ESurv 2020](https://www.jmir.org/2020/5/e16084/) | 사용자 데이터, 생존분석, lasso·elastic net·network Cox와 교차검증 | 웹 생존분석·변수 선택 자체는 신규 기여가 아님 |
| [KM Plotter custom data, 2021](https://www.jmir.org/2021/7/e27633/) | 사용자 자료의 다변량 Cox, 절단값과 FDR | 임상변수 조정·사용자 자료 업로드·다중검정 자체는 신규 기여가 아님 |
| [surviveR 2023](https://www.nature.com/articles/s41598-023-48894-9) | 사용자 데이터의 유연한 생존분석 | 쉬운 사용성은 참가자 근거가 필요 |
| [PATH-SURVEYOR 2023](https://pubmed.ncbi.nlm.nih.gov/37380943/) | 경로 수준 분석과 Cox 공변량을 다루는 Shiny 도구 | 임상변수와 마커를 같이 분석한다는 설명만으로 차별화하기 어려움 |
| [DoSurvive 2023](https://pubmed.ncbi.nlm.nih.gov/37609633/) | 단일·결합 biomarker, Cox·AFT 및 여러 endpoint | 결합 마커 분석 자체는 신규 기여가 아님 |
| [SurvBoard 2025](https://pubmed.ncbi.nlm.nih.gov/41031875/) | 표준화된 다중오믹스 생존모델 비교와 검증의 함정 | 모델 개수나 작은 leaderboard만으로 우월성을 주장하지 않음 |
| [Survival Genie 2, 2026](https://link.springer.com/article/10.1186/s13073-026-01651-9) | 단일세포에서 만든 표적·gene set과 생존분석, 132데이터셋 | genome-wide 조회·gene set·풍부한 공개 데이터 자체를 최초로 주장하지 않음 |

PubMed/PMC와 출판사 원문·저장소를 함께 검색했다. 일부 출판사/PMC 열람은 제한되었으며 각 도구의 현재 서비스
기능을 모두 직접 비교한 체계적 연구는 아니다. 위 기능의 부재를 단정하지 않는다. 새 조합과 재현 근거를 갖춘
소프트웨어 논문이라는 **중간 수준의 신규성**은 검토자의 판단이며 게재 확률이나 저널 심사 결과를 보장하지 않는다.

현재 가장 효과적인 구성은 기본 수치 일치, 조건별 오류율과 실패 조건, 선택 절차의 내부 추정과 고정 외부 검증의
차이, 코호트별 paired gain·calibration, 동일 test 환자의 모델 비교를 중심에 두는 것이다. 유리한 결과를 얻기 위해
표본·endpoint·threshold를 바꾸거나 가장 좋은 코호트만 남기는 방식은 채택하지 않는다.

## 제출 방향과 필요한 증거

2026-10-02에 확인한 각 저널의 공식 안내를 기준으로 한 전략 판단이다. 심사 결과의 예측이 아니다.

1. **첫 검토 후보: BMC Bioinformatics의 Software article.** 기존 방법을 연결한 도구라도 넓은 활용성과
   유의미한 개선을 입증해야 한다. [공식 Software article 안내](https://link.springer.com/journal/12859/submission-guidelines/software-article)는
   보통 기존 도구와의 직접 비교를 요구한다. 계산량이 큰 P1–P3 비교는 통계적 주장과 선택 효과의 실험이므로,
   ESurv·surviveR 같은 사용자 도구와의 기능·과제 수행 비교를 대체하지 못한다. 재현 기록, 실제 사용자 오류,
   독립 검증을 올바르게 수행하도록 돕는 기능을 중심으로 구성한다.
2. **조건부 상향 후보: Bioinformatics Application Note.**
   [공식 안내](https://academic.oup.com/bioinformatics/pages/author-guidelines)는 새 소프트웨어의 중요한 기여와
   넓은 사용·설치 가능성을 요구하며 형식은 4페이지로 제한된다. 검증 가능한 작업 흐름 개선과 직접 비교가
   충분해질 때 검토한다. 기존 통계 기능의 개수나 DL 모델 수만으로 경쟁력을 판단하지 않는다.
3. **소프트웨어 보존 경로: JOSS.** [공식 제출 조건](https://joss.readthedocs.io/en/latest/submitting.html)은
   공개 개발의 최소 6개월 이력, 연구 활용·영향, 공동체와 유지보수의 근거를 포함한다. 저장소 생성일
   2026-03-27은 확인했지만 생성일이 당시부터 공개였다는 증거는 아니다. 실제 공개 이력과 연구 활용을 먼저
   확인해야 한다. 새 통계·생물학적 발견을 대신 발표하는 쉬운 예비 경로로 간주하지 않는다.

우선순위는 (1) 모든 예정 계산의 성공·실패 확정과 결과 일관성, (2) 실제 경쟁 사용자 도구와의 기능/과제 비교,
(3) 승인된 사람 대상 연구, (4) 본문·legend·source table·영구 보존 패키지의 일치이다. 현재 공개 웹서비스의
기능을 직접 시험하지 않은 항목은 미확인으로 둔다. 인간 대상 결과나 실사용 영향은 만들어 넣을 수 없다.

[직접 비교의 접근 기록](competitor_access_20261002.md)에는 ESurv의 이 환경에서의 DNS 오류와 surviveR의
로그인 페이지 도달을 기록했다. 문헌의 기능 설명과 실제 현재 기능 시험을 구분한다. Group Cox HR을
age/sex/stage 다변량 Cox와 같은 기능으로 취급하지 않는다. 실제 공통 과제의 수행 가능성을 확인한 뒤
사용자 연구의 비교 과제를 고정해야 한다.
KM Plotter의 custom-data 화면과 공개 예제는 접근 가능했고 다변량 옵션도 직접 확인했다. 숫자 분석은 제출하지
않았으므로 task compatibility와 결과 일치를 확인한 것으로 보고하지 않는다.

## 사람이 수행해야 하는 단계

[사용자 연구 protocol](user_study_protocol_20261002.md)과 `user_study_pack/`의 A/B 합성 과제, 독립 R 정답,
빈 관찰표를 준비했다. 실제 윤리 심사 판단, 책임자와 동의·보상·보관기간, 비교 도구 실행·버전 확인, 파일럿과
본 연구의 모집·관찰이 필요하다. 이 준비를 실제 사용성·시간·오류 감소 결과로 서술하지 않는다.

논문 전체 계산과 비교 도구 실험의 완료·실패 기록, 그림 검토와 최종 hash를 이 문서에 갱신한다.
