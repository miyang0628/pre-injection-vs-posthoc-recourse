# 전체 기획안 (Project Plan)

가드레일 제약 기반 반사실적 recourse 프레임워크 — 재설계 기획서

> 이 문서는 다음 세션에서 이어서 작업하기 위한 맥락 인수인계용이다. 무엇을, 왜, 어떻게
> 바꿨는지와 남은 작업을 담는다. 실제 수치는 `ANALYSIS_RESULTS.md`를 참조.

---

## 1. 배경: 왜 재설계했나

### 1.1 최초 상황
기존 원고("LLM-Guided Counterfactual Explanation for Comorbid Chronic Disease Risk
in Health Insurance")는 한 경영·경제학 저널에서 게재 거절(reject)됨. 편집자 지적 세 가지:

1. **탐색적 성격** — 기존 문헌 대비 진전 평가가 어렵다.
2. **자기 선행연구 미인용** — 저자 본인의 유사 선행 논문과 기본 아키텍처
   (KNHANES, 성별 XGBoost, DiCE, 보험 맥락 LLM)가 거의 동일한데 인용·차별화가 불충분.
3. **LLM 기여 미검증** — 핵심인 LLM 가드레일이 실증되지 않았고, 코호트 평가에서 LLM이
   결정론적 규칙으로 대체됨. 그 규칙이 hard constraint라 위반율 0%는 "설계상 당연(by
   design)"하다.

### 1.2 진단
지적 3은 반박 불가다. Hard constraint로 위반을 0으로 만드는 건 쉽다. 방어하려 하면 안
되고, **프레임을 옮겨야** 한다. 위반율 0%를 결과가 아니라 sanity check로 강등하고, 진짜
기여를 재정의하는 것이 핵심 전략.

가장 결정적인 구조적 문제: 기존 코드의 `apply_physical_rules`가 LLM 출력을 **무조건
덮어씀**. 이 구조에서는 LLM이 무엇을 하든 최종 제약이 룰과 같아져, "LLM 기여"를 실증할
방법이 원천적으로 없었다. 그래서 실험을 덧붙이는 것으로는 불가능하고, **아키텍처부터
다시** 짜야 했다.

---

## 2. 재설계 핵심 결정

### 2.1 3층 가드레일 구조 (핵심 아키텍처 변경)
LLM override를 안전선으로 좁혀, LLM 재량을 살린다.

- **Layer 1 (SAFETY)** — 절대 생리학적 하한 + 해부학적 방향 일관성. **항상 코드로 강제**,
  LLM도 룰도 못 건드림. 예: 에너지 500kcal, 나트륨 800mg(WHO 최소), BMI·허리·체중 동시
  감소 커플링.
- **Layer 2 (DISCRETION)** — "얼마나 움직일지"의 판단 구간. LLM 또는 룰이 채움. **LLM vs
  룰 차이(claim a)가 발생하는 층.**
- **Layer 3 (FIXED)** — non-modifiable 변수(스트레스·소득·교육 등). 현재값 고정.

이 구조 덕분에 LLM과 룰이 Layer 2에서만 갈리고, 안전성은 Layer 1이 양쪽 동일하게 보장.

### 2.2 위임 수준(delegation) ablation
Layer 1/Layer 2 경계 = LLM에게 재량을 얼마나 위임하느냐. 이걸 실험 변수로.
- **conservative**: 나트륨·에너지를 코드로 통째 고정, LLM은 BMI·지방·당만.
- **aggressive**: 절대 안전바닥만 코드, 나머지 LLM 재량.

### 2.3 두 개의 검증 축
- **(a) LLM ≠ 룰**: LLM 가드레일이 룰의 재현이 아님을 정량화 (구간 차이 → recourse 차이).
- **(b) 사전주입 > 사후필터**: 제약을 어떻게 넣느냐(pre-injection vs post-hoc filtering)가
  feasibility를 가른다. 위반율이 아니라 feasibility가 요점.

### 2.4 모델 티어 확장 (a의 확장)
weak/mid/strong 세 모델(gpt-4o-mini / gpt-5.4-mini / gpt-5.4)로 (a)를 확장. 모델 능력과
가드레일 품질의 관계를 3축(발산도 IoU / recourse 품질 / 안전바닥 침범)으로 측정.

---

## 3. 실험 설계

### 3.1 4조건 비교 (핵심, claim b)
| 조건 | 제약 처리 |
|---|---|
| C0 Pure DiCE | unconstrained, Class 0 타깃, fallback 없음 |
| C1 Post-hoc filter | unconstrained 후 안전 위반 후보 제거 |
| C2 Pre-injection | 가드레일 사전 주입 + stepwise fallback (C0→C1→C2) |

주 지표: feasibility, 유효 후보 수, 다양성. 위반율은 sanity check.

### 3.2 모델 티어 비교 (claim a)
전체 Class 3 코호트 × 3티어, paired. 3축:
- Axis 1: 룰 대비 IoU (낮을수록 덜 룰-like)
- Axis 2: recourse 품질 (feasibility, 변경변수)
- Axis 3: raw 안전바닥 침범 (코드 교정 전, 높을수록 덜 self-safe)

### 3.3 데이터·모델 (Stage 1, 유지)
- KNHANES 2020–2024 병합, 9,738 표본, 4클래스.
- 성별 XGBoost (DiCE가 sklearn 호환 부스팅 트리를 요구 → XGBoost 유지가 적절).
- 편집자가 Stage 1은 지적 안 함 → 그대로 재사용.

---

## 4. 노트북 구성 (6개)

각 노트북: 단일 셀, 최상단 영문 노트북명 주석, 첫 셀에 전체 코드.

| # | 파일 | 역할 | API 키 |
|---|---|---|---|
| 01 | data_preprocessing | 병합·라벨·영문 변수명 | 불필요 |
| 02 | xgboost_modeling | 성별 XGBoost + 평가 | 불필요 |
| 03 | representative_cases | 대표 케이스 (LLM+DiCE) | 선택 |
| 04 | condition_comparison | ★ 4조건 코호트 (b), 자체 실행 | 불필요 |
| 05 | llm_vs_rule_guardrails | ★ 모델 티어 비교 (a) | **필요** |
| 06 | actuarial_projection | illustrative cost sizing | 불필요 |

공유 모듈: `guardrail_core.py`(3층 로직 + IoU 지표), `experiment_core.py`(C0–C2 평가기).

### 4.1 노트북별 구현 주의점
- **04**: 코호트 루프를 노트북 안에서 직접 실행하고 성별 캐시 CSV에 저장. `FORCE_RERUN`으로
  재계산. 첫 실행 ~25분(남 10 + 여 15).
- **05**: `.env`에서 키+모델 티어 로드(`load_dotenv(find_dotenv())`). 전체 155명 ×
  3티어 = 465 호출. **증분저장+재개** 내장(환자별 CSV append, 완료 pid skip, 재개 중복
  dedup). GPT-5.x는 Responses API로 자동 분기. strong(gpt-5.4) 비용 주의.
- **06**: 04가 만든 `experiment_cohort.csv`의 C2-aggressive 전환율을 읽어 offset 계산.

---

## 5. 재현성·환경 주의사항

- **pkl 배포 금지**: joblib 피클은 pandas/numpy 버전에 묶임. 다른 환경에서 열면
  `NotImplementedError`. 반드시 01→02를 각자 환경에서 돌려 로컬 생성. `.gitignore`에 제외.
- **원자료(SAS)**: KDCA 배포. 재배포 불가. `data/`에 직접 배치.
- **`.env`**: 절대 커밋 금지(`.gitignore` 등록). 키는 로컬에만.
- **실행 순서**: 01 → 02 → 03 → 04 → 06 (05는 키 필요, 독립 실행 가능).

---

## 6. 남은 작업 (다음 세션 인수인계)

### 6.1 즉시 가능
- [ ] **(a)+(b) 논문 결과 섹션 초안** 작성 (방법·결과·한계). `ANALYSIS_RESULTS.md`의
      검정 수치 사용.
- [ ] 노트북 04 그림의 C1 NaN 라벨 경고 제거 (사소, 선택).
- [ ] 검정 수치를 논문용 표로 (Friedman/Wilcoxon p값).

### 6.2 논문 서술 방침 (확정됨)
- **무게중심은 (b)에.** n=155로 통계적으로 탄탄. feasibility 0% vs 98.7%.
- **(a)는 IoU까지만 강하게.** "LLM ≠ 룰"(p<10⁻²⁶)은 확실. **세대 격차는 주장 금지**
  (Axis 3 Friedman p=0.079, 비유의). 축 3 약한 결과는 "어느 모델도 침범을 못 없애니 Layer 1
  안전장치가 필수"로 전환.
- **IoU 비단조성은 있는 그대로 보고** (확정된 방침).
- **횡단면 한계 선제 인정.** cross-sectional proof-of-concept로 못박고, transition을
  claims 아닌 classifier 예측으로 일관되게 표현.

### 6.3 배포 전 필요 (논문 limitation)
- 종단 코호트 검증 (claims linkage).
- 가드레일 값 임상 전문가 검토 (내분비내과·영양사·언더라이터).
- 공정성/규제 (연령·사회경제·인종별 subgroup 분석).

### 6.4 편집자 지적 2 (자기 선행연구) 대응 — 미해결
저자의 유사 선행 논문과의 차별화를 본문에서 명시적으로 서술해야 함. 재설계된 (a)/(b)가
선행연구 대비 무엇이 새로운지(3층 가드레일, 사전주입 vs 사후필터, 모델 티어 분석)를
Related Work에서 분명히 할 것. **이 부분은 아직 원고에 반영 안 됨.**

---

## 7. 파일 트리

```
llm-cf-comorbid/
├── PROJECT_PLAN.md          # 이 문서
├── ANALYSIS_RESULTS.md      # 결과·검정 수치
├── README.md                # GitHub용 (익명)
├── requirements.txt
├── .gitignore
├── data/  (README만; SAS·pkl은 로컬 생성)
├── notebooks/  (01–06 + guardrail_core.py + experiment_core.py)
└── results/
    ├── figures/  (png+pdf, 회색조, dpi 600)
    └── tables/   (LaTeX + CSV/JSON)
```
