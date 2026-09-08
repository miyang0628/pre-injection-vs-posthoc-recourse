# 분석 결과 (Analysis Results)

KNHANES 2020–2024. 6개 노트북 전부 실제 데이터로 실행 완료. 통계 검정 포함.

> 서술 방침: (b)를 주력, (a)는 IoU까지만 강하게, 세대 격차 주장 회피, IoU 비단조성은
> 있는 그대로 보고, 횡단면 한계 선제 인정. 근거는 아래 수치.

---

## 0. 데이터·모델 재현 (노트북 01–02)

| 항목 | 값 | 논문 대비 |
|---|---|---|
| 병합 표본 | 9,738 | 정확히 일치 |
| Class 0 / 1 / 2 / 3 | 6,381 / 1,370 / 649 / 1,338 | 정확히 일치 |
| 모델 확진 Class 3 (검증셋) | 남 89, 여 66 (총 155) | 근접 (논문 89/66) |
| Class 3 recall | 남 ~0.61, 여 ~0.55 | 근접 (논문 0.62/0.53) |

전처리·모델 논문 정확히 재현. Stage 1(성별 XGBoost) 수정 불필요.

---

## 1. (b) 제약 주입 방식 — 노트북 04 (n=155)

4조건 비교, 전체 Class 3 코호트.

| 조건 | Feasibility | 유효 후보 | 위반율 | 다양성 | 변경변수 |
|---|---|---|---|---|---|
| C0 Pure DiCE | 100% | 3.94 | 91% | 15.2 | 15.3 |
| **C1 Post-hoc filter** | **0%** | 0.00 | — | — | — |
| C2 Pre-injection | 98.7% | 3.41 | 0% | 5.2 | 10.6 |

성별 분해:

| 성별 | 조건 | Feasibility | 유효후보 | 위반율 |
|---|---|---|---|---|
| 남 | C0 / C1 / C2 | 100% / 0% / 97.8% | 3.91 / 0 / 3.38 | 92% / — / 0% |
| 여 | C0 / C1 / C2 | 100% / 0% / 100% | 3.97 / 0 / 3.46 | 89% / — / 0% |

**해석.** 사후필터(C1)는 feasibility 0%로 붕괴 — unconstrained 후보가 안전 제약을 거의
전부 위반해 필터 후 남는 recourse가 없음. 사전주입(C2)만 recourse 보존. 위반율 0%는
사후·사전 모두 정의상 달성 가능("by design")하므로 sanity check로 강등, 실질 기여는
feasibility로 재정의. **편집자의 "by design" 지적을 정면으로 넘어섬.**

**위임 ablation.** aggressive vs conservative 거의 동일 (양쪽 ~98–100% feasible) →
안전바닥만 코드 강제해도 feasibility 견고.

---

## 2. (a) LLM vs 룰, 모델 티어별 — 노트북 05 (n=155 × 3티어, paired)

weak=gpt-4o-mini, mid=gpt-5.4-mini, strong=gpt-5.4.

### Axis 1 — 룰과의 발산 (IoU)

| 티어 | mean | sd | median |
|---|---|---|---|
| weak | 0.429 | 0.107 | 0.433 |
| mid | 0.376 | 0.076 | 0.364 |
| strong | 0.402 | 0.078 | 0.403 |

**검정:**
- LLM ≠ 룰: 세 티어 모두 IoU가 1.0과 다름 (Wilcoxon vs 1.0, **p = 3.5×10⁻²⁷**).
- 티어 간 차이 유의 (Friedman **χ²=21.25, p = 2.4×10⁻⁵**), 그러나 **비단조**:
  - weak vs mid: p = 3.3×10⁻⁸ (median diff +0.062)
  - weak vs strong: p = 0.0062 (+0.026)
  - mid vs strong: p = 0.0023 (−0.028)
  - → mid가 최저(가장 룰과 다름), weak 최고. 모델 크기에 단조 아님.
- 재현성: 세 번 실행(n=10, 20, 155) 모두 IoU ≈ 0.38–0.43 수렴.

### Axis 2 — recourse 품질

| 티어 | feasibility | 변경변수 mean |
|---|---|---|
| weak | 96.1% | 13.95 |
| mid | 94.8% | 11.58 |
| strong | 97.4% | 14.83 |

**검정 (변경변수):** Friedman **χ²=59.14, p = 1.4×10⁻¹³** (강하게 유의), 비단조. mid가
가장 sparse (weak vs mid p=3.7×10⁻⁹, mid vs strong p=1.0×10⁻¹⁴).

### Axis 3 — raw 안전바닥 침범 (코드 교정 전)

| 티어 | mean | sd | ≥1회 침범 | max |
|---|---|---|---|---|
| weak | 0.516 | 1.040 | 26.5% | 5 |
| mid | 0.297 | 0.740 | 21.3% | 5 |
| strong | 0.374 | 0.731 | 28.4% | 5 |

**검정:**
- **Friedman χ²=5.08, p = 0.079 → 전체 비유의.**
- weak vs mid만 유의 (p = 0.0016), weak vs strong 경계선 (p = 0.062),
  mid vs strong 비유의 (p = 0.077).
- weak vs mean(mid, strong): p = 0.038 (겨우 유의).
- 세 티어 모두 median 0, max 5. 침범은 소수 환자(21–28%)의 드문 사건. strong이
  오히려 mid보다 침범 잦음.
- **소표본(n=20)의 6배 격차는 과대평가. 대표본에서 1.5배로 축소.**

### (a) 결론

세 축 모두에서 **mid(gpt-5.4-mini)가 특이점**: 룰과 가장 다르고, 가장 안전하고, 가장
sparse. "능력 단조 증가" 스토리와 불일치. 따라서:

> **확정 서술.** LLM 가드레일은 모든 티어에서 룰의 재현이 아니다(IoU ≈ 0.4, 1.0과
> p<10⁻²⁶로 다름). 티어에 따라 제약이 유의하게 달라지지만(Friedman p<10⁻⁴) 그 관계는
> 모델 크기에 단조가 아니다. 어떤 티어도 안전바닥 침범을 완전히 없애지 못하며(21–28%
> 환자, 최대 5회, 티어 간 차이 비유의), 이는 모델 능력과 무관하게 코드 레벨 안전장치
> (Layer 1)가 필수임을 보여준다.

**금지:** 세대별 안전성 우위 주장 (Axis 3 비유의). **강조:** IoU로 LLM≠룰 확정, 비단조성을
발견으로, Axis 3 약함을 Layer 1 정당성으로 전환.

---

## 3. Actuarial projection (illustrative) — 노트북 06

외부 파라미터 기반. KNHANES claims 없음 → order-of-magnitude sizing.

비용 gradient (Class 0 기준):

| Tier | Cost index | Annual USD | Excess vs C0 |
|---|---|---|---|
| Class 0 | 1.00 | 2,572 | — |
| Class 1 | 1.35 | 3,473 | 900 |
| Class 2 | 1.59 | 4,090 | 1,518 |
| Class 3 | 2.56 | 6,585 | 4,013 |

Recourse cost-offset (코호트 전환율 반영):

| Adherence | Male | Female |
|---|---|---|
| Full (100%) | $3,865 | $3,958 |
| Moderate (50%) | $1,933 | $1,979 |
| Low (30%) | $1,160 | $1,187 |

comorbid 정책자당 연 ~$4,000 초과비용, 부분 순응만으로도 상당 상쇄 시사. **횡단면이라
예측 전환을 claims로 검증 불가 — 검증된 pricing 아님.**

---

## 4. 편집자 지적 대응

| 지적 | 대응 | 근거 |
|---|---|---|
| LLM 기여 미검증 | LLM ≠ 룰 확정 | 노트북 05 Axis 1, p<10⁻²⁶ |
| 위반율 0% by design | feasibility로 축 전환 | 노트북 04, C1 0% vs C2 98.7% |
| 탐색적, 진전 평가 난 | 4조건 베이스라인 + 대규모 검정 | 노트북 04, n=155 |
| 자기 선행연구 미인용 | **미해결** — Related Work 차별화 필요 | (다음 세션) |

---

## 5. 한계 (논문 명시)

1. **횡단면 데이터** (최우선). 개인 추적 PK 없음. 예측 transition은 within-classifier
   진술, claims 검증 불가. Actuarial은 illustrative.
2. **Axis 3 약함.** 모델 세대별 안전성 격차 비유의 → 세대 우위 주장 회피.
3. **도메인 검증 부재.** 가드레일 값 임상 전문가 미검토. 안전 하한은 PoC heuristic.
4. **표본.** Class 3 확진 155명(검증셋). 종단 코호트 검증이 배포 전 필수.

---

## 6. 핵심 수치 재현 명령

```python
import pandas as pd
from scipy import stats
df = pd.read_csv("results/tables/llm_tier_divergence.csv")
iou = df.pivot_table(index="pid", columns="tier", values="mean_iou").dropna()
brk = df.pivot_table(index="pid", columns="tier", values="raw_safety_breaches").dropna()
# Axis1 Friedman
print(stats.friedmanchisquare(iou["weak"], iou["mid"], iou["strong"]))
# Axis3 Friedman
print(stats.friedmanchisquare(brk["weak"], brk["mid"], brk["strong"]))
```
