# BVFF 연구 계획

> 이 문서는 Claude Code 세션이 참조하는 상위 로드맵이다.
> 작업 규칙: 코드를 수정하기 전에 반드시 변경 계획을 먼저 제시하고 승인을 받는다.
> 한 세션에서는 한 항목만 다루고, 완료 시 아래 체크박스를 갱신한다.

## 목표

강유전체 MD를 위한 Bond Valence Force Field(BVFF)를 확립하고 실제 연구로 검증한 뒤,
NequIP을 결합한 강화 BVFF(NequIP-BVFF)로 확장한다.

진행 순서: **Phase 1 (코드 확립) → Phase 2 (응용 검증) → Phase 3 (이론 확장) → Phase 4 (NequIP-BVFF)**

## 현재 코드 요약

- Python 3.12, `pymatgen`, `scipy`, `tomllib`
- 입력: `controls.toml`, `parameters.toml`; 데이터: VASP `vasprun.xml`
- 에너지: E_tot = E_repulsive + E_coulomb + E_BV + E_BVV (+ E_angle), `controls.toml`로 토글
- Ewald: `src/extensions/ewald.py` (실공간, 역공간, 표면항, 자기에너지)
- Fitting: multi-start least squares 기본 (SA 옵션), σ-정규화 손실, 종별 에너지 기준 profiling, V0 고정
- Cutoff: quintic (C²)
- 구조: `io/`, `src/main.py`, `src/potentials.py`, `src/fitting.py`, `scripts/analysis.py`

---

## Phase 1. BVFF 확립

### 1-1. 코드 마무리
- [ ] `__init__.py`와 import 경로 정리 (어느 디렉토리에서 실행해도 동작)
- [ ] 장시간 계산의 진행 출력: `flush=True`, fitting 반복별 로깅, vasprun 읽기/에너지 계산 진행 표시
- [ ] `lattice_vectors @ frac_vector` 방향 문제 검토 (행 벡터/열 벡터 규약 통일). 단사정·삼사정 셀 테스트 케이스로 확인

### 1-2. 단위 검증 (`tests/` 신설)
- [ ] **미분 일관성**: 무작위 배치에서 힘 = −∂E/∂r, 응력 = (1/Ω)∂E/∂ε 를 중심차분과 비교 (상대 오차 < 1e-6)
- [ ] **Ewald 검증**: NaCl 구조의 Madelung 상수(1.7476) 재현; 실공간/역공간 분할 파라미터 α를 바꿔도 에너지 불변
- [ ] **Cutoff 연속성**: quintic cutoff에서 에너지, 1차, 2차 도함수가 r_c에서 0
- [ ] **문헌 재현**: 문헌 BVFF 파라미터(BaTiO₃ 또는 PbTiO₃)로 격자상수, 분극 방향, 상 안정성 순서 재현
- [ ] **Fitting 재현성**: 합성 데이터(알려진 파라미터로 생성)에서 파라미터 복원

### 1-3. LAMMPS 연동
- [ ] C++ 플러그인 (pair style) 완성
- [ ] 같은 구조에서 Python 구현과 힘·에너지 일치 확인
- [ ] NPT/NVT 안정성 (에너지 보존, 온도 제어)

---

## Phase 2. BVFF 활용 연구 (검증)

### 2-1. BaTiO₃ 기준
- [ ] AIMD 데이터 생성 (여러 온도, 여러 상)
- [ ] Fitting 후 MD로 상전이 순서(C→T→O→R)와 T_c 재현
- [ ] 문헌 BVFF 결과와 비교

### 2-2. HZO
- [ ] Pca2₁ 상 안정성, 스위칭 장벽, 자발 분극
- [ ] P–E 이력곡선, 도메인월 에너지
- [ ] AIMD와 직접 비교

### 2-3. 실패 지점 목록
- [ ] BVFF가 재현하지 못하는 물리를 정리 → Phase 4에서 NN이 보완할 대상

---

## Phase 3. 이론 확장 (ML 없이 가능)

이론 배경은 아래 "이론 요약" 참조.

- [ ] Madelung 보정항 Δᵀφ⁰ 추가 (V⁰ → V⁰ − φ⁰/J 에 해당). `controls.toml` 토글
- [ ] 별도의 쌍극자 결합가 a_b = A_d exp(−r/b_d) r̂_b 도입, Z\*·ε∞ 계산 루틴 추가
- [ ] DFPT Z\*, ε∞로 A_d, b_d, κ, κᵈ, λ, J fitting (단계 A′)
- [ ] 이론 논문: BVFF를 결합 흐름 범함수의 극한으로 유도

---

## Phase 4. NequIP-BVFF

- [ ] numpy/scipy → PyTorch 전환 (자동미분)
- [ ] NequIP 모듈: BV feature (V_i: 0e, W_i: 1o), δθ head, E_Δ head
- [ ] 단계적 학습 A → A′ → B → C
- [ ] Fisher 정보 행렬로 데이터셋 식별성 진단
- [ ] LAMMPS: TorchScript 기반 연동 + PPPM
- [ ] Phase 2 실패 지점의 개선 여부로 성능 판단

---

## 이론 요약 (Phase 3, 4의 근거)

### 통합 범함수
결합 b의 전하 흐름 p_b(스칼라)와 쌍극자 흐름 t_b(벡터)를 변수로 둔다.
접속행렬 B에 대해 q = Bp, 국소 비대칭 m = Bt, 총 쌍극자 D = Σ_b (t_b − p_b R_b).

E = Σ_i ½J_i(q_i − Q_i)² + Σ_i f_i(|m_i|²) + Σ_b [½κ u_b² + ½κᵈ|v_b|² + λ u_b (r̂_b·v_b)] + ½yᵀ𝕄y

- u = p − s(r), v = t − a(r); s는 결합가, a = A_d exp(−r/b_d) r̂
- f_i = α|m|² + β|m|⁴ (Landau; α < 0 이면 SOJT 불안정)
- 𝕄 = [[C, Γ],[Γᵀ, T]]: Ewald 전하–전하, 전하–쌍극자, 쌍극자–쌍극자 (tin-foil 경계)
- 안정성: λ² < κκᵈ, 전자 Hessian 𝓗 양정치

### 극한
- κ, κᵈ → ∞ 이면 E_BV = ½ΣJ(V − V⁰)², E_BVV = D(|W|² − W₀²)² 로 BVFF를 정확히 재현 (S_i = J_i/2, α = −2DW₀²/ℓ², β = D/ℓ⁴)
- BVFF가 생략하는 항: Δᵀφ⁰ (Madelung), 결합망 상관 −½Σ(1/κ)|ν_i − ν_j|², W·E⁰ 및 탈분극
- 유한 κ: E\* = ½Δᵀ(J⁻¹ + BK⁻¹Bᵀ)⁻¹Δ (원자가 불일치의 결합망 차폐)

### Born 유효전하와 ε∞
Z\*_k = q_k I + 𝒅ᵀ(1 + χ_loc𝕄)⁻¹[δy⁰_k − χ_loc(∂𝕄/∂r_k)y]
ε∞ = I + (4π/Ω)𝒅ᵀ(χ_loc⁻¹ + 𝕄)⁻¹𝒅

- χ_loc = P𝓗_loc⁻¹Pᵀ, P = diag(B, I), 𝒅 = ∂D/∂y
- 강체 극한(양이온 k): Z\* = Σ_j s_kj[(1 + (ℓ − r)/b) r̂r̂ᵀ − ((ℓ − r)/r)(I − r̂r̂ᵀ)]
  → 전하 흐름(−r/b)은 Z\*를 줄이고, 결합 쌍극자(+ℓ/b)는 키운다. 이상 증폭에는 결합 쌍극자 우세가 필요
- 음향 합규칙 Σ_k Z\*_k = 0 은 항등적으로 성립

### 식별성
- 정확한 축퇴: ℓ과 r_d → A_d = ℓ e^{r_d/b_d} 하나로 재매개화
- 약한 축퇴: J–κ (E와 Z\* 동시 사용으로 분리), λ (저대칭 구조의 Z\* 비등방성 필요), α (에너지와 Z\*에서 과결정 → 검증 수단)
- 학습 전 Fisher 정보 행렬 스펙트럼으로 데이터셋 진단

### 학습 손실
L = Σ_O w_O L_O + λ_θΣ‖δθ‖² − μ log(κκᵈ − λ²) − μ log λ_min(𝓗),
O ∈ {E, F, σ, Z\*, χ = (ε∞ − I)Ω/4π, ΔP(경로)}; 각 항 데이터 표준편차로 정규화.
DFPT Z\*는 ASR 강제 후 사용. ΔP는 연속 경로 위 차이만 사용.
