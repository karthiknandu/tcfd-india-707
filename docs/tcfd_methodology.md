# TCFD Framework Methodology

## Tripartite Causal Fairness Decomposition

### Theoretical Foundation

The TCFD framework extends Bareinboim et al.'s (2022) binary causal fairness classification
(acceptable vs. discriminatory paths) to a **tripartite taxonomy** applied to population-level
structural system design rather than individual algorithmic decisions.

---

## Formal Definition

Let M = ⟨V, U, F, P(u)⟩ be a Structural Causal Model where:
- V = endogenous variables (district health outcomes and determinants)  
- U = exogenous background variables (unobserved district factors)  
- F = structural equations V_i = f_i(PA(V_i), U_i)  
- P(u) = distribution over exogenous variables

The **do-operator** do(X = x) formalises intervention:
> P(Y | do(X = x)) ≠ P(Y | X = x) in general

---

## TCFD Pathway Types

### Type I — Structural-Physical Constraints

**Definition:** A causal path π from determinants to UHCd outcome is Type I if and only if
all intermediate nodes have parents exclusively in the geographic/physical exogenous set V_geo.

**Mathematical condition:**
> ∀ V_j ∈ π: parents(V_j) ⊆ V_geo = {terrain, altitude, area, remoteness, climate}

**Examples:** Geographic remoteness → travel time to hospital → institutional delivery rate → UHCd

**Policy implication:** These constraints **cannot be overcome by insurance expansion alone**.
Required: physical infrastructure investment (roads, telemedicine, mobile units).

**Operationalisation in master_final_v3.csv:**
- `geo_tax_index` (composite: 50% remoteness + 30% area + 20% shape complexity)
- `geo_dist_to_capital_km` (Haversine distance to state capital)
- `geo_compactness` (Polsby-Popper ratio)
- `census_pop_density`

---

### Type II — Historically-Produced Injustice Pathways

**Definition:** A causal path π is Type II if at least one intermediate node V_j has a parent
in the historical determinants set V_hist with R²(V_j ~ V_hist) > 0.5, where V_hist encodes
caste discrimination, gender exclusion, and colonial infrastructure gaps.

**Mathematical condition:**
> ∃ V_j ∈ π: ∃ H ∈ V_hist such that R²(V_j ~ H) > 0.5

**Examples:** SC/ST share → education deficit → women's agency → institutional delivery → UHCd

**Policy implication:** These require **redistributive, rights-based interventions** that address
historical exclusion — not technical programme fixes.

**Operationalisation:**
- `Women (age 15-49) who are literate4 (%)` — education equity pathway
- `Women age 20-24 years married before age 18 years (%)` — gender injustice pathway
- SC/ST population share (via SECC deprivation proxy) — caste exclusion pathway
- Sex ratio at birth — son preference / gender discrimination

---

### Type III — Policy-Actionable Levers

**Definition:** A causal path π is Type III if at least one intermediate node belongs to the
directly controllable policy set V_policy = {insurance coverage, programme delivery, workforce}.

**Mathematical condition:**
> ∃ V_j ∈ π: V_j ∈ V_policy AND do(V_j = v) is administratively feasible

**Examples:** PM-JAY enrollment → hospital utilisation → out-of-pocket reduction → UHCd

**Policy implication:** **Standard programme reform** can produce measurable near-term change.
PM-JAY expansion, ASHA worker density, DIU strengthening.

**Operationalisation:**
- `pmjay_treatment_flag` — PM-JAY policy treatment
- `diu_established` — governance infrastructure
- `Institutional births (%)`, `ANC4+ visits (%)`, `Full immunisation (%)`

---

## TCFD Attribution Algorithm

### Step 1: Train XGBoost with UHCd as target

```python
model = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                     random_state=42)
model.fit(X_train, y_train)
```

### Step 2: Compute SHAP value matrix

```python
explainer = shap.TreeExplainer(model)
Phi = explainer.shap_values(X)  # shape: (707, p)
```

### Step 3: Aggregate SHAP by TCFD type

For each district i and pathway type k:

```
Attribution_Share(type_k, district_i) = Σ_{j ∈ type_k} |Φ_{ij}| / Σ_j |Φ_{ij}|
```

### Step 4: K-means clustering on 3×707 attribution matrix

```python
X_cluster = np.column_stack([type_I_shares, type_II_shares, type_III_shares])
km = KMeans(n_clusters=5, random_state=42, n_init=10)
cluster_labels = km.fit_predict(StandardScaler().fit_transform(X_cluster))
```

---

## DiD Identification Strategy

### Treatment Variable
- `pmjay_treatment_flag = 1`: 33 states actively implementing PM-JAY (604 districts)
- `pmjay_treatment_flag = 0`: 3 opted-out states (Delhi, Odisha, West Bengal = 50 districts)

### Specification

```
UHCd_{st} = α_s + γ_t + β(PM-JAY_s × Post_t) + δX_{st} + ε_{st}
```

- α_s = district fixed effects
- γ_t = year fixed effects  
- β = ATT (Average Treatment Effect on the Treated)
- Staggered adoption (J&K/Ladakh from Dec 2020): Callaway-Sant'Anna (2021) estimator

---

## Transportability (Bareinboim et al. 2022)

For each target country c ∈ {Nigeria, Bangladesh, Kenya, Cambodia}:

1. Identify selection variables S that differ between India and country c
2. For each causal quantity Q in the TCFD framework, determine if Q is transportable:
   - Q is transportable if it can be estimated from India data and applied to country c
   - Q is non-transportable if S-nodes affect the causal mechanism

3. Apply transportable causal weights to analogous DHS indicators in target country
4. Compute Spearman ρ between predicted TCFD attribution shares and WHO UHC SCI

**Test:** H4 is supported if |ρ| ≥ 0.60 and permutation p < 0.05 for at least 3 of 4 countries.

---

## References

1. Bareinboim E, Correa JD, Ibeling D, Icard T. (2022). On Pearl's Hierarchy and the
   Foundations of Causal Inference. ACM. doi:10.1145/3501714.3501743

2. Pearl J. (2000). Causality: Models, Reasoning, and Inference. Cambridge University Press.

3. Callaway B, Sant'Anna PHC. (2021). Difference-in-differences with multiple time periods.
   Journal of Econometrics, 225(2), 200-230. doi:10.1016/j.jeconom.2020.12.001

4. Lundberg SM, Lee S-I. (2017). A unified approach to interpreting model predictions.
   NeurIPS, 30. arxiv:1705.07874
