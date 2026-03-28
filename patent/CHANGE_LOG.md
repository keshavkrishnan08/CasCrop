# CasCrop — Verifiable Invention Change Log

## Purpose

This document serves as a contemporaneous record of the conception, development, and reduction to practice of the inventions described in U.S. Provisional Patent Application: **"System and Method for Predicting Agricultural Crop Waste Using Economic Contagion Graph Neural Networks with Asymmetric Shock Conditioning."**

This log is intended to establish priority dates, document the inventive process, and provide a verifiable chain of evidence for patent prosecution. All entries are corroborated by git commit hashes, which provide cryptographic timestamps of each development milestone.

---

## Inventor

**Name:** Keshav Krishnan
**Role:** Sole Inventor
**Affiliation:** Independent Researcher

---

## Prior Work Establishing Foundation

### Published Works by Inventor

1. **Complexity Economics Book** (published)
   - 300+ pages establishing theoretical foundation for understanding economies as complex adaptive systems
   - Introduced the framework of cascading failures in economic networks that directly motivated this invention

2. **Nature Climate Change Paper** (under review)
   - Proves American agriculture is shifting northward due to climate change
   - Quantifies $150 billion economic effect
   - Establishes that climate-driven geographic shifts create new inter-regional economic dependencies

3. **Nature Food Paper** (under review)
   - Proves commodity market structures (e.g., CME exchange locations) create $5 billion in annual welfare costs for farmers
   - Identifies the specific market mechanisms through which economic shocks propagate between regions
   - Directly motivates the contagion modeling approach in CasCrop

4. **CropConnect** (active startup)
   - B2B marketplace improving agricultural distribution for small farmers
   - Provided firsthand domain knowledge of how market forces cause crop waste
   - Real-world observations of farmers abandoning healthy crops due to price depression inspired the core thesis

---

## Invention Timeline

### Phase 0 — Conception

**Date:** Prior to 2026-03-28
**Description:** The core inventive concept — that crop waste propagates across regions through economic channels analogously to financial contagion, and that this can be modeled using graph neural networks with directional shock conditioning — was conceived by Keshav Krishnan based on:

1. Observation that existing crop prediction models universally treat each region independently
2. Recognition from the Nature Food paper that commodity market structures create economic interdependencies between farming regions
3. Insight from the complexity economics book that cascading failures in networks require directional (asymmetric) modeling because positive and negative shocks propagate through fundamentally different mechanisms
4. Identification that USDA RMA crop insurance claims provide labeled training data distinguishing biophysical waste (drought, frost) from economic waste (price decline)

**Key Inventive Insights:**
- **Insight 1 (ECMP):** Standard graph attention mechanisms treat all neighbor signals symmetrically. In agricultural economics, a price DROP (from oversupply) causes waste contagion, while a price SPIKE (from undersupply) REDUCES waste. Separate learned transformations for positive and negative shocks (W_pos, W_neg) are necessary to capture this asymmetry. This does not exist in any prior graph neural network architecture.
- **Insight 2 (Disentanglement):** Because weather patterns are spatially correlated and affect both crop health and commodity prices, a naive model combining satellite and price data would confuse shared weather effects with genuine economic contagion. Adversarial disentanglement forces the biophysical and economic encoders to capture independent information, enabling verification that the graph captures true market contagion.
- **Insight 3 (Waste ≠ Yield):** Crop waste is a fundamentally different prediction target from crop yield. A high-yield crop can be entirely wasted if prices crash. Targeting waste as the prediction objective, using insurance claims as labels, is novel.

---

### Phase 1 — Architecture Design and Implementation

**Date:** 2026-03-28
**Git Commit:** `e6b5aef`
**Description:** Complete implementation of the CasCrop system architecture, including:

#### 1.1 Novel Mechanism: Economic Contagion Message Passing (ECMP)
- **File:** `src/models/graph/ecmp.py` (321 lines)
- **Implementation:** Asymmetric shock embedding φ(Δp) = W_pos · max(Δp, 0) + W_neg · min(Δp, 0)
- **Architecture:** Multi-head attention (H=4) with per-edge attention weight extraction
- **Key Design Decision:** Pure PyTorch scatter operations (no torch_scatter dependency) for broad compatibility
- **Residual stack:** Two ECMP layers with skip connections and LayerNorm

#### 1.2 Novel Mechanism: Biophysical-Economic Disentanglement
- **File:** `src/models/encoders/disentanglement.py`
- **Implementation:** Gradient reversal layer (GRL) for end-to-end adversarial training
- **Discriminator:** 3-layer MLP attempting to predict z_econ from z_bio
- **Verification method:** Linear probe accuracy target < 55% (random = 50%)

#### 1.3 Full CasCrop Architecture
- **File:** `src/models/cascrop.py` (234 lines)
- **Pipeline:** BiophysicalEncoder → EconomicEncoder → DisentanglementModule → ECMP Stack → WasteClassifier + CauseClassifier
- **Loss:** L_total = L_waste + 0.3 · L_cause + 0.1 · L_disentangle
- **Model size:** ~209K trainable parameters

#### 1.4 Ablation Baselines (for proving each component matters)
- **Row 1:** Local-only MLP (no graph, no economics) — `src/models/baselines/local_only.py`
- **Row 2:** Local + economic features (no graph) — `src/models/baselines/local_econ.py`
- **Row 3:** Geographic GAT (standard attention, no shock conditioning) — `src/models/baselines/geo_gat.py`
- **Row 4:** Symmetric ECMP (single W_sym instead of W_pos/W_neg) — `src/models/baselines/symmetric_ecmp.py`
- **Row 5:** Full CasCrop (asymmetric ECMP + disentanglement)

#### 1.5 Data Pipeline
- 9 data acquisition modules for freely available US agricultural data
- County-crop-month matching at FIPS granularity (2008-2024)
- Dynamic graph construction with learnable edge weight combination
- Temporal train/val/test split preventing data leakage

#### 1.6 Evaluation Infrastructure
- DeLong test for AUC comparison
- McNemar's test for error pattern comparison
- Bootstrap confidence intervals (10,000 iterations)
- Paired t-test and Wilcoxon signed-rank across seeds
- Case study cascade reconstruction from attention weights

#### 1.7 Training Infrastructure
- Focal loss for class imbalance (γ=2.0, α=0.75)
- Disentanglement warmup schedule (10 epochs without, then activate)
- Discriminator scheduling (5 steps per encoder step)
- Early stopping on validation AUC-ROC (patience=20)

---

### Phase 2 — Patent Application Drafting

**Date:** 2026-03-28
**Git Commit:** `aa1c1f5`
**Description:** Complete provisional patent application with:

#### 2.1 Claims
- **3 Independent Claims:**
  - Claim 1 (System): Complete CasCrop system
  - Claim 2 (Method): Prediction method steps
  - Claim 3 (ECMP — Domain-Independent): Asymmetric shock conditioning for ANY graph network
- **23 Dependent Claims** covering dynamic graphs, multi-head attention, focal loss, multi-task learning, warm-up scheduling, and domain-specific applications (finance, supply chain, epidemiology, energy)

#### 2.2 Prior Art Search
- 18+ specific prior art items analyzed across 6 categories
- No anticipation found for any of the three independent claims
- Freedom-to-operate analysis: low infringement risk
- Claim 3 (domain-independent ECMP) identified as strongest novelty position

#### 2.3 Formal Economic Model
- Proposition 1: Waste Contagion (with proof)
- Corollary 1: Asymmetric Propagation
- 3 Testable Hypotheses mapping theory to ablation experiments

---

### Phase 3 — Experimental Validation

**Date:** Pending
**Description:** Training all models, running ablation experiments, statistical tests, and generating publication-ready results. To be documented upon completion.

---

## Verification

All development artifacts are stored in a git repository with cryptographic commit hashes providing tamper-evident timestamps. The complete repository can be inspected to verify:

1. The implementation matches the patent claims
2. The development timeline is consistent
3. No prior art was incorporated after the conception date

### Git Commit Hashes (SHA-1)

| Commit | Date | Description |
|--------|------|-------------|
| `e6b5aef` | 2026-03-28T16:04:35-04:00 | Complete architecture implementation |
| `28cead1` | 2026-03-28T16:04:48-04:00 | Repository cleanup |
| `aa1c1f5` | 2026-03-28T16:12:11-04:00 | Patent application and prior art |

---

## Signatures

**Inventor:** Keshav Krishnan

**Date of this log:** 2026-03-28

**Witness:** _(To be signed by a witness who is not a co-inventor)_

**Witness Date:** ________________

---

## Notes for Patent Counsel

1. **First-to-file jurisdiction:** The US operates under first-to-file (AIA). This log establishes conception date and diligence toward reduction to practice.

2. **One-year grace period:** Under 35 U.S.C. § 102(b)(1)(A), the inventor has a one-year grace period from any public disclosure. If any of the related papers (Nature Climate Change, Nature Food) are published before filing, the provisional application should be filed within one year of the earliest publication.

3. **Claim 3 breadth:** The domain-independent ECMP claim (Claim 3) is intentionally broad. Patent counsel should assess whether additional dependent claims narrowing to specific domains (finance, epidemiology) strengthen or weaken the prosecution position.

4. **Continuation potential:** The provisional application supports multiple continuation applications:
   - Agricultural waste prediction system (Claims 1-2 + dependents)
   - Domain-independent asymmetric graph attention (Claim 3 + dependents)
   - Agricultural data fusion with adversarial disentanglement

5. **Trade secret consideration:** The specific hyperparameters (shock_embed_dim=8, num_heads=4, disentangle_lambda=0.1) and training procedures (warmup schedule, discriminator ratio) could be maintained as trade secrets rather than disclosed in the patent, if counsel advises.
