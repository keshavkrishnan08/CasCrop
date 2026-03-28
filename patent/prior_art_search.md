# Prior Art Analysis for CasCrop Patent Application

## Search Methodology

This prior art search was conducted across the following databases and repositories:

- **USPTO Full-Text and Patent Application Full-Text databases** (patents.google.com, patft.uspto.gov)
- **Google Patents** (patents.google.com) --- worldwide patent coverage
- **IEEE Xplore** --- engineering and computer science publications
- **ACM Digital Library** --- computing research publications
- **arXiv** --- preprints in machine learning, computer science, and quantitative agriculture
- **Google Scholar** --- cross-disciplinary academic publications
- **Scopus** --- multidisciplinary scientific literature

### Search Terms and Queries

Primary queries:
- "crop waste prediction" AND ("graph neural network" OR "graph attention")
- "agricultural contagion" AND "machine learning"
- "asymmetric" AND "graph attention" AND "shock"
- "crop insurance" AND "prediction" AND "neural network"
- "economic contagion" AND "graph" AND "agriculture"
- "disentanglement" AND "agriculture" AND "biophysical"
- "directional" AND "message passing" AND "graph"
- "supply chain" AND "cascade" AND "graph neural network"
- "financial contagion" AND "graph attention"
- "asymmetric propagation" AND "network"

Classification-based searches (CPC codes):
- G06N3/08 (neural network learning methods)
- G06Q50/02 (agriculture; fishing; forestry; mining)
- G06Q40/08 (insurance)
- G06N3/045 (graph neural networks)

---

## Identified Prior Art

### Category 1: Graph Attention Networks and Graph Neural Networks

#### 1.1 Graph Attention Networks (GAT)
- **Citation**: Velickovic et al., "Graph Attention Networks," ICLR 2018
- **Type**: Academic paper (no patent)
- **Description**: Introduces learnable attention coefficients for graph neural networks. Attention is computed as alpha_ij = softmax_j(LeakyReLU(a^T [Wh_i || Wh_j])), where attention depends solely on the node features of source and target nodes.
- **Relevance**: ECMP extends the GAT attention mechanism.
- **Key Distinction**: GAT attention is computed exclusively from node features. It has no mechanism for conditioning attention on exogenous variables (price shocks) and no asymmetric decomposition of any input signal. The ECMP formulation alpha_ij = softmax_j(LeakyReLU(a^T [Wh_i || Wh_j || phi(Delta_p_j)])) with asymmetric phi is structurally distinct from GAT.

#### 1.2 GATv2
- **Citation**: Brody et al., "How Attentive are Graph Attention Networks?" ICLR 2022
- **Type**: Academic paper (no patent)
- **Description**: Identifies a limitation in GAT's static attention and proposes a more expressive variant where the attention function computes a^T LeakyReLU(W[h_i || h_j]).
- **Relevance**: Another GAT variant that could be compared to ECMP.
- **Key Distinction**: GATv2, like GAT, computes attention from node features only. It does not condition on exogenous shock variables, does not decompose any input into positive and negative components, and does not use separate transformations for directional signals.

#### 1.3 Graph Convolutional Networks (GCN)
- **Citation**: Kipf and Welling, "Semi-Supervised Classification with Graph Convolutional Networks," ICLR 2017
- **Type**: Academic paper (no patent)
- **Description**: Spectral graph convolutions with fixed (non-learnable) attention weights based on degree normalization.
- **Key Distinction**: No learnable attention, no exogenous conditioning, no asymmetry.

#### 1.4 GraphSAGE
- **Citation**: Hamilton et al., "Inductive Representation Learning on Large Graphs," NeurIPS 2017
- **Type**: Academic paper (no patent)
- **Description**: Sampling and aggregation framework for inductive graph learning.
- **Key Distinction**: Uses uniform or importance-based sampling, not shock-conditioned attention. No asymmetric signal processing.

#### 1.5 Heterogeneous Graph Attention
- **Citation**: Wang et al., "Heterogeneous Graph Attention Network," WWW 2019
- **Type**: Academic paper (no patent)
- **Description**: Extends GAT to heterogeneous graphs with different node and edge types, using type-specific attention.
- **Key Distinction**: Type-conditioned attention is categorically different from shock-conditioned attention. No continuous-valued exogenous conditioning. No asymmetric positive/negative decomposition.

**Summary for Category 1**: No existing graph attention mechanism conditions attention coefficients on exogenous directional shock variables with asymmetric positive/negative decomposition. The ECMP attention computation is structurally novel.

---

### Category 2: Crop Yield Prediction and Agricultural Machine Learning

#### 2.1 Satellite-Based Crop Yield Prediction
- **Citations**: You et al. (2017), Khaki & Wang (2020), Kang et al. (2020), Wang et al. (2020), van Klompenburg et al. (2020 survey)
- **Type**: Academic papers
- **Description**: Use CNN, LSTM, Random Forest, and hybrid architectures on satellite imagery (NDVI, EVI) and weather data to predict crop yield at county or field level. Accuracy typically 85-95% for county-level yield prediction.
- **Key Distinction**: All predict yield (how much a crop will produce), not waste (whether production will be utilized). All treat each observation independently. None use graph-based inter-regional modeling. None incorporate economic contagion mechanisms.

#### 2.2 U.S. Patent No. 10,963,606 (Climate Corporation, 2021)
- **Title**: "Generating digital models of nutrient levels in crop fields"
- **Description**: Machine learning system for field-level agronomic predictions using weather, soil, and management data.
- **Key Distinction**: Field-level yield/nutrient prediction. No waste prediction. No inter-regional economic modeling. No graph structure. No price shock conditioning.

#### 2.3 U.S. Patent No. 11,288,577 (Descartes Labs, 2022)
- **Title**: "Crop type classification from satellite imagery"
- **Description**: CNN-based system for classifying crop types and estimating yields from satellite imagery.
- **Key Distinction**: Crop type classification and yield estimation. No waste prediction. No graph neural networks. No economic features. No contagion modeling.

#### 2.4 U.S. Patent No. 10,671,891 (Granular/Corteva, 2020)
- **Title**: "Agricultural decision support system"
- **Description**: Combines agronomic data with market data for farm management decisions. Includes commodity price inputs.
- **Key Distinction**: Decision support system, not waste prediction. Includes price data as input features but does not model how price shocks propagate between regions through graph-based contagion. No asymmetric shock processing. No adversarial disentanglement.

#### 2.5 U.S. Patent Application 2023/0034635 (Indigo Agriculture, 2023)
- **Title**: "Systems and methods for predicting agricultural outcomes"
- **Description**: Multisource agricultural prediction combining satellite, weather, and agronomic inputs.
- **Key Distinction**: Local predictions without graph structure. No economic contagion modeling. No asymmetric shock embedding.

**Summary for Category 2**: The entire crop prediction patent landscape targets yield or crop type. No identified patent or publication targets crop waste as a prediction objective, uses graph-based economic contagion, or employs asymmetric shock conditioning.

---

### Category 3: Financial Contagion and Cascade Models

#### 3.1 DebtRank
- **Citation**: Battiston et al., "DebtRank: Too Central to Fail?" Scientific Reports, 2012
- **Description**: Network-based algorithm for measuring systemic risk in financial networks. Models how bank defaults cascade through interbank lending networks.
- **Key Distinction**: Operates on financial institution networks with binary default events. Does not use neural network learning. Does not combine biophysical sensor data. Different contagion mechanism (credit default vs. commodity price depression).

#### 3.2 U.S. Patent No. 10,621,583 (2020)
- **Title**: "System for predicting financial contagion risk"
- **Description**: Models financial shock propagation through interconnected institution networks.
- **Key Distinction**: Financial domain only. No agricultural data. No satellite imagery or biophysical features. No adversarial disentanglement. Different graph construction (counterparty exposures vs. commodity market similarity). No asymmetric shock embedding in the attention mechanism.

#### 3.3 Systemic Risk in Financial Networks
- **Citation**: Acemoglu, Ozdaglar, Tahbaz-Salehi, "Systemic Risk and Stability in Financial Networks," AER 2015
- **Description**: Theoretical framework for understanding how network structure determines fragility versus resilience in financial systems.
- **Key Distinction**: Theoretical economics paper without machine learning implementation. Establishes that network structure matters for contagion but does not provide a graph neural network mechanism. The CasCrop economic model draws intellectual inspiration from this literature but the implementation (ECMP) is entirely novel.

#### 3.4 Cascade Prediction with GNNs
- **Citation**: Various works on information cascade prediction (e.g., CasCN, DeepCas)
- **Description**: GNN-based models for predicting information cascade size and timing in social networks.
- **Key Distinction**: Information cascades (resharing, virality) are fundamentally different from economic contagion (price depression, waste). No commodity price conditioning. No asymmetric shock decomposition. No agricultural application.

**Summary for Category 3**: While financial contagion theory motivates the CasCrop approach, no existing system combines financial contagion modeling with agricultural data, graph neural networks, asymmetric shock embeddings, or biophysical-economic disentanglement.

---

### Category 4: Supply Chain Disruption Prediction

#### 4.1 GNN-Based Supply Chain Prediction
- **Citation**: Kosasih and Brintrup, "A Machine Learning Approach for Predicting Hidden Links in Supply Chain with Graph Neural Networks," IJPR 2022
- **Description**: Applies GNNs to predict missing links and disruptions in industrial supply chain networks.
- **Key Distinction**: Industrial supplier-buyer networks, not agricultural production networks. No commodity price shock conditioning. No asymmetric shock embedding. Different graph structure (binary supplier-buyer vs. weighted economic/geographic connectivity). No biophysical features. No adversarial disentanglement.

#### 4.2 Supply Chain Risk Analytics
- **Citation**: Brintrup et al., "Supply Chain Data Analytics for Predicting Supplier Disruptions," IJPR 2020
- **Description**: Machine learning for predicting supplier disruption events from structured supply chain data.
- **Key Distinction**: Tabular ML on supply chain data without graph structure. No graph attention. No shock conditioning. No agricultural application.

#### 4.3 U.S. Patent No. 11,461,693 (2022)
- **Title**: "Supply chain disruption prediction system"
- **Description**: Predictive analytics for supply chain disruption using network analysis.
- **Key Distinction**: Industrial supply chains with different node types (factories, warehouses, retailers). No commodity price shocks. No asymmetric decomposition. No agricultural data.

**Summary for Category 4**: Supply chain GNN applications operate on fundamentally different networks with different contagion mechanisms. None use asymmetric shock conditioning or biophysical-economic disentanglement.

---

### Category 5: Adversarial Disentanglement and Domain Adaptation

#### 5.1 Domain-Adversarial Neural Networks
- **Citation**: Ganin et al., "Domain-Adversarial Training of Neural Networks," JMLR 2016
- **Description**: Introduces the gradient reversal layer for domain adaptation, forcing feature representations to be domain-invariant.
- **Relevance**: CasCrop uses gradient reversal for disentanglement.
- **Key Distinction**: Ganin et al. use gradient reversal for domain adaptation (making representations invariant across source/target domains). CasCrop uses it for a novel purpose: forcing biophysical and economic representations to carry independent information within the same domain. The application to agricultural signal separation and the specific architecture (discriminator predicting z_econ from z_bio) are novel.

#### 5.2 Disentangled Representation Learning
- **Citations**: Mathieu et al. (2016), Creager et al. (2019), Locatello et al. (2019)
- **Description**: Adversarial and variational methods for disentangling latent factors of variation in images, text, and other data.
- **Key Distinction**: Applied to image generation, fairness, and style transfer. Never applied to agriculture. The specific use case --- separating weather-driven agricultural signals from market-driven signals to prove economic contagion --- is entirely novel.

**Summary for Category 5**: While gradient reversal and adversarial disentanglement are known techniques, their application to separating biophysical from economic representations in agricultural prediction, for the specific purpose of distinguishing weather correlation from genuine market contagion, has no precedent.

---

### Category 6: Asymmetric Processing in Neural Networks

#### 6.1 Asymmetric Activation Functions
- **Citations**: Various works on PReLU, Leaky ReLU, ELU
- **Description**: Activation functions that treat positive and negative inputs differently.
- **Key Distinction**: These are activation functions applied element-wise. They do not decompose an exogenous shock variable into positive and negative components for separate learned transformations within a graph attention mechanism. The ECMP asymmetric embedding phi(Delta_p) = W_pos * max(Delta_p, 0) + W_neg * min(Delta_p, 0) is a fundamentally different construct: it decomposes a shock signal, applies separate learned projections, and injects the result into attention computation.

#### 6.2 U.S. Patent No. 11,256,867 (2022)
- **Title**: "Asymmetric neural network for sequence modeling"
- **Description**: Uses asymmetric processing for different types of sequential signals.
- **Key Distinction**: Operates on sequential data, not graph structures. Does not condition graph attention on directional shocks. The specific ECMP formulation is structurally distinct.

#### 6.3 Signed Graph Neural Networks
- **Citations**: Various works on signed graph learning (Derr et al. 2018, Li et al. 2020)
- **Description**: GNNs operating on graphs with positive and negative edges (e.g., trust/distrust in social networks).
- **Key Distinction**: Signed GNNs treat edge signs as discrete labels on graph structure. ECMP conditions on continuous-valued node-level exogenous shocks with learned asymmetric transformation. These are fundamentally different: signed GNNs answer "is this edge positive or negative?" while ECMP asks "how does the magnitude and direction of a continuous shock at the source node modulate information flow?"

**Summary for Category 6**: No existing neural network architecture combines all of: (a) decomposition of a continuous exogenous signal into positive and negative components, (b) separate learned linear projections for each component, (c) injection into graph attention coefficient computation. The ECMP shock embedding is novel.

---

## Novelty Assessment

### Novel Aspect 1: Asymmetric Shock Conditioning in Graph Attention (Claim 3)

**Assessment: NOVEL --- No anticipation found.**

The ECMP attention mechanism is structurally distinct from all identified prior art. Specifically, no existing graph attention mechanism:

1. Conditions attention on an exogenous continuous-valued variable (the price shock) in addition to node features
2. Decomposes this variable into positive and negative components via max/min operations
3. Applies separately parameterized learned linear transformations to the positive and negative components
4. Injects the resulting asymmetric embedding into the attention coefficient computation via concatenation with projected node features

Each individual component (graph attention, ReLU decomposition, linear projections) is known. But their specific combination in the ECMP formulation is novel and non-obvious. A person of ordinary skill in the art, starting from GAT, would not obviously arrive at the specific ECMP construction without the economic motivation of asymmetric shock propagation.

The domain-independent formulation (Claim 3) extends applicability beyond agriculture to any network with directional exogenous shocks: financial contagion, supply chain disruptions, epidemiological spreading, and energy grid cascades. No prior art implements this general mechanism.

**Strength of novelty**: Strong. This is the most defensible claim.

---

### Novel Aspect 2: Biophysical-Economic Disentanglement for Agriculture (Claims 1, 2)

**Assessment: NOVEL --- Novel combination of known technique in new domain with new purpose.**

While adversarial disentanglement via gradient reversal is a known technique (Ganin et al., 2016), its application to agricultural prediction with the specific architecture and purpose described here is novel:

1. **New domain**: No prior art applies adversarial disentanglement to agriculture
2. **New architecture**: The specific discriminator structure (predicting z_econ from z_bio rather than domain labels) is distinct from standard domain adaptation
3. **New purpose**: The disentanglement serves a specific scientific purpose --- proving that the graph captures genuine economic contagion rather than shared weather patterns. This verification-oriented use of disentanglement has no precedent

The combination is non-obvious. A person of ordinary skill would not necessarily recognize that weather correlation could confound economic contagion signals in a graph model, nor would they necessarily apply gradient reversal as the solution.

**Strength of novelty**: Moderate-to-strong. Novel combination may face higher scrutiny than the purely structural novelty of ECMP, but the specific purpose (contagion verification) strengthens the claim.

---

### Novel Aspect 3: Crop Waste Prediction via Economic Contagion (Claims 1, 2)

**Assessment: NOVEL --- Novel problem formulation and system architecture.**

No prior art combines the following elements into a single system:

1. **Waste as prediction target** (distinct from yield): No identified patent or publication targets crop waste as a prediction objective
2. **Insurance claims as training labels**: No system uses USDA RMA cause-of-loss data as labeled training data for machine learning models
3. **Graph-based economic contagion in agriculture**: No system models inter-regional economic dependencies in agriculture via graph neural networks
4. **The specific contagion mechanism**: overproduction in region j depresses prices in region i, increasing waste probability in i --- this mechanism has not been modeled computationally in any prior system

The combination constitutes a novel problem framing (crop waste as network contagion rather than independent local prediction) and a novel system architecture for addressing it.

**Strength of novelty**: Strong for the overall system. The problem formulation itself is a contribution.

---

## Freedom to Operate Analysis

### Potentially Relevant Patents Reviewed

| Patent | Title | Risk | Notes |
|--------|-------|------|-------|
| US 10,963,606 | Nutrient level models (Climate Corp) | Low | Yield prediction, no graphs, no waste |
| US 11,288,577 | Crop classification (Descartes) | Low | Classification, no graphs, no economics |
| US 10,671,891 | Decision support (Granular) | Low | No graph contagion, no asymmetric shocks |
| US 10,621,583 | Financial contagion | Low | Different domain entirely |
| US 11,461,693 | Supply chain disruption | Low | Different domain, no asymmetric shocks |
| US 11,256,867 | Asymmetric sequence model | Low | Sequential, not graph-based |

**Conclusion**: No identified patents would be infringed by the CasCrop system. The GAT mechanism (Velickovic et al.) was published as academic research without patent protection. The gradient reversal layer (Ganin et al.) was published as academic research without patent protection. Underlying technologies (neural networks, graph convolutions, adversarial training, focal loss) are in the public domain or covered by permissive licenses.

**Freedom to operate risk**: Low.

---

## Recommendations

### Filing Strategy

1. **File all three independent claims.** Each has demonstrated novelty:
   - Claim 1 (System) and Claim 2 (Method): Novel system for crop waste prediction via economic contagion
   - Claim 3 (ECMP): Novel domain-independent graph attention mechanism with broadest commercial applicability

2. **Prioritize Claim 3 for prosecution.** The ECMP mechanism is the structurally strongest claim with the broadest applicability. Its domain-independent formulation enables licensing across agriculture, finance, supply chain, epidemiology, and energy sectors.

3. **Strengthen prosecution with experimental evidence.** The ablation study demonstrating that asymmetric ECMP outperforms symmetric ECMP (Row 5 vs. Row 4) provides direct empirical evidence of the non-obviousness of the asymmetric decomposition.

4. **Consider international filing.** Agricultural prediction systems have global applicability. Consider PCT filing within 12 months of provisional.

### Potential Challenges

1. **Obviousness argument for Claim 3**: An examiner might argue that combining GAT with any form of exogenous conditioning is an obvious extension. Counter: (a) no prior art actually does this despite GAT being published in 2018, (b) the specific asymmetric decomposition is motivated by economic theory (Corollary on asymmetric propagation) and is not an arbitrary design choice, (c) empirical evidence shows asymmetric outperforms symmetric.

2. **Alice/101 challenges for Claims 1-2**: Agricultural prediction might be characterized as an "abstract idea." Counter: the claims recite specific technical implementations (neural network architectures, gradient reversal, asymmetric embedding computation) tied to specific technical improvements (improved waste prediction accuracy, economic contagion detection) using specific technical inputs (satellite imagery, commodity futures). The claims satisfy the Alice two-step test.

3. **Anticipation by combination**: An examiner might combine GAT + financial contagion literature + adversarial disentanglement to argue the system is anticipated. Counter: the combination itself is non-obvious, and no prior art suggests applying financial contagion frameworks to agriculture with graph neural networks.
