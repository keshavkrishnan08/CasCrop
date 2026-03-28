# United States Provisional Patent Application

## Title of Invention

SYSTEM AND METHOD FOR PREDICTING AGRICULTURAL CROP WASTE USING ECONOMIC CONTAGION GRAPH NEURAL NETWORKS WITH ASYMMETRIC SHOCK CONDITIONING

---

## Cross-Reference to Related Applications

Not applicable.

---

## Field of the Invention

This invention relates generally to agricultural prediction systems and, more particularly, to computer-implemented systems and methods for predicting crop waste events using graph neural networks that model economic contagion between agricultural production regions, incorporating asymmetric conditioning on commodity price shock direction and adversarial disentanglement of biophysical and economic feature representations.

---

## Background of the Invention

### The Problem of Crop Waste

Every year, billions of dollars of agricultural crops in the United States are wasted---abandoned in the field, plowed under, or left unharvested. The United States Department of Agriculture's Risk Management Agency paid out over $19 billion in crop insurance indemnities in 2023 alone. A substantial portion of these losses resulted not from biological failures such as drought, frost, or disease, but from economic forces: commodity prices fell below the cost of harvesting, overproduction in neighboring regions flooded local markets, or logistics bottlenecks made transportation uneconomical.

These economic waste events propagate across regions like a financial contagion. When one region produces a bumper harvest, the resulting supply glut depresses commodity prices in connected markets, pushing marginal farmers in neighboring regions below their break-even threshold. Their resulting crop abandonment further distorts supply signals, and the cascade continues.

### Limitations of Prior Art

Existing crop prediction systems treat each geographic unit (field, county, or region) as an independent observation. These systems suffer from several fundamental limitations:

**1. Yield prediction rather than waste prediction.** All existing satellite-based crop prediction systems predict how much a field will produce (yield). They do not predict whether that production will be utilized (waste). This distinction is critical: a high-yield crop can be entirely wasted if commodity prices crash below harvesting costs, while a low-yield crop can be fully utilized if prices are sufficiently high. Waste is determined by the intersection of biology and economics, not biology alone.

Representative prior art includes U.S. Patent No. 10,963,606 (Climate Corporation), which uses machine learning on weather and soil data for field-level yield prediction; U.S. Patent No. 11,288,577 (Descartes Labs), which applies convolutional neural networks to satellite imagery for crop type classification and yield estimation; and U.S. Patent No. 10,671,891 (Granular), which combines agronomic data with market data for farm decision support. None of these systems predict waste as a distinct outcome from yield.

**2. Independent observation assumption.** No existing system models the economic interdependencies between agricultural production regions. Each county or field is treated as if events in neighboring regions have no bearing on its outcomes. This assumption is provably false: a bumper harvest in Iowa depresses corn prices across the entire Midwest, affecting harvesting decisions in Illinois, Indiana, and Ohio regardless of their local biophysical conditions.

**3. No asymmetric shock modeling.** In existing graph neural network literature, including Graph Attention Networks (Velickovic et al., 2018), Graph Convolutional Networks (Kipf and Welling, 2017), and GraphSAGE (Hamilton et al., 2017), attention mechanisms treat all neighbor signals symmetrically. However, in commodity markets, positive shocks (price increases from undersupply) and negative shocks (price decreases from oversupply) propagate with fundamentally different dynamics. A price decrease triggers cascading waste as harvesting becomes uneconomical in connected regions. A price increase cannot generate a symmetric cascade of "anti-waste." No existing graph neural network architecture captures this directional asymmetry.

**4. Confounded biophysical and economic signals.** Because weather patterns are spatially correlated and affect both crop health and commodity prices (through supply effects), any model combining these data sources risks confusing shared weather effects with genuine economic contagion. No existing system enforces independence between biophysical and economic feature representations, making it impossible to verify that observed inter-regional effects capture market dynamics rather than correlated climate patterns.

**5. No graph-based economic contagion in agriculture.** While graph neural networks have been applied to supply chain disruption prediction in industrial contexts (predicting factory closures propagating through supplier-buyer networks), no system applies graph-based contagion modeling to agricultural waste prediction with the specific mechanism of overproduction causing price depression causing waste in connected regions.

### Need in the Art

There exists an unmet need for a prediction system that: (a) targets crop waste rather than yield as the prediction objective; (b) models economic contagion between agricultural regions via graph-structured machine learning; (c) captures the asymmetric propagation dynamics of positive versus negative commodity price shocks; and (d) enforces independence between biophysical and economic feature representations to distinguish weather correlation from genuine market contagion.

---

## Summary of the Invention

The present invention provides a computer-implemented system and method for predicting agricultural crop waste events by modeling economic contagion across a network of agricultural production regions. The system overcomes all limitations of the prior art through three novel components:

**First**, a biophysical-economic disentanglement architecture that encodes satellite imagery, weather, and soil data into a biophysical latent representation, and commodity prices, costs, and market signals into an economic latent representation, with an adversarial training objective (using gradient reversal) that forces these representations to capture genuinely independent information. This enables the system to prove that inter-regional effects represent true economic contagion rather than shared weather patterns.

**Second**, an Economic Contagion Message Passing (ECMP) mechanism---a novel graph attention method where attention coefficients between regions are conditioned on an asymmetric embedding of commodity price shocks. Positive price shocks (increases) and negative price shocks (decreases) are processed through separately parameterized learned linear transformations, enabling the system to learn that oversupply-driven price decreases cause waste contagion while undersupply-driven price increases reduce waste risk. This mechanism is domain-independent and applicable beyond agriculture to any network where directional exogenous shocks propagate between nodes.

**Third**, a dynamic graph construction method that represents agricultural regions as nodes and builds edges from a learned combination of geographic proximity, commodity market similarity, and transportation infrastructure connectivity. The combination weights are learnable parameters optimized during training, allowing the system to discover the optimal blend of connectivity sources.

The combination of these components enables the system to predict crop waste events weeks in advance by detecting economic cascade patterns propagating through commodity markets, achieving significant performance improvements over systems treating regions independently.

---

## Brief Description of the Figures

- **FIG. 1** is a block diagram of the overall CasCrop system architecture, showing data flow from multimodal input sources through encoding, adversarial disentanglement, graph construction, ECMP message passing, and multi-task prediction.

- **FIG. 2** is a detailed diagram of the Economic Contagion Message Passing (ECMP) layer, illustrating the asymmetric shock embedding function, attention coefficient computation with shock conditioning, multi-head attention, and weighted neighbor aggregation.

- **FIG. 3** is a diagram of the adversarial disentanglement module, showing the biophysical encoder, gradient reversal layer, discriminator network, and gradient flow during training.

- **FIG. 4** is a diagram of the dynamic graph construction module, showing the three pre-computed adjacency matrices (geographic, commodity, transportation), learnable softmax combination weights, and top-K sparsification.

- **FIG. 5** is a geographic visualization showing a portion of the county-level economic contagion graph overlaid on a map of the U.S. Midwest, with nodes representing counties and edges representing learned economic connections.

- **FIG. 6** is a flowchart of the complete method for predicting crop waste events, from data ingestion through feature extraction, encoding, disentanglement, graph construction, ECMP message passing, prediction, economic impact estimation, and alert generation.

- **FIG. 7** is a diagram showing the ECMP stack architecture with two stacked ECMP layers, residual connections, layer normalization, and ELU activations.

---

## Detailed Description of the Invention

### 1. System Overview (FIG. 1)

Referring to FIG. 1, the CasCrop system 100 comprises a data ingestion module 110, a feature extraction module 120, a biophysical encoder 130, an economic encoder 140, an adversarial disentanglement module 150, a dynamic graph constructor 160, an Economic Contagion Message Passing (ECMP) module 170, a waste prediction head 181, a cause classification head 182, a training controller 190, and an alert generation module 195.

The data ingestion module 110 receives data from multiple heterogeneous sources including but not limited to: satellite imagery providers (Sentinel-2 at 10m resolution, Landsat at 30m resolution), weather observation networks (NOAA Climate Data Online), soil moisture sensors (NASA SMAP at 9km resolution, NLDAS-2 reanalysis), commodity price feeds (CME Group futures, USDA NASS prices received, FRED economic data), crop insurance databases (USDA Risk Management Agency cause-of-loss records), and geographic/infrastructure databases (U.S. Census Bureau county adjacency, OpenStreetMap transportation networks, USDA GIPSA grain elevator locations).

### 2. Feature Extraction Module (120)

The feature extraction module 120 processes raw data from the data ingestion module 110 into structured feature vectors at county-commodity-month granularity. The module produces three categories of features:

**Biophysical features** (approximately 30 dimensions): satellite-derived vegetation indices including Normalized Difference Vegetation Index (NDVI) mean and standard deviation, Enhanced Vegetation Index (EVI), Soil-Adjusted Vegetation Index (SAVI), and Normalized Difference Water Index (NDWI); weather aggregates including maximum and minimum temperature, total precipitation, cumulative growing degree days, frost days, consecutive dry days, and Palmer Drought Severity Index (PDSI); and soil moisture measurements including surface and root-zone soil moisture. Additional biophysical features include USDA VegScape vegetation condition, crop area fraction from the USDA Cropland Data Layer, and historical yield statistics (mean and standard deviation).

**Economic features** (approximately 15 dimensions): commodity price level, 1-month and 3-month price percentage changes, 30-day rolling price volatility, cost of production from USDA Economic Research Service, revenue-to-cost ratio, futures basis (difference between spot and futures price), national supply estimate, and export demand index.

**Historical risk features** (approximately 10 dimensions): county historical loss frequency and severity from RMA records, average indemnity amount, crop diversity index (Herfindahl index of planted acreage by crop), irrigation fraction, and temporal embeddings (sinusoidal month encoding and linear year trend).

All features are normalized using z-score standardization with statistics computed exclusively from the training data period to prevent information leakage. Features for predicting waste at time t use only data available before time t, with a minimum one-week lag.

### 3. Biophysical Encoder (130)

The biophysical encoder 130 receives a biophysical feature vector x_bio of dimension d_bio (approximately 30) and produces a latent representation z_bio of dimension d (e.g., d = 64). In the preferred embodiment, the encoder is implemented as a three-layer multilayer perceptron (MLP):

- Layer 1: Linear transformation (d_bio to 128), batch normalization, ReLU activation, dropout (p = 0.3)
- Layer 2: Linear transformation (128 to 128), batch normalization, ReLU activation, dropout (p = 0.3)
- Layer 3: Linear transformation (128 to d), no activation (raw latent embedding)

Weight initialization uses Kaiming normal initialization with ReLU nonlinearity. The final layer produces raw embeddings without activation to permit downstream modules (particularly the disentanglement module) to impose their own constraints on the latent space geometry.

In alternative embodiments, the biophysical encoder may comprise:
- A convolutional neural network (CNN) or residual network (ResNet-18) operating on raw satellite imagery patches rather than pre-computed vegetation indices
- A long short-term memory (LSTM) or gated recurrent unit (GRU) network operating on temporal sequences of biophysical features
- A vision transformer (ViT) operating on satellite imagery tiles
- Any combination thereof with a fusion layer

### 4. Economic Encoder (140)

The economic encoder 140 receives an economic feature vector x_econ of dimension d_econ (approximately 15) and produces a latent representation z_econ of dimension d. The architecture mirrors the biophysical encoder but with a smaller hidden dimension (64 versus 128), reflecting the lower dimensionality of the economic feature space:

- Layer 1: Linear transformation (d_econ to 64), batch normalization, ReLU activation, dropout (p = 0.3)
- Layer 2: Linear transformation (64 to 64), batch normalization, ReLU activation, dropout (p = 0.3)
- Layer 3: Linear transformation (64 to d), no activation

### 5. Adversarial Disentanglement Module (150, FIG. 3)

Referring to FIG. 3, the adversarial disentanglement module 150 enforces statistical independence between z_bio and z_econ. This is critical for distinguishing genuine economic contagion from confounded weather effects, because weather patterns are spatially correlated and affect both vegetation health (biophysical) and commodity prices (economic through supply effects).

The module comprises:

**Gradient Reversal Layer (GRL) 151**: During forward computation, the GRL passes its input unchanged: GRL(x) = x. During backpropagation, the GRL multiplies incoming gradients by negative lambda: d/dx GRL(x) = -lambda * I. The hyperparameter lambda controls the strength of the reversal signal and may be set to a fixed value (e.g., lambda = 1.0) or scheduled during training.

**Discriminator Network 152**: A three-layer MLP (d to d to d to d) with ReLU activations that attempts to reconstruct z_econ from z_bio. The discriminator receives z_bio after passage through the GRL and produces a predicted z_econ_hat.

**Disentanglement Loss**: The loss is computed as the mean squared error between the discriminator's prediction and the actual economic embedding:

```
L_dis = ||D(GRL(z_bio)) - z_econ||_2^2
```

The key insight is that minimizing this loss with respect to the discriminator's parameters trains D to predict z_econ from z_bio (making D better at detecting shared information), while the gradient reversal simultaneously pushes the biophysical encoder's parameters in the direction that makes z_bio less predictive of z_econ (removing shared information from z_bio).

**Training Schedule**: In the preferred embodiment, the disentanglement module uses a warm-up schedule:
- Epochs 1-10: lambda = 0 (disentanglement inactive, encoders learn freely)
- Epochs 11+: lambda = 1.0 (disentanglement active)
- The discriminator is trained for 5 gradient steps per encoder step, following standard adversarial training practices

**Verification**: At convergence, a linear probe (simple linear classifier) trained on z_bio to predict z_econ should achieve near-chance accuracy (approximately 50% on binary classification tasks or cosine similarity near 0 on regression tasks), confirming that the two representations carry independent information.

### 6. Dynamic Graph Constructor (160, FIG. 4)

Referring to FIG. 4, the dynamic graph constructor 160 builds a graph G = (V, E) where vertices V represent agricultural production regions (e.g., U.S. counties, identified by 5-digit FIPS codes) and edges E represent economic and geographic relationships.

Three pre-computed adjacency matrices serve as inputs:

**Geographic Adjacency (G_geo)**: Edge weights encode spatial proximity. In the preferred embodiment, weights are computed as exponential decay of great-circle distance between county centroids: g_ij = exp(-d_ij / sigma), where sigma is a bandwidth parameter set to the median inter-county distance or learned from data.

**Commodity Market Similarity (G_commodity)**: Edge weights encode market connectivity between counties growing the same commodity. Weights are computed from: (a) cosine similarity of production profiles (crop area fraction and yield history vectors), (b) rolling 12-month price correlation between received prices, and (c) shared market access (counties selling to the same processors or elevators). This matrix is time-varying---it is recomputed monthly as production profiles and price correlations change.

**Transportation Connectivity (G_transport)**: Edge weights encode logistics connectivity from highway networks, railroad access, and grain elevator proximity. Weights are computed as a function of road distance, rail access indicators, and the number of shared grain elevators between counties.

**Learned Combination**: The final edge weight is:

```
w_ij(t) = alpha * g_ij + beta * c_ij(t) + gamma * tau_ij
```

where alpha, beta, and gamma are learnable scalar parameters normalized via softmax:

```
[alpha, beta, gamma] = softmax([logit_alpha, logit_beta, logit_gamma])
```

This ensures the weights are non-negative and sum to one, and are optimized jointly with the rest of the network during training. The softmax normalization allows the model to discover the optimal blend of geographic, market, and logistics connectivity without manual tuning.

**Sparsification**: The combined adjacency matrix is sparsified by retaining only the top-K (e.g., K = 20) neighbors per node, producing a sparse graph suitable for efficient message passing. Self-loops are removed prior to sparsification.

**Output**: The module outputs a sparse edge index (COO format) and edge weights compatible with standard graph neural network frameworks.

### 7. Economic Contagion Message Passing (ECMP) Module (170, FIG. 2)

Referring to FIG. 2, the ECMP module 170 is the core novel component of the invention. It extends the graph attention mechanism with asymmetric shock conditioning, enabling the network to learn how commodity price shocks propagate directionally across the agricultural graph.

#### 7.1 Node Feature Construction

For each node i in the graph (representing one county-commodity pair at one time step), the input feature vector is the concatenation:

```
h_i = [z_bio_i || z_econ_i || x_hist_i]
```

where z_bio_i is the biophysical latent representation (dimension d), z_econ_i is the economic latent representation (dimension d), and x_hist_i is the historical risk feature vector (dimension d_hist). The total input dimension is d_in = 2*d + d_hist (e.g., 2*64 + 10 = 138).

#### 7.2 Asymmetric Shock Embedding Function

For each node j, the system computes a shock embedding from the scalar commodity price shock Delta_p_j (the one-month percentage change in the relevant commodity futures price):

```
phi(Delta_p_j) = W_pos * max(Delta_p_j, 0) + W_neg * min(Delta_p_j, 0)
```

where:
- W_pos is a learned linear transformation matrix of dimension s x 1 (s = shock embedding dimension, e.g., s = 8)
- W_neg is a separately parameterized learned linear transformation matrix of dimension s x 1
- max(Delta_p_j, 0) extracts the positive component (price increase) via ReLU
- min(Delta_p_j, 0) extracts the negative component (price decrease) via negative ReLU, preserving the sign

**This asymmetric decomposition is the key innovation of the ECMP mechanism.** Because W_pos and W_neg are independently parameterized, the network can learn fundamentally different representations for positive versus negative shocks. In agricultural markets:

- A negative shock (price decrease from oversupply) triggers waste contagion: the price drop makes harvesting uneconomical for connected farmers, and oversupply drives prices further down
- A positive shock (price increase from undersupply) reduces waste risk: the price increase makes harvesting economical even for marginal fields, and undersupply incentivizes increased utilization

The asymmetry is well-grounded in economic theory. The waste condition (revenue < cost) is a one-sided constraint: prices below the threshold cause waste, but prices above it do not cause symmetric "anti-waste." This creates inherently asymmetric propagation dynamics.

**Symmetric Variant**: In an alternative embodiment used as a baseline, the symmetric shock embedding uses a single transformation: phi(Delta_p_j) = W_sym * Delta_p_j. This variant serves to quantify the value of asymmetric conditioning.

#### 7.3 Attention Coefficient Computation

For each directed edge from source node j to target node i, the raw attention score is computed as:

```
e_ij = LeakyReLU(a^T * [W*h_i || W*h_j || phi(Delta_p_j)])
```

where:
- W is a shared linear projection (dimension d' * H x d_in), reshaped to produce per-head projections of dimension d' for each of H attention heads
- a is a learnable attention vector of dimension 2*d' + s per head
- || denotes concatenation
- LeakyReLU uses a negative slope of 0.2

The attention scores are normalized across all incoming edges to each target node using numerically stable softmax:

```
alpha_ij = exp(e_ij - max_k(e_ik)) / sum_{k in N(i)} exp(e_ik - max_k(e_ik))
```

Optional: if edge features are available (e.g., edge weight from the graph constructor), an edge bias is added to the raw attention score via a learned edge projection: e_ij' = e_ij + W_edge * edge_attr_ij.

Attention dropout (p = 0.3) is applied to the normalized coefficients to regularize training.

#### 7.4 Multi-Head Attention

The system uses H = 4 attention heads, each with independent parameter sets {W^h, a^h, W_pos^h, W_neg^h} for h = 1, ..., H. Each head independently computes attention coefficients and aggregates neighbor features, allowing different heads to specialize in different contagion patterns (e.g., one head for short-range geographic contagion, another for long-range market contagion, another for transportation-mediated effects).

In the first ECMP layer, head outputs are concatenated (output dimension = d' * H). In the second (final) ECMP layer, head outputs are averaged (output dimension = d').

#### 7.5 Weighted Neighbor Aggregation

For each target node i and each head h, the updated representation is:

```
h'_i^h = sum_{j in N(i)} alpha_ij^h * W^h * h_j
```

where the sum ranges over all nodes j in the neighborhood N(i) of node i (as defined by the graph edges).

#### 7.6 ECMP Stack (FIG. 7)

Referring to FIG. 7, the full ECMP module comprises a stack of two ECMP layers with residual connections and layer normalization:

**Layer 1 (concat heads):**
```
h^(1) = ELU(LayerNorm(ECMP_1(h^(0), E, Delta_p) + W_res^(1) * h^(0)))
```

**Layer 2 (average heads):**
```
h^(2) = ELU(LayerNorm(ECMP_2(h^(1), E, Delta_p) + W_res^(2) * h^(1)))
```

where W_res^(l) are linear projection matrices that match dimensions for the residual connection when input and output dimensions differ, and ELU is the Exponential Linear Unit activation function.

The two-layer stack enables two-hop message passing, allowing indirect contagion pathways (e.g., county A's shock affects county B's price, which then affects county C's waste probability) to influence predictions.

### 8. Prediction Module (180)

The prediction module 180 operates on the graph-enriched representations h_graph produced by the ECMP module 170 and comprises two parallel prediction heads:

**Waste Prediction Head 181**: A two-layer MLP (d to 32, ReLU, dropout p = 0.3, 32 to 1) producing a scalar logit for binary waste prediction. The sigmoid function converts the logit to a probability in [0, 1]. The output represents the estimated probability that the given county-commodity-month observation will experience a crop waste event within the prediction horizon.

**Cause Classification Head 182**: A two-layer MLP (d to 32, ReLU, dropout p = 0.3, 32 to K) producing K logits for multi-class cause-of-loss prediction. In the preferred embodiment, K = 6 classes: DROUGHT (RMA cause codes 2, 3), EXCESS_MOISTURE (codes 10, 11, 14), COLD (codes 15, 16, 17), HEAT (codes 36, 40), PRICE_DECLINE (codes 47, 48), and OTHER (all remaining codes). Softmax converts logits to class probabilities.

### 9. Training Controller (190)

The training controller 190 orchestrates end-to-end training by minimizing the combined loss function:

```
L = L_waste + mu * L_cause + lambda * L_dis
```

where:
- L_waste is the waste prediction loss, implemented as focal loss with focusing parameter gamma = 2.0 and class weight alpha = 0.75 (emphasizing the minority waste class). Focal loss FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t) down-weights easy examples and focuses training on hard-to-classify cases, critical when waste events constitute only 5-15% of observations.
- L_cause is multi-class cross-entropy loss for cause classification
- L_dis is the disentanglement loss from the adversarial module
- mu = 0.3 is the cause loss coefficient
- lambda = 0.1 is the disentanglement loss coefficient

The training controller manages:
- AdamW optimizer with learning rate 10^-3 and weight decay 10^-4
- Cosine annealing learning rate schedule with warm restarts (T_0 = 50 epochs)
- Gradient norm clipping at 1.0
- Disentanglement warm-up (lambda = 0 for first 10 epochs)
- Discriminator training schedule (5 discriminator steps per encoder step)
- Early stopping on validation AUC-ROC with patience of 20 epochs
- Random seed control for reproducibility

### 10. Alert Generation Module (195)

In a preferred deployment embodiment, the alert generation module 195 monitors waste probability predictions from the prediction module 180. When the waste probability for a given county-commodity pair exceeds a configurable threshold (e.g., 50%) at a specified prediction lead time (e.g., 4 weeks ahead), the module generates an alert comprising:

1. The waste probability and its temporal trend
2. The predicted cause category and confidence
3. An attention map identifying the top-K connected counties contributing most to the elevated waste risk, derived from the ECMP attention coefficients alpha_ij
4. An economic impact estimate based on historical indemnity amounts for the county-commodity pair
5. Recommended intervention actions (forward contracting, market redirection, storage strategy adjustment, food bank partnership activation)

Alerts are delivered to configured stakeholders via at least one of: a display device (dashboard), a storage device (database), a communication interface (email, SMS, API endpoint), or an integration with existing USDA early warning systems.

---

## Claims

### Independent Claim 1: System

**1.** A computer-implemented system for predicting agricultural crop waste events, the system comprising:

(a) a processor;

(b) a memory coupled to the processor, the memory storing instructions that, when executed by the processor, cause the processor to:

(i) receive, for each of a plurality of agricultural regions, biophysical feature data comprising at least one of satellite-derived vegetation indices, weather observations, and soil moisture measurements;

(ii) receive, for each of the plurality of agricultural regions, economic feature data comprising at least one of commodity prices, production costs, and commodity price changes over one or more time horizons;

(iii) encode the biophysical feature data into a biophysical latent representation using a biophysical encoder neural network;

(iv) encode the economic feature data into an economic latent representation using an economic encoder neural network;

(v) enforce statistical independence between the biophysical latent representation and the economic latent representation using an adversarial disentanglement module comprising a gradient reversal layer and a discriminator network, wherein the discriminator network is trained to reconstruct the economic latent representation from the biophysical latent representation, and the gradient reversal layer reverses gradient direction during backpropagation to penalize the biophysical encoder when reconstruction succeeds;

(vi) construct a graph connecting the plurality of agricultural regions based on at least two of: geographic proximity, commodity market similarity, and transportation connectivity, wherein at least one edge weight is determined by a learnable combination parameter optimized during training;

(vii) perform economic contagion message passing on the graph, wherein attention coefficients between pairs of agricultural regions are conditioned on an asymmetric shock embedding of a commodity price shock at a source region, the asymmetric shock embedding computing a first embedding by applying a first learned linear transformation to the positive component of the price shock and a second embedding by applying a second, independently parameterized learned linear transformation to the negative component of the price shock, and summing the first and second embeddings;

(viii) generate, for each agricultural region, a waste probability indicating the likelihood that a crop waste event will occur in the region within a specified prediction horizon; and

(ix) output the waste probability to at least one of a display device, a storage device, and a communication interface.

### Dependent Claims on Claim 1

**4.** The system of claim 1, wherein the biophysical encoder neural network comprises a multilayer perceptron with batch normalization and rectified linear unit activations, and the economic encoder neural network comprises a multilayer perceptron with a hidden dimension smaller than that of the biophysical encoder, reflecting the lower dimensionality of the economic feature space.

**5.** The system of claim 1, wherein the adversarial disentanglement module further comprises a warm-up schedule that sets the gradient reversal strength to zero for a predetermined number of initial training epochs before activating adversarial training at a target reversal strength.

**6.** The system of claim 1, wherein the asymmetric shock embedding uses a shock embedding dimension of at least 4, and wherein the economic contagion message passing uses multi-head attention with at least 2 independently parameterized attention heads, each head learning distinct contagion patterns.

**7.** The system of claim 1, wherein the graph construction uses learnable combination parameters for geographic proximity, commodity market similarity, and transportation connectivity, said parameters normalized via softmax to ensure the parameters are non-negative and sum to one, and wherein the graph is sparsified by retaining only the top-K neighbors per node, where K is a configurable parameter.

**8.** The system of claim 1, further comprising a cause classification head that categorizes predicted waste events into at least three cause categories including at least one biophysical cause category selected from drought, excess moisture, cold, and heat, and at least one economic cause category comprising price decline.

**9.** The system of claim 1, wherein the waste prediction is trained using focal loss with a focusing parameter gamma greater than 1.0 and a positive class weight alpha greater than 0.5, to address class imbalance between waste and non-waste observations, and wherein the total training loss further comprises a cause classification loss weighted by a first coefficient and a disentanglement loss weighted by a second coefficient.

**10.** The system of claim 1, wherein the instructions further cause the processor to generate an alert when the waste probability for a given agricultural region exceeds a configurable threshold, the alert comprising the waste probability, the predicted cause category, an identification of connected regions contributing most to the waste risk based on the attention coefficients, and an estimated economic impact.

**11.** The system of claim 1, wherein the economic contagion message passing comprises two or more stacked message passing layers with residual connections and layer normalization between layers, enabling multi-hop contagion propagation through indirect pathways.

**12.** The system of claim 1, wherein the commodity market similarity component of the graph is time-varying and recomputed at each prediction time step based on current production profiles and commodity price correlations between regions.

### Independent Claim 2: Method

**2.** A computer-implemented method for predicting crop waste events across a plurality of agricultural regions, the method comprising:

(a) ingesting, for each agricultural region, multimodal input data comprising biophysical features derived from remote sensing and weather observations, and economic features derived from commodity market data;

(b) encoding the biophysical features into a first latent representation using a first neural network encoder;

(c) encoding the economic features into a second latent representation using a second neural network encoder;

(d) applying an adversarial training objective to enforce independence between the first and second latent representations, comprising passing the first latent representation through a gradient reversal layer and training a discriminator to reconstruct the second latent representation from the gradient-reversed first latent representation;

(e) constructing a dynamic graph with nodes representing agricultural regions and edges representing economic and geographic relationships, wherein edge weights are determined by a learned combination of at least two of geographic proximity, commodity market similarity, and transportation connectivity;

(f) performing graph neural network message passing on the dynamic graph using attention coefficients conditioned on asymmetric commodity price shock embeddings, wherein for each source node in the graph:

- computing a scalar price shock value representing a percentage change in commodity price over a time horizon;
- decomposing the price shock value into a positive component via max(shock, 0) and a negative component via min(shock, 0);
- transforming the positive component using a first learned linear projection to produce a positive embedding;
- transforming the negative component using a second learned linear projection, independently parameterized from the first, to produce a negative embedding;
- summing the positive and negative embeddings to produce an asymmetric shock embedding;
- concatenating the shock embedding with projected node features of the source and target nodes;
- computing attention coefficients from the concatenated vector using a learnable attention vector and nonlinear activation;
- normalizing attention coefficients across all incoming edges to each target node;

(g) generating a waste prediction for each agricultural region based on the graph-enriched node representations;

(h) generating a cause-of-loss classification for each agricultural region indicating the predicted mechanism of a potential waste event; and

(i) providing the waste prediction and cause-of-loss classification to at least one of a user interface, a database, and an alert system.

### Dependent Claims on Claim 2

**13.** The method of claim 2, wherein the biophysical features comprise at least three of: normalized difference vegetation index (NDVI), enhanced vegetation index (EVI), growing degree days, frost days, Palmer Drought Severity Index, surface soil moisture, root-zone soil moisture, and USDA VegScape vegetation condition index.

**14.** The method of claim 2, wherein the economic features comprise at least three of: commodity price level, one-month price change, three-month price change, 30-day price volatility, cost of production, revenue-to-cost ratio, futures basis, and national supply estimate.

**15.** The method of claim 2, further comprising applying temporal constraints to prevent data leakage, wherein features for predicting waste at time t use only data available before time t with at least a configurable minimum lag period.

**16.** The method of claim 2, wherein the dynamic graph is time-varying, with commodity market similarity edges recomputed at each time step based on current production profiles, price correlations, and market access patterns.

**17.** The method of claim 2, further comprising training the method end-to-end by minimizing a combined loss function comprising a waste prediction focal loss, a cause classification cross-entropy loss weighted by a first coefficient, and a disentanglement mean squared error loss weighted by a second coefficient.

**18.** The method of claim 2, wherein generating the waste prediction comprises generating predictions at multiple prediction horizons including at least a 2-week horizon and a 4-week horizon, enabling lead-time-dependent decision making.

**19.** The method of claim 2, further comprising computing an economic impact estimate by multiplying the waste probability for each agricultural region by a historical indemnity amount for that region and aggregating across regions to estimate total preventable waste.

### Independent Claim 3: ECMP (Domain-Independent)

**3.** A computer-implemented method for graph neural network message passing with asymmetric shock conditioning, the method comprising:

(a) receiving a graph comprising a plurality of nodes and a plurality of edges connecting pairs of nodes, each node associated with a feature vector of dimension d_in and each node associated with a scalar shock value representing a directional exogenous perturbation at the node;

(b) for each node, computing an asymmetric shock embedding by:

- decomposing the scalar shock value into a positive component equal to max(shock_value, 0) and a negative component equal to min(shock_value, 0);
- applying a first learned linear transformation of dimension s x 1 to the positive component to produce a positive embedding vector in R^s;
- applying a second learned linear transformation of dimension s x 1, with parameters independent of the first transformation, to the negative component to produce a negative embedding vector in R^s;
- summing the positive and negative embedding vectors to produce the asymmetric shock embedding of dimension s;

(c) for each directed edge from a source node j to a target node i, computing an attention coefficient by:

- projecting the feature vectors of the source node and the target node using a shared linear projection of dimension d' x d_in;
- concatenating the projected target features, the projected source features, and the asymmetric shock embedding of the source node to form a concatenated vector of dimension 2*d' + s;
- computing a scalar attention score by computing the inner product of a learnable attention vector of dimension 2*d' + s with the concatenated vector, followed by a leaky rectified linear unit activation;
- normalizing the attention score across all edges incoming to the target node using a softmax function to produce a normalized attention coefficient;

(d) for each target node, computing an updated feature vector by weighted aggregation of transformed source node features, where each source node's contribution is weighted by the corresponding normalized attention coefficient; and

(e) outputting the updated feature vectors for all nodes in the graph.

### Dependent Claims on Claim 3

**20.** The method of claim 3, wherein the graph neural network message passing is performed with multi-head attention comprising H independently parameterized attention heads, each head having independent learned parameters for the first and second linear transformations, the shared linear projection, and the learnable attention vector, and wherein head outputs are concatenated or averaged to produce the final updated feature vector.

**21.** The method of claim 3, further comprising stacking two or more layers of graph neural network message passing with residual connections and layer normalization between layers, wherein a linear projection matches dimensions for the residual connection when input and output dimensions differ.

**22.** The method of claim 3, wherein the scalar shock value represents a financial market perturbation selected from asset price change, interest rate change, credit spread change, and volatility change, and the graph represents a financial network with nodes representing financial entities and edges representing financial exposures, counterparty relationships, or correlated portfolio holdings.

**23.** The method of claim 3, wherein the scalar shock value represents a supply chain disruption magnitude selected from production capacity change, delivery delay, and demand fluctuation, and the graph represents a supply chain network with nodes representing supply chain entities and edges representing supplier-buyer relationships.

**24.** The method of claim 3, wherein the scalar shock value represents an epidemiological metric change selected from infection rate change, hospitalization rate change, and mortality rate change, and the graph represents a population mobility network with nodes representing geographic regions and edges representing population movement flows.

**25.** The method of claim 3, wherein the scalar shock value represents an energy grid perturbation selected from generation capacity change, demand change, and transmission capacity change, and the graph represents an energy transmission network with nodes representing grid zones and edges representing transmission lines.

**26.** The method of claim 3, further comprising incorporating edge features into the attention computation by projecting edge feature vectors through a learned linear layer and adding the projected edge features as a bias to the raw attention scores prior to softmax normalization.

---

## Abstract of the Disclosure

A computer-implemented system and method for predicting agricultural crop waste events using a graph neural network that models economic contagion between agricultural production regions. The system encodes satellite imagery, weather observations, and commodity market data into disentangled biophysical and economic latent representations using adversarial training with gradient reversal. A novel Economic Contagion Message Passing (ECMP) mechanism propagates information across a dynamic county graph using attention coefficients conditioned on asymmetric commodity price shock embeddings, where positive price shocks (from undersupply) and negative price shocks (from oversupply) are processed through separately parameterized learned linear transformations. This asymmetric conditioning captures the fundamental directionality of agricultural market contagion: price decreases from oversupply cause waste contagion across connected regions while price increases from undersupply reduce waste risk. A dynamic graph constructor combines geographic proximity, commodity market similarity, and transportation connectivity with learnable combination weights. The system generates waste probability predictions and cause-of-loss classifications, enabling early warning and targeted intervention. The ECMP mechanism is domain-independent and applicable to any network where asymmetric directional shocks propagate between nodes, including financial contagion, supply chain disruption, epidemiological spreading, and energy grid cascades.
