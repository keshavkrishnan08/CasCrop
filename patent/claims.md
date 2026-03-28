# CasCrop Patent Claims

---

## Independent Claims

### Claim 1 --- System

**1.** A computer-implemented system for predicting agricultural crop waste events, the system comprising:

(a) a processor;

(b) a memory coupled to the processor, the memory storing instructions that, when executed by the processor, cause the processor to:

> (i) receive, for each of a plurality of agricultural regions, biophysical feature data comprising at least one of satellite-derived vegetation indices, weather observations, and soil moisture measurements;
>
> (ii) receive, for each of the plurality of agricultural regions, economic feature data comprising at least one of commodity prices, production costs, and commodity price changes over one or more time horizons;
>
> (iii) encode the biophysical feature data into a biophysical latent representation using a biophysical encoder neural network;
>
> (iv) encode the economic feature data into an economic latent representation using an economic encoder neural network;
>
> (v) enforce statistical independence between the biophysical latent representation and the economic latent representation using an adversarial disentanglement module comprising a gradient reversal layer and a discriminator network, wherein the discriminator network is trained to reconstruct the economic latent representation from the biophysical latent representation, and the gradient reversal layer reverses gradient direction during backpropagation to penalize the biophysical encoder when reconstruction succeeds;
>
> (vi) construct a graph connecting the plurality of agricultural regions based on at least two of: geographic proximity, commodity market similarity, and transportation connectivity, wherein at least one edge weight is determined by a learnable combination parameter optimized during training;
>
> (vii) perform economic contagion message passing on the graph, wherein attention coefficients between pairs of agricultural regions are conditioned on an asymmetric shock embedding of a commodity price shock at a source region, the asymmetric shock embedding computing a first embedding by applying a first learned linear transformation to the positive component of the price shock and a second embedding by applying a second, independently parameterized learned linear transformation to the negative component of the price shock, and summing the first and second embeddings;
>
> (viii) generate, for each agricultural region, a waste probability indicating the likelihood that a crop waste event will occur in the region within a specified prediction horizon; and
>
> (ix) output the waste probability to at least one of a display device, a storage device, and a communication interface.

---

### Claim 2 --- Method

**2.** A computer-implemented method for predicting crop waste events across a plurality of agricultural regions, the method comprising:

(a) ingesting, for each agricultural region, multimodal input data comprising biophysical features derived from remote sensing and weather observations, and economic features derived from commodity market data;

(b) encoding the biophysical features into a first latent representation using a first neural network encoder;

(c) encoding the economic features into a second latent representation using a second neural network encoder;

(d) applying an adversarial training objective to enforce independence between the first and second latent representations, comprising passing the first latent representation through a gradient reversal layer and training a discriminator to reconstruct the second latent representation from the gradient-reversed first latent representation;

(e) constructing a dynamic graph with nodes representing agricultural regions and edges representing economic and geographic relationships, wherein edge weights are determined by a learned combination of at least two of geographic proximity, commodity market similarity, and transportation connectivity;

(f) performing graph neural network message passing on the dynamic graph using attention coefficients conditioned on asymmetric commodity price shock embeddings, wherein for each source node in the graph:

> - computing a scalar price shock value representing a percentage change in commodity price over a time horizon;
> - decomposing the price shock value into a positive component via max(shock, 0) and a negative component via min(shock, 0);
> - transforming the positive component using a first learned linear projection to produce a positive embedding;
> - transforming the negative component using a second learned linear projection, independently parameterized from the first, to produce a negative embedding;
> - summing the positive and negative embeddings to produce an asymmetric shock embedding;
> - concatenating the shock embedding with projected node features of the source and target nodes;
> - computing attention coefficients from the concatenated vector using a learnable attention vector and nonlinear activation;
> - normalizing attention coefficients across all incoming edges to each target node;

(g) generating a waste prediction for each agricultural region based on the graph-enriched node representations;

(h) generating a cause-of-loss classification for each agricultural region indicating the predicted mechanism of a potential waste event; and

(i) providing the waste prediction and cause-of-loss classification to at least one of a user interface, a database, and an alert system.

---

### Claim 3 --- ECMP (Domain-Independent)

**3.** A computer-implemented method for graph neural network message passing with asymmetric shock conditioning, the method comprising:

(a) receiving a graph comprising a plurality of nodes and a plurality of edges connecting pairs of nodes, each node associated with a feature vector of dimension d_in and each node associated with a scalar shock value representing a directional exogenous perturbation at the node;

(b) for each node, computing an asymmetric shock embedding by:

> - decomposing the scalar shock value into a positive component equal to max(shock_value, 0) and a negative component equal to min(shock_value, 0);
> - applying a first learned linear transformation of dimension s x 1 to the positive component to produce a positive embedding vector in R^s;
> - applying a second learned linear transformation of dimension s x 1, with parameters independent of the first transformation, to the negative component to produce a negative embedding vector in R^s;
> - summing the positive and negative embedding vectors to produce the asymmetric shock embedding of dimension s;

(c) for each directed edge from a source node j to a target node i, computing an attention coefficient by:

> - projecting the feature vectors of the source node and the target node using a shared linear projection of dimension d' x d_in;
> - concatenating the projected target features, the projected source features, and the asymmetric shock embedding of the source node to form a concatenated vector of dimension 2*d' + s;
> - computing a scalar attention score by computing the inner product of a learnable attention vector of dimension 2*d' + s with the concatenated vector, followed by a leaky rectified linear unit activation;
> - normalizing the attention score across all edges incoming to the target node using a softmax function to produce a normalized attention coefficient;

(d) for each target node, computing an updated feature vector by weighted aggregation of transformed source node features, where each source node's contribution is weighted by the corresponding normalized attention coefficient; and

(e) outputting the updated feature vectors for all nodes in the graph.

---

## Dependent Claims on Claim 1 (System)

**4.** The system of claim 1, wherein the biophysical encoder neural network comprises a multilayer perceptron with batch normalization and rectified linear unit activations, and the economic encoder neural network comprises a multilayer perceptron with a hidden dimension smaller than that of the biophysical encoder, reflecting the lower dimensionality of the economic feature space.

**5.** The system of claim 1, wherein the adversarial disentanglement module further comprises a warm-up schedule that sets the gradient reversal strength to zero for a predetermined number of initial training epochs before activating adversarial training at a target reversal strength.

**6.** The system of claim 1, wherein the asymmetric shock embedding uses a shock embedding dimension of at least 4, and wherein the economic contagion message passing uses multi-head attention with at least 2 independently parameterized attention heads, each head learning distinct contagion patterns.

**7.** The system of claim 1, wherein the graph construction uses learnable combination parameters for geographic proximity, commodity market similarity, and transportation connectivity, said parameters normalized via softmax to ensure the parameters are non-negative and sum to one, and wherein the graph is sparsified by retaining only the top-K neighbors per node, where K is a configurable parameter.

**8.** The system of claim 1, further comprising a cause classification head that categorizes predicted waste events into at least three cause categories including at least one biophysical cause category selected from drought, excess moisture, cold, and heat, and at least one economic cause category comprising price decline.

**9.** The system of claim 1, wherein the waste prediction is trained using focal loss with a focusing parameter gamma greater than 1.0 and a positive class weight alpha greater than 0.5, to address class imbalance between waste and non-waste observations, and wherein the total training loss further comprises a cause classification loss weighted by a first coefficient and a disentanglement loss weighted by a second coefficient.

**10.** The system of claim 1, wherein the instructions further cause the processor to generate an alert when the waste probability for a given agricultural region exceeds a configurable threshold, the alert comprising the waste probability, the predicted cause category, an identification of connected regions contributing most to the waste risk based on the attention coefficients, and an estimated economic impact based on historical indemnity data.

**11.** The system of claim 1, wherein the economic contagion message passing comprises two or more stacked message passing layers with residual connections and layer normalization between layers, enabling multi-hop contagion propagation through indirect pathways in the graph.

**12.** The system of claim 1, wherein the commodity market similarity component of the graph is time-varying and recomputed at each prediction time step based on current production profiles, price correlations, and shared market access patterns between agricultural regions.

---

## Dependent Claims on Claim 2 (Method)

**13.** The method of claim 2, wherein the biophysical features comprise at least three of: normalized difference vegetation index (NDVI), enhanced vegetation index (EVI), growing degree days, frost days, Palmer Drought Severity Index, surface soil moisture, root-zone soil moisture, and USDA VegScape vegetation condition index.

**14.** The method of claim 2, wherein the economic features comprise at least three of: commodity price level, one-month price change, three-month price change, 30-day price volatility, cost of production, revenue-to-cost ratio, futures basis, and national supply estimate.

**15.** The method of claim 2, further comprising applying temporal constraints to prevent data leakage, wherein features for predicting waste at time t use only data available before time t with at least a configurable minimum lag period.

**16.** The method of claim 2, wherein the dynamic graph is time-varying, with commodity market similarity edges recomputed at each time step based on current production profiles, price correlations, and market access patterns between agricultural regions.

**17.** The method of claim 2, further comprising training the method end-to-end by minimizing a combined loss function comprising a waste prediction focal loss, a cause classification cross-entropy loss weighted by a first coefficient, and a disentanglement mean squared error loss weighted by a second coefficient.

**18.** The method of claim 2, wherein generating the waste prediction comprises generating predictions at multiple prediction horizons including at least a 2-week horizon and a 4-week horizon, enabling lead-time-dependent decision making by stakeholders.

**19.** The method of claim 2, further comprising computing an economic impact estimate by multiplying the waste probability for each agricultural region by a historical indemnity amount for that region, and aggregating across regions to produce an estimated total value of preventable crop waste.

---

## Dependent Claims on Claim 3 (ECMP, Domain-Independent)

**20.** The method of claim 3, wherein the graph neural network message passing is performed with multi-head attention comprising H independently parameterized attention heads, each head having independent learned parameters for the first and second linear transformations, the shared linear projection, and the learnable attention vector, and wherein head outputs are concatenated in intermediate layers and averaged in a final layer to produce the updated feature vector.

**21.** The method of claim 3, further comprising stacking two or more layers of graph neural network message passing with residual connections and layer normalization between layers, wherein a learned linear projection matches dimensions for the residual connection when input and output dimensions of a layer differ.

**22.** The method of claim 3, wherein the scalar shock value represents a financial market perturbation selected from asset price change, interest rate change, credit spread change, and volatility change, and the graph represents a financial network with nodes representing financial entities and edges representing financial exposures, counterparty relationships, or correlated portfolio holdings between entities.

**23.** The method of claim 3, wherein the scalar shock value represents a supply chain disruption magnitude selected from production capacity change, delivery delay, and demand fluctuation, and the graph represents a supply chain network with nodes representing supply chain entities and edges representing supplier-buyer relationships between entities.

**24.** The method of claim 3, wherein the scalar shock value represents an epidemiological metric change selected from infection rate change, hospitalization rate change, and mortality rate change, and the graph represents a population mobility network with nodes representing geographic regions and edges representing population movement flows between regions.

**25.** The method of claim 3, wherein the scalar shock value represents an energy grid perturbation selected from generation capacity change, demand change, and transmission capacity change, and the graph represents an energy transmission network with nodes representing grid zones and edges representing transmission lines or energy flow connections.

**26.** The method of claim 3, further comprising incorporating edge features into the attention computation by projecting edge feature vectors through a learned linear layer to produce edge biases of dimension equal to the number of attention heads, and adding the edge biases to the raw attention scores prior to softmax normalization.
