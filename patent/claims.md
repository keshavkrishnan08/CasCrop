# CasCrop Patent Claims

## Independent Claims

### Claim 1 (System)

1. A computer-implemented system for predicting agricultural crop waste events, comprising:

   (a) a biophysical encoder module configured to receive satellite imagery data, weather data, and soil moisture data for a plurality of agricultural production regions, and to generate a biophysical latent representation for each region;

   (b) an economic encoder module configured to receive commodity price data, production cost data, and market signal data, and to generate an economic latent representation for each region;

   (c) an adversarial disentanglement module configured to enforce statistical independence between said biophysical latent representation and said economic latent representation through adversarial training, comprising a discriminator network trained to predict economic representations from biophysical representations, and a gradient reversal mechanism that penalizes the biophysical encoder when the discriminator succeeds;

   (d) a graph neural network module operating on a dynamic graph wherein nodes represent agricultural production regions and edges represent economic and geographic relationships between regions, said graph neural network comprising an attention mechanism wherein attention coefficients between a source node and a destination node are computed as a function of:
   - projected feature representations of the source and destination nodes, and
   - a directional shock embedding that applies separate learned linear transformations to positive price changes and negative price changes at the source node;

   (e) a prediction module that receives graph-updated node representations and generates a crop waste probability estimate for each agricultural production region.

### Claim 2 (Method)

2. A computer-implemented method for predicting crop waste events in agricultural production regions, comprising the steps of:

   (a) receiving, for each of a plurality of agricultural production regions, biophysical feature data comprising at least one of satellite-derived vegetation indices, weather measurements, and soil moisture measurements;

   (b) receiving, for each region, economic feature data comprising at least one of commodity futures prices, local production costs, and price change rates;

   (c) encoding said biophysical feature data into a biophysical latent vector using a first neural network encoder;

   (d) encoding said economic feature data into an economic latent vector using a second neural network encoder;

   (e) computing a disentanglement loss that measures and penalizes mutual information between said biophysical and economic latent vectors;

   (f) constructing a graph structure wherein nodes correspond to agricultural regions and edges represent at least one of geographic adjacency, commodity market connectivity, and transportation network proximity;

   (g) for each edge in said graph, computing an attention coefficient that is conditioned on the magnitude and sign of a commodity price change at the source node, wherein positive price changes and negative price changes are processed through separate learned transformation functions;

   (h) aggregating neighbor node representations using said attention coefficients to produce graph-updated representations;

   (i) generating, from said graph-updated representations, a waste probability for each region indicating the likelihood that crops in said region will be abandoned, left unharvested, or otherwise wasted.

### Claim 3 (ECMP — Domain-Independent)

3. A graph neural network message passing method for propagating information through a network of interconnected nodes, comprising:

   (a) receiving node feature vectors for a plurality of nodes and an exogenous directional shock variable associated with each node;

   (b) for each directed edge from a source node to a destination node, computing an attention coefficient by:
   - projecting source and destination node features through a learned linear transformation,
   - computing a shock embedding of the source node's directional shock variable through asymmetric processing, wherein positive shock values are transformed by a first learned linear function and negative shock values are transformed by a second, distinct learned linear function,
   - concatenating the projected source features, projected destination features, and the shock embedding to form an attention input vector,
   - applying a learned attention vector to said input to produce a raw attention score,
   - normalizing attention scores across all incoming edges to each destination node;

   (c) aggregating source node representations weighted by said attention coefficients to produce updated node representations;

   wherein said first and second learned linear functions are independently parameterized to capture asymmetric propagation dynamics of positive and negative exogenous shocks through the network.

---

## Dependent Claims

### Claims Dependent on Claim 1

4. The system of claim 1, wherein the dynamic graph is updated at each time step by recomputing commodity connectivity edges based on current price correlations between regions.

5. The system of claim 1, wherein edge weights in the dynamic graph are computed as a weighted combination of geographic adjacency, commodity market connectivity, and transportation network proximity, with learnable combination weights.

6. The system of claim 1, wherein the prediction module further comprises a cause-of-loss classifier that predicts the category of waste cause from among drought, excess moisture, cold, heat, price decline, and other causes.

7. The system of claim 1, wherein the graph neural network module comprises a stack of at least two attention layers with residual connections and layer normalization.

8. The system of claim 1, wherein the biophysical encoder and economic encoder each comprise multi-layer perceptron architectures with batch normalization.

9. The system of claim 1, wherein the system is trained using a focal loss function with class weighting to handle imbalanced waste/no-waste label distributions.

10. The system of claim 1, wherein crop waste training labels are derived from USDA Risk Management Agency crop insurance indemnity claims.

### Claims Dependent on Claim 2

11. The method of claim 2, further comprising: training the first and second neural network encoders with a warm-up period during which the disentanglement loss is set to zero, followed by activation of adversarial disentanglement training.

12. The method of claim 2, wherein the attention coefficient computation uses multi-head attention with a plurality of independently learned attention heads, and results from said heads are concatenated or averaged.

13. The method of claim 2, wherein the graph structure is sparsified by retaining only the top-K highest-weighted edges for each node.

14. The method of claim 2, further comprising: generating an early warning alert when the waste probability for a region exceeds a predetermined threshold at a specified lead time.

### Claims Dependent on Claim 3

15. The method of claim 3, wherein said network of interconnected nodes represents a financial network and said exogenous directional shock variable represents an asset price change.

16. The method of claim 3, wherein said network represents a supply chain and said exogenous directional shock variable represents a demand or supply perturbation.

17. The method of claim 3, wherein the attention coefficients further incorporate edge feature information through a learned edge projection.

18. The method of claim 3, wherein the asymmetric processing further comprises normalizing the shock embedding magnitude to prevent gradient explosion from extreme shock values.
