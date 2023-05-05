# GAFExplainer
Code for paper "GAFExplainer: Global Explanation of Graph Neural Networks through Attribute Augmentation and Fusion Embedding"
## Abstract:
The excellent performance of graph neural networks (GNNs), which learn node representations by aggregating their neighborhood information, has led to their use in various graph tasks. However, GNNs are black box models, the prediction results of which are difficult to understand directly. Although node attributes are vital in making predictions, previous studies have ignored their importance for explanation. This study presents GAFExplainer, a novel GNN explainer that emphasizes node attributes via attribute augmentation and fusion embedding in a global view. In addition, a weight generator and mask fuser are set to select suitable pseudo discrete masks, thus rendering the explainable network trainable while easing the "introduced evidence" problem. By training the network, a global explanation of the GNN models is obtained, and reasonably explainable subgraphs are available for new graphs, thus rendering the model well generalizable. Multiple sets of experimental results on real and synthetic datasets demonstrate that the proposed model provides valid and accurate explanations. In the visual analysis, the explanations obtained by the proposed model are more comprehensible than those in existing work. Further, the fidelity evaluation and efficiency comparison reveal that with an average performance improvement of 8.9% compared with representative baselines, GAFExplainer achieves the best fidelity metrics while maintaining computational efficiency.

## Requirements：
matplotlib==3.5.2 <br/>
networkx==2.7.1 <br/>
numpy==1.21.6 <br/>
rdkit==2022.9.3 <br/>
rdkit_pypi==2022.3.2.1 <br/>
scipy==1.8.0 <br/>
torch==1.11.0 <br/>
torch_geometric==2.0.4 <br/>
torch_sparse==0.6.13 <br/>
tqdm==4.64.0 <br/>
## Datasets:
|     Datasets            |     MUTAG    |     BBBP     |     BA-2Motif    |     BA-Shape    |
|-------------------------|--------------|--------------|------------------|-----------------|
|     Edges（average）    |     30.77    |     25.95    |     25.48        |     4110        |
|     Nodes（average）    |     30.32    |     24.06    |     25.00        |     700         |
|     Graphs              |     4337     |     2039     |     1000         |     1           |
|     Classes             |     2        |     2        |     2            |     4           |

## The structure of the code is as follows:
Process folder: Preprocessing raw data <br/>
`AAFExplainer.py` : The main file that explains the network <br/>
`GNNs.py`: Train the GNN model and get the edge node embedding <br/>
`main.py`: run the program <br/>
`metrics.py`: metrics for quantitative evaluation of models <br/>
`node_aa.py`: node attribute enhancement module <br/>
`utils.py`: some useful utility functions <br/>

After configuring the required environment, run the `main.py` file directly.
<br/>
### Take the explanation of the GAT model on the mutag dataset as an example:

**1 Data preprocessing**<br/>
Download the MUTAG dataset through the built-in `InMemoryDataset` package of `torch_geometric.data`
Data processing of the MUTAG dataset through `mol_dataset.py`<br/>
<br/>**2 Train the GAT model**<br/>
`GNNs.py` training GNN model
The node embedding of each layer can be obtained through the trained GNN model for edge fusion embedding.<br/>
<br/>**3 Train the AAFExplainer model**<br/>
The final edge mask can be obtained through the trained explanation model.<br/>
<br/>**4 Visualization**<br/>
<br/>**5 Metrics**<br/>
Using fidelity as an evaluation metric：
 
 $$\[\text{ }\text{Fidelity=}\frac{1}{N}\underset{i=1}{\mathop{\overset{N}{\mathop{\sum }}\,}}\,\left( f{{\left( {{\mathcal{G}}_{i}} \right)}_{{{y}_{i}}}}-f{{\left( \mathcal{G}_{Si}^{1-{{m}_{i}}} \right)}_{{{y}_{i}}}} \right)\]$$
 
Fidelity comparisons were performed at identical sparsity levels.


