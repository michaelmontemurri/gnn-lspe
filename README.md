# Exploration of Positional Encodings within the LSPE Framework

This repository builds upon the **[Graph Neural Networks with Learnable Structural and Positional Representations (LSPE)](https://openreview.net/pdf?id=wTTjnvGphYj)** by Dwivedi et al., originally presented at ICLR 2022. The original LSPE framework decouples structural and positional embeddings in GNNs and updates both through learnable modules. While the initial paper explored Laplacian and Random Walk-based positional encodings, many alternative strategies remain underexplored.

> 📎 **This repository includes our extended exploration of initial positional encodings within the LSPE framework, including:**  
> - Generalized PageRank distance encodings  
> - Random anchor-based encodings  
> - Structure-Preserving Embeddings (SPE)  
> - Centrality-based encodings (degree, closeness, betweenness)

Our full report is available [here (PDF)](./docs/Exploration_of_PE_in_LSPE_Framework.pdf).

---

## 🔬 Research Contributions

We systematically evaluate the effects of these alternative initialization strategies across multiple molecular property prediction benchmarks: **ZINC**, **OGBG-MOLTOX21**, and **OGBG-MOLPCBA**, using four model backbones provided in the original LSPE paper:

- GatedGCN
- PNA
- SAN (Self-Attention Network)
- GraphiT (Graph Transformer with structural priors)

Key findings include:

- **Closeness centrality** outperforms RWPE on MOLTOX21 in nearly all model variants.
- **Anchor-based and PageRank-based PEs** show promise but often suffer from overfitting or instability on small molecular graphs.
- **SPE embeddings** preserve topological structure well but are computationally infeasible at scale.
- We propose **element-aware sampling** for PageRank anchors to mitigate overfitting.

---

## 📦 Repository Structure

This repo is based on a fork of [vijaydwivedi75/gnn-lspe](https://github.com/vijaydwivedi75/gnn-lspe) and includes:

- 🧪 Our added PE modules: `positional_encodings.py`
- 📄 Reproducibility scripts for new experiments: `scripts/`
- 📊 Extended results, evaluation logs, and visualization scripts: `logs/`, `plots/`
- 📁 All original baselines and models preserved

---

## 🚀 Getting Started

### 1. Installation

Follow the environment setup instructions here:  
[📖 docs/01_repo_installation.md](./docs/01_repo_installation.md)

### 2. Dataset Downloads

Download and preprocess benchmark datasets:  
[📖 docs/02_download_datasets.md](./docs/02_download_datasets.md)

### 3. Running Experiments

Reproduce our extended experiments or baseline results:  
[📖 docs/03_run_codes.md](./docs/03_run_codes.md)

---

## 📚 References

```bibtex
@inproceedings{dwivedi2022graph,
  title={Graph Neural Networks with Learnable Structural and Positional Representations},
  author={Vijay Prakash Dwivedi and Anh Tuan Luu and Thomas Laurent and Yoshua Bengio and Xavier Bresson},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=wTTjnvGphYj}
}



<br><br><br>

