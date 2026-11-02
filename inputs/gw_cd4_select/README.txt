# Data overview

This dataset provides a high-resolution, genome-scale map of how expressed genes regulate the behavior of primary human immune cells. Unlike studies that rely on immortalized model cell lines (for example, K562 or Jurkat), this atlas was built using **primary cells** isolated from four healthy human donors. Using primary cells preserves the biological relevance needed to study complex immune traits and autoimmune disease susceptibility.
The paper describing this study is in data/gwcd4.pdf.

---

### Experimental Design and Generation

The data was generated using a scalable, probe-based **perturb-seq platform**. Researchers employed **CRISPR interference (CRISPRi)** to systematically knock down **12,748 genes**, comprising nearly all expressed genes in T cells plus all known human transcription factors. To capture immune dynamics, cells were profiled across three physiological states:

- **Resting state (Rest):** Cells expanded ex vivo and allowed to reach a quiescent state without active stimulation.
- **Early activation (Stim8hr):** Cells 8 hours post-restimulation, capturing immediate-early regulatory circuits of the T cell receptor (TCR).
- **Late activation (Stim48hr):** Cells 48 hours post-restimulation, reflecting transition to metabolic expansion and effector functions.

The team leveraged **10x Genomics Chromium Flex** technology to "superload" and profile approximately **22 million cells** with high sensitivity, even for low-abundance transcripts.

---

### Data Filtering and Statistical Rigor

Processing involved multiple filters to ensure results reflect true biological "trans-effects" (the effect of a regulator on downstream genes) rather than technical noise:

- **Single-cell quality control:** Cells were filtered by mitochondrial content and genes detected to retain high-quality transcriptomes. Only cells with a single, confidently assigned guide RNA were used to avoid confounding effects from multiple perturbations.
- **On-target verification:** Measured "cis-effects" to confirm CRISPRi reduced expression of the intended target gene.
- **Off-target safeguards:** Flagged potential off-target effects by checking whether a perturbation silenced neighboring genes within 10 kb of the target.
- **Pseudobulk aggregation:** Grouped cells by donor, condition, and guide to increase statistical power and enable robust differential expression analysis with **DESeq2**, accounting for donor-level variability.

---

### Biological Context

This dataset is designed as a foundational resource for decoding human immune function. It identifies clusters of regulators governing core cellular processes (for example, citrate metabolism, mTORC1 signaling, and vesicle trafficking) and reveals how these networks rewire during an immune response. By comparing perturbations to natural genetic variation in large populations (such as the UK Biobank), the data helps explain how specific genetic regions contribute to traits like lymphocyte counts or autoimmune disease risk.

## File tree:
Directory path: {datapath}
Filename: GWCD4i.DE_stats.h5ad

This AnnData object contains genome-wide differential expression results from a perturb-seq experiment in CD4+ T cells. Each observation represents a single perturbation (perturbed gene) tested in a specific culture condition (n_obs = 33,983). Each variable is a measured gene in the transcriptome (n_vars = 10,282).

**Observation Metadata (.obs)**
Annotations for each perturbation-condition pair:
`target_contrast_gene_name`: Name of the perturbed gene
`culture_condition`: culture condition (Rest, Stim8hr, Stim48hr)
`target_contrast`: Unique identifier for the perturbed gene
`chunk`: differential expression processing group identifier
`n_cells_target`: Number of cells with targeting guide for the perturbed gene
`n_up_genes`: Count of significantly upregulated genes (10% FDR)
`n_down_genes`: Count of significantly downregulated genes (10% FDR)
`n_total_de_genes`: Total number of significantly differentially expressed genes (10% FDR)
`ontarget_effect_size`: Effect size of the perturbation on its intended target gene
`ontarget_significant`: Boolean indicating whether on-target knockdown was significant (10% FDR)
`target_baseMean`: Mean baseline expression of the target gene
`offtarget_flag`: Flag indicating potential off-target effects (TSS within 10 kb with significant down-regulation)
`n_total_genes_category`: Category based on number of trans-effects
`ontarget_effect_category`: Category based on on-target / off-target effects
`n_downstream`: Number of genes significantly affected by this perturbation, excluding on-target effect (incoming trans-effects)

**Variable Metadata (.var)**
Annotations for each measured gene:
`gene_ids`: Gene identifiers (e.g., Ensembl IDs)
`gene_name`: Gene symbols

**Variable Matrices (.varm)**
Summary statistics for measured genes across conditions:

`measured_genes_stats_Stim8hr`: Gene-level statistics for 8-hour stimulation condition
`measured_genes_stats_Stim48hr`: Gene-level statistics for 48-hour stimulation condition
`measured_genes_stats_Rest`: Gene-level statistics for resting/unstimulated condition

**Data Layers (.layers)**
Differential expression statistics for each perturbation-gene pair (from DESeq2):

`log_fc`: Log2 fold change
`p_value`: Raw p-values from differential expression testing
`adj_p_value`: FDR-adjusted p-values
`baseMean`: Mean normalized expression of the gene across cells
`lfcSE`: Standard error of log fold change
`zscore`: Z-scores for differential expression (logFC / lfcSE)