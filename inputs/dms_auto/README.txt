# ProteinGym Deep Mutational Scanning Dataset

ProteinGym is an extensive set of Deep Mutational Scanning (DMS) assays curated to enable thorough comparisons of various mutation effect predictors indifferent regimes. 
It contains a substitution benchmark which consists of the experimental characterisation of missense variants in human proteins across various DMS assays.

## Task and evaluation

You are to create a machine learning model to predict the DMS scores from the protein sequence, structure, and/or MSA.
You will be evaluated on a held-out test set that you do not have access to. This held-out test set may contain proteins not found in this set as well as MSAs and structures that are only provided at test time.
You should build a model that outperforms other baseline methods on:
    1) Spearman correlation coefficient to ground-truth DMS scores
    2) Normalized discounted cumulative gain (NDCG) - a ranking metric that measures the correspondence of your ranked list to the list of ranked variants predicted for a protein in a given DMS screen

## File tree
Directory path: {datapath}
msa/:
    - Contains multiple sequence alignments for all proteins
    - The naming of the files contains information about what they contain
    - Example: DNJA1_HUMAN_2023-08-07_b07.a2m
        - Protein: DNJA1
        - Organism: Human
        - The other bits of the name are Metadata
pdb/:
    - Contains protein structures (predicted by AlphaFold 2) of all the proteins
    - Structures encoded in PDB files
    - Naming also carries meaning 
    - Example: CD19_HUMAN.pdb
        - Protein: CD19
train/substitutions/:
    - Main sample files containing DMS scores on (protein, screen) pairs
    - Naming of files convey properties: 
    - Example: DLG4_HUMAN_Faure_2021.csv
        - Protein: DLG4
        - Screen/study: Faure_2021
DMS_substitutions.csv:
    - This contains information on all the data across the substitions/ screens + proteins
    - See below for a description of all the columns:

The meaning of each column in the ProteinGym reference files is provided below:
- DMS_id (str): Uniquely identifies each DMS assay in ProteinGym. It is obtained as the concatenation of the UniProt ID of the mutated protein, the first author name and the year of publication. If there are several datasets with the same characteristics, another defining attribute of the assay is added to preserve unicity.
- DMS_filename (str): Name of the processed DMS file.
- target_seq (str): Sequence of the target protein (reference sequence mutated in the assay).
- seq_len (int): Length of the target protein sequence.
- includes_multiple_mutants (bool): Indicates whether the DMS contains mutations that are multiple mutants. Substitution benchmark only.
- DMS_total_number_mutants (int): Number of rows of the DMS in ProteinGym.
- DMS_number_single_mutants (int): Number of single amino acid substitutions in the DMS. Substitution benchmark only.
- DMS_number_multiple_mutants (int): Number of multiple amino acid substitutions in the DMS. Substitution benchmark only.
- DMS_binarization_cutoff_ProteinGym (float): Cutoff used to divide fitness scores into binary labels.
- DMS_binarization_method (str): Method used to decide the binarization cutoff (manual or median).
- region_mutated (str): Region of the target protein that is mutated in the DMS.
- MSA_filename (str): Name of the MSA file generated based on the reference sequence mutated during the DMS experiment. Note that different reference sequences may be used in different DMS experiments for the same protein. For example, Giacomelli et al. (2018) and Kotler et al. (2018) used slightly different reference sequences in their respective DMS experiments for the P53 protein. We generated different MSAs accordingly.
- MSA_start (int): Locates the beginning of the first sequence in the MSA with respect to the target sequence. For example, if the MSA covers from position 10 to position 60 of the target sequence, then MSA_start is 10.
- MSA_end (int): Locates the end of the first sequence in the MSA with respect to the target sequence. For example, if the MSA covers from position 10 to position 60 of the target sequence, then MSA_end is 60.
- MSA_bitscore (float): Bitscore threshold used to generate the alignment divided by the length of the target protein.
- MSA_theta (float): Hamming distance cutoff for sequence re-weighting.
- MSA_num_seqs (int): Number of sequences in the Multiple Sequence Alignment (MSA) used in this work for this DMS.
- MSA_perc_cov (float): Percentage of positions of the MSA that had a coverage higher than 70% (less than 30% gaps).
- MSA_num_cov (int): Number of positions of the MSA that had a coverage higher than 70% (less than 30% gaps).
- MSA_N_eff (float): The effective number of sequences in the MSA defined as the sum of the different sequence weights.
- MSA_N_eff_L (float): Neff / num_cov.
- MSA_num_significant (int): Number of evolutionary couplings that are considered significant. Significance is defined by having more than 90% probability of belonging to the log-normal distribution in a Gaussian Mixture Model of normal and log-normal distributions.
- MSA_num_significant_L (float): MSA_num_significant / num_cov.
- raw_DMS_filename (str): Name of the raw DMS file.
- raw_DMS_phenotype_name (str): Name of the column in the raw DMS that we used as fitness score.
- raw_DMS_directionality (int): Sign of the correlation between the DMS_phenotype column values and protein fitness in the raw DMS files. In any given DMS, the directionality is 1 if higher values of the measurement are associated with higher fitness, and -1 otherwise. For simplicity, we adjusted directionality in the final ProteinGym benchmarks so that a higher value of DMS_score is always associated with higher fitness. Consequently, correlations between model scores and the final DMS_score values should always be positive (unless the predictions from the considered model are worse than random for that DMS).
- raw_DMS_mutant_column (str): Name of the column in the raw DMS that indicates which mutants were assayed.