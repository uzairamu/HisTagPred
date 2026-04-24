# Data Description

## final_dataset.csv
1,567 non-redundant cytoplasmic proteins from 5 organisms after CD-HIT 40% filtering.

Columns:
- protein: UniProt accession and gene name
- fasta_sequence: amino acid sequence
- n_plddt_native: mean pLDDT of N-terminal 10 residues (native structure)
- c_plddt_native: mean pLDDT of C-terminal 10 residues (native structure)
- n_plddt_ntag: mean pLDDT of N-terminal 10 residues (N-tagged structure)
- c_plddt_ctag: mean pLDDT of C-terminal 10 residues (C-tagged structure)
- drop_N: pLDDT drop at N-terminus upon N-tag insertion
- drop_C: pLDDT drop at C-terminus upon C-tag insertion
- drop_diff: |drop_N - drop_C|
- Label: C_terminal or N_terminal (preferred placement)
- organism: source organism
- local_rmsd_N: local backbone RMSD at N-terminal 10 residues
- local_rmsd_C: local backbone RMSD at C-terminal 10 residues

## combined_with_rmsd.csv
Full dataset before CD-HIT filtering (1,984 proteins, Equal class removed).
Same columns as final_dataset.csv.

## Label definition
- C_terminal: N-terminus is more perturbed by tag insertion
  (drop_N > drop_C by >= 2.0 pLDDT units)
  → C-terminus is more tolerant → place tag at C-terminus
- N_terminal: C-terminus is more perturbed
  (drop_C > drop_N by >= 2.0 pLDDT units)
  → N-terminus is more tolerant → place tag at N-terminus
