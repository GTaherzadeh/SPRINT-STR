# SPRINT-STR
Protein–peptide interactions are one of the most important biological interactions and play crucial role in many diseases including cancer. Therefore, knowledge of these interactions provides invaluable insights into all cellular processes, functional mechanisms, and drug discovery. Protein–peptide interactions can be analyzed by studying the structures of protein–peptide complexes. However, only a small portion has known complex structures and experimental determination of protein–peptide interaction is costly and inefficient. Thus, predicting peptide-binding sites computationally will be useful to improve efficiency and cost effectiveness of experimental studies. Here, we established a machine learning method called SPRINT-Str (Structure-based prediction of protein–Peptide Residue-level Interaction) to use structural information for predicting protein–peptide binding residues. These predicted binding residues are then employed to infer the peptide-binding site by a clustering algorithm.


Cite: Taherzadeh, G., Zhou, Y., Liew, A. W. C., & Yang, Y. (2018). Structure-based prediction of protein–peptide binding regions using Random Forest. Bioinformatics, 34(3), 477-484.

Instruction:

* Protein-peptide dataset are stored in Data directory. Dataset file contains protein sequences labeled as 1 and 0 for binding and non-binding residues, respectively. Train and test files contain actual binding residues used in this study.
* Run ./SPRINT.py for feature extraction and peptide binding site prediction.
* The pre-trained model and protein-peptide complexes in pdb format are store as LargeFiles in Release.
