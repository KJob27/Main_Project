ipf.py

This script provides the 3D spatial gene expression pattern for ~9000 genes of tomoseq data constructed from the mouse olfactory mucosa. (1). 
The algorithm used to construct the 3d spatial pattern is IPF (Iterative proportional fitting) - As this reconstructs the pattern for each gene separately, this needs to be run on the server, as it otherwise will exhaust the memory.
The data is RPM normalized and fitted when inserting it to the algorithm.
In the algorithm it is normalized by the volume of each slice.



(1) The dataset is available here: https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-10211?query=E-MTAB-10211#
