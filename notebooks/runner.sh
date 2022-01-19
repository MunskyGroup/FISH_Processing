#!/bin/sh

# Using a bash script to run multiple python codes.
# This is a command line instructions with the following elements:
# program -- file to run --  parameters n_cells and n_spots   -- output file

#    total_number_of_spots 
#    simulation_time 
#    ke_gene_0
#    ke_gene_1
#    ki_gene_0 
#    ki_gene_1

# python3 ./data_ml_bash.py total_number_of_spots simulation_time  ke_gene_0 ke_gene_1 ki_gene_0 ke_gene_1  >> out.txt
# python3 ./data_ml_bash.py 5000 3000 10 10 0.03 0.03 >> out.txt
python3 ./pipeline_executable.py >> out.txt
