# Final-assignment
the final assignment for programming2 DSLS1

# lungcancer analysis

## Overview
This repository contains some  analysis of two lung cancer datasets. The data were collected from  https://ftp.ncbi.nlm.nih.gov/geo/series/GSE58nnn/GSE58661/matrix and http://doi.org/10.7937/K9/TCIA.2015.L4FRET6Z 

## Purpose
Investigate if there is a meaningful relationship between gene expression and tumor types.
Explore the relationship between NSCLC tumor types, tumor size, and patient gender.

## Methodology
The analysis includes:
- Data cleaning and preprocessing
- Exploratory data analysis (EDA) to visualize and understand data distributions
- Statistical analysis to test hypotheses 
- Visualization of interactions between various factors with different types of tumor

## Files
- `finalassignment475082.ipynb`: Jupyter Notebook containing the complete analysis workflow
- `lungcancer.yaml`: Configuration file with data file path and image references
- `useful_module.py`: Module containing custom statistical functions used in the analysis


## Analysis Details
- Data Inspection: Examining dataset structure, missing values, and data types
- Visualization: Utilizing bar plots, and box plots for data exploration
- Statistical Tests: Applying Mann-Whitney U,1 sample Z test to assess factor impacts on tumor types

## Conclusion
Summary: Based on these datasets, there is no difference between location and size with different tumor types in NSCLC.The proportion of males in squ_cell might be higher than in other types, but because of the small sample, we could not find the difference by chi-square test. The median expressions of the 'Merck-NM_000996_s_at' probe in ade_cell type are significantly different from the Squ_cell samples.

## How to Use
1. Clone the repository.
2. Ensure required dependencies are installed (see requirements.txt)
3. Execute `finalassignment475082.ipynb` in a Jupyter Notebook environment.

NB. This readme is partly generated with chatgpt 3.5

## Author
z.he@st.hanze.nl
