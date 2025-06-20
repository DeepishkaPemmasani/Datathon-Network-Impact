# Datathon-Network-Impact
Datathon Network Impact - Comparing the evolution of scholarly growth and Networking in attendees and presenters of SCCM datathon over the years from 2017 vs 2023.
# Analysis Overview: 
Comparing Diversity in SCCMDAT and SCCMPRES Datasets in the years 2017 vs 2023.
# Datasets used:

1. For gender diversity analysis:
   
•	"SCCMDATupdated_FINALANALYSISCSV.csv"

•	"SCCMPRESupdated_FINALANALYSIS_csv.csv"

2. For Country diversity analysis:
  
•  "fork_publication_summary (2)_countries_updated.csv" (for SCCMDAT)

•	"PRESfork_publication_summary (2)_countries_updated.csv" (for SCCMPRES)

3.	For Educational diversity analysis
   
•	"finalcombanalysis_degree_comparison_updated.csv" (for SCCMDAT)

•	"finalcombpresanalysis_degree_comparison_updated.csv" (for SCCMPRES)

# Objective: 
This analysis aims to compare diversity between two datasets, SCCMDAT and SCCMPRES, focusing on three key aspects:

1.	Gender diversity
   
3.	Country diversity (based on high-income vs. low-income countries)
   
5.	Educational diversity
   
# Methodology:
•	Utilizes the Gini Impurity Index to quantify diversity

•	Gini Impurity scale: 0 (complete homogeneity) to 1 (maximum diversity)

•	Analysis conducted at the publication level for each dataset

•	Comparison of results between 2017 and 2023 for both datasets (SCCMDAT and SCCMPRES)

# Analysis Components:
1.	Gender Diversity Analysis:
   
o	Employed NLTK (Natural Language Toolkit) for gender prediction based on author names

o	Developed a function to calculate Gini impurity for gender diversity

o	Applied the function to SCCMDAT and SCCMPRES datasets for 2017 and 2023

o	Created visualizations (box plots, histograms) to compare gender diversity across datasets and years

3.	Country Diversity Analysis:
   
o	Utilized datasets categorizing co-authors as from high-income or low-income countries

o	Developed a function to calculate Gini impurity for country diversity

o	Applied the function to both datasets for 2017 and 2023

o	Generated visualizations (box plots, line plots) to illustrate country diversity trends over time

5.	Educational Diversity Analysis:
   
o	Incorporated datasets containing information on authors' highest educational degrees

o	Developed a function to calculate Gini impurity for educational diversity

o	Applied the function to both SCCMDAT and SCCMPRES datasets

o	Produced visualizations (box plots, histograms, scatter plots) to compare educational diversity between datasets

