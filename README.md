# Neural Geometries in the Human Hippocampus

## Overview 
Code accompanying:

**Chericoni et al., 2026. _Neural geometry in the human hippocampus enables generalization across spatial position and gaze._**

[Paper on arXiv](https://arxiv.org/abs/2603.04747)

## Contents

* [Data availability](#data-availability)
* [Authorship statement](#authorship-statement)
* [System requirements](#system-requirements)
* [Figure 2 and 4: LN/GLM Analysis and SPAEF Computation](#figure-2-and-4-lnglm-analysis-and-spaef-computation)
* [Figure 3 and 4: Neural Subspace Analyses](#figure-3-and-4-neural-subspace-analyses)
* [Figure 3 and 4: Neural Subspace Orthogonalization and OLS Decoding](#figure-3-and-4-neural-subspace-orthogonalization-and-ols-decoding)
* [Figure 5: CCGP Analysis](#figure-5-ccgp-analysis)

## Data availability
The datasets generated during this study are not publicly available due to participant privacy and institutional restrictions associated with human intracranial recordings but are available from the lead contact upon reasonable request and pending appropriate institutional approvals.

This repository includes behavioral and neural data from one representative subject in the data/ directory.

## Authorship statement
All the codes were implemented by Assia Chericoni

## System requirements
No special hardware requirements beyond a standard workstation. The code is compatible with macOS, Linux, and Windows and requires MATLAB R2024b or later.

## Figure 2 and 4: LN/GLM analysis and SPAEF computation
LN/GLM analysis: Inspired by Hardcastle et al., 2017

### LN/GLM Analysis

Inspired by Hardcastle et al., 2017.

Code for assessing neuronal tuning to task variables and extracting the percentage of tuned neurons:

`Figure 2ABCDE_Figure 4CDE_LN_GLM/MAIN_neural_fit.m`

**Input:** `data/DM.mat` — design matrix containing X and Y position variables and spike trains sampled at 60 Hz.

In the current implementation, predictors are one-hot encoded self and prey positions. The code can be extended to include additional predictors.

**Output:** Bar plots showing the percentage of tuned neurons.

### SPAEF Analysis

Inspired by Koch et al., 2018.

Code for quantifying spatial similarity across tuning maps:

`Figure 2FG_SPAEFAnalysis/MAIN_extract_SPAEF_within_neurons.m`

**Input:** `data/tuningCurves.mat` — tuning functions for all neurons computed over 36 spatial bins.

**Output:** Box plots showing SPAEF values across spatial maps for each neuron.


## Figure 3 and 4: Neural subspaces analyses
Inspired by Elsayed et al., 2016 and Yoo & Hayden 2020

### Covariance Structure and Agent Preference Index (API)

Code for inspecting covariance/correlation structure across neuronal tuning functions and computing the Agent Preference Index (API):

`Figure 3ABCDE_Figure 4HIJ_SubspaceAnalysis/covarianceAnalysis_andAPI_Panels3AB - 4HI/MAIN_inspect_covarianceStructure_acrossSelfPrey.m`

**Input:** `data/tuningCurves.mat`

**Output:** Covariance matrices and API histograms.

The example implementation is provided for self versus prey representations; analogous code is included for all other agent and gaze comparisons.

### PCA Decomposition and Shared Variance Analysis

Code for performing PCA decomposition on self representations and projecting prey representations onto the self subspace:

`Figure 3ABCDE_Figure 4HIJ_SubspaceAnalysis/PCAdecomposition_andProjections_Panels3CD - 4J/MAIN_PCADecomposition_andProject.m`

**Input:** `data/tuningCurves.mat`

**Output:** PCA variance explained.

Analogous code is provided for other agent and gaze comparisons. Code implementing half-split cross-validation analyses starting from raw data is also included.

### Alignment Index Analysis

Code for computing the Alignment Index and performing random-subspace control analyses:

`Figure 3ABCDE_Figure 4HIJ_SubspaceAnalysis/alignmentIndex_Panel3E - 4K/MAIN_computeAlignmentIndex_allCombinations.m`

**Input:** `data/tuningCurves.mat`

**Output:** Alignment indices across representations compared against random subspaces.

## Figure 3 and 4: Neural subspaces orthogonalization and OLS decoding 
Inspired by Elsayed et al., 2016 and Yoo and Hayden 2020

Code for identifying orthogonal subspaces and performing OLS decoding across agent representations:

`Figure 3FGHI_Figure 4LMN_SubspaceOrthogonalization/MAIN_orthogonalize_subspaces_and_decode.m`

**Input:** `data/tuningCurves.mat`

**Output:** Variance explained within orthogonal subspaces and LOOCV R² values from decoding analyses.

**Required toolbox:** Manopt Version 8.0 (Boumal et al., 2014)

https://manopt.org

## Figure 5: CCGP Analysis

Inspired by Bernardi et al., 2020.

Code for performing CCGP analyses, including cross-validation and shuffled-control procedures:

`Figure 5BCE_CCGP/MAIN_CCGP_selfVSprey.m`

**Input:** `data/dataCCGP.mat` — chunked neural activity data.

**Output:** Cross-validated decoding accuracies and null distributions.

