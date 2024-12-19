# Clustering Sales Data

**requirements**

our code is written in python3 and for ease of viewing has been placed in jupyter notebooks.

the following libraries are needed to run our code (as defined at the top of each notebook):

- numpy
- matplotlib
- pandas
- sklearn (not used for clustering)
- scipy
- itertools

The data we used has been placed in the data folder allongside a link to the source and a discription of the columns.

**How To Run**

Our main presentation of our project occurs in the jupyter notebooks. These should be run using python3 (we used python3.12). The notebooks are numbered roughly chronologically with a number before their name.

We have also left all our old python3 files in the old_files folder (including old versions of our project plan). These are included for completenes and can still be run. They dont however contain a structured narative. These files should be run from the root folder (ie the one this README is in) using python3 old_files/filename.py

**Synopsis**

This project explores computational methods to address real-world shopping analysis challenges.  

Key objectives include:  
- Using polynomial and Fourier fitting to identify trends in noisy data;
- Applying k-means clustering to group shopping patterns;
- Evaluating the quality of both fitting and clustering outcomes. 

**Hypothesis**

Ha (goodness-of-fit): Test if the residuals from the polynomial fit and Fourier fit follow a normal distribution.

H0_a1: The residuals from the polynomial fit follow a standard normal distribution.
H0_a2: The residuals from the Fourier fit it follow a standard normal distribution.

Hb (compare between models)

H0_b: There is no significant difference in the clustering performance of the polynomial fit and Fourier Fit.


**methods**

These are explained in the jupyter notebooks.

The suggested order is: 0_exploration.ipynb, 1_polynominal_fit.ipynb, 2_fourier_fit.ipynb, 3_conclusion.ipynb

**Conclusion**

In the file 3_conclusion.ipynb we see that the test suggest that we do not have
enough evidence to accept our hypotheses. We see that both the polynominal fit
and the fourier fit do not agree. This means that one of both method(s) are wrong.
