# Confound checks

A signal is trustworthy if its AUROC stays high *within* length quintiles and *within* each domain, and clearly beats the length-only baseline.

| score | AUROC | direction | median human | median AI | min AUROC in length quintile | mean AUROC in length quintiles | min AUROC in a domain | Spearman with length |
|---|---|---|---|---|---|---|---|---|
| PHD | 0.795 | AI lower | 9.127 | 7.675 | 0.724 | 0.805 | 0.703 | -0.059 |
| MLE dim | 0.732 | AI lower | 6.737 | 5.605 | 0.605 | 0.699 | 0.593 | -0.446 |
| TwoNN dim | 0.71 | AI lower | 6.656 | 5.431 | 0.648 | 0.71 | 0.617 | -0.134 |
| H1 loop count | 0.889 | AI higher | 49.0 | 67.0 | 0.83 | 0.863 | 0.774 | 0.296 |
| H1 total persistence | 0.92 | AI higher | 1.421 | 2.351 | 0.865 | 0.899 | 0.791 | 0.308 |
| H0 persistence entropy | 0.626 | AI lower | 0.99 | 0.989 | 0.653 | 0.699 | 0.532 | 0.31 |
| log-likelihood | 0.992 | AI higher | -2.976 | -1.377 | 0.979 | 0.991 | 0.891 | 0.407 |
| Fast-DetectGPT | 0.925 | AI higher | -0.389 | 1.963 | 0.894 | 0.922 | 0.885 | 0.177 |
| Binoculars | 0.957 | AI lower | 1.017 | 0.837 | 0.949 | 0.961 | 0.944 | -0.165 |
| Length (tokens) | 0.648 | AI higher | 156.0 | 220.0 | 0.222 | 0.509 | 0.467 | 1.0 |
