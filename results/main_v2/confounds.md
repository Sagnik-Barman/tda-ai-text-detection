# Confound checks

A signal is trustworthy if its AUROC stays high *within* length quintiles and *within* each domain, and clearly beats the length-only baseline.

| score | AUROC | direction | median human | median AI | min AUROC in length quintile | mean AUROC in length quintiles | min AUROC in a domain | Spearman with length |
|---|---|---|---|---|---|---|---|---|
| PHD | 0.68 | AI lower | 9.127 | 8.194 | 0.57 | 0.682 | 0.547 | -0.075 |
| MLE dim | 0.681 | AI lower | 6.737 | 5.872 | 0.631 | 0.672 | 0.609 | -0.498 |
| TwoNN dim | 0.589 | AI lower | 6.656 | 6.043 | 0.482 | 0.586 | 0.397 | -0.185 |
| H1 loop count | 0.844 | AI higher | 49.0 | 65.0 | 0.81 | 0.844 | 0.71 | 0.275 |
| H1 total persistence | 0.871 | AI higher | 1.421 | 2.199 | 0.834 | 0.868 | 0.72 | 0.292 |
| H0 persistence entropy | 0.596 | AI lower | 0.99 | 0.989 | 0.517 | 0.627 | 0.486 | 0.322 |
| log-likelihood | 0.956 | AI higher | -2.976 | -1.562 | 0.928 | 0.959 | 0.781 | 0.371 |
| Fast-DetectGPT | 0.876 | AI higher | -0.389 | 1.577 | 0.808 | 0.871 | 0.793 | 0.173 |
| Binoculars | 0.918 | AI lower | 1.017 | 0.862 | 0.861 | 0.917 | 0.859 | -0.116 |
| Length (tokens) | 0.578 | AI higher | 156.0 | 197.0 | 0.234 | 0.488 | 0.453 | 1.0 |
