## Robustness summary (AUROC)

| detector | in_domain | loo_dom | mean_OOD | gap | TPR@1%FPR (in-domain) |
|---|---|---|---|---|---|
| Hybrid topo+likelihood [GBM] | 1.0 | 0.998 | 0.998 | 0.001 | 0.988 |
| Likelihood feats [LR] | 0.995 | 0.997 | 0.997 | -0.002 | 0.936 |
| log-rank | 0.993 | 0.997 | 0.997 | -0.004 | 0.905 |
| log-likelihood | 0.992 | 0.996 | 0.996 | -0.005 | 0.905 |
| RoBERTa fine-tuned (supervised) | 0.999 | 0.995 | 0.995 | 0.004 | 0.997 |
| Attention topology [LR] | 0.999 | 0.995 | 0.995 | 0.005 | 0.989 |
| All topology [GBM] | 0.997 | 0.989 | 0.989 | 0.009 | 0.97 |
| Embedding topology [LR] | 0.959 | 0.959 | 0.959 | 0.001 | 0.524 |
| Binoculars | 0.957 | 0.958 | 0.958 | -0.001 | 0.626 |
| Fast-DetectGPT | 0.922 | 0.922 | 0.922 | 0.001 | 0.542 |
| H1 total persistence (zero-shot) | 0.913 | 0.916 | 0.916 | -0.003 | 0.263 |
| H1 loop count (zero-shot) | 0.881 | 0.887 | 0.887 | -0.006 | 0.154 |
| PHD (zero-shot) | 0.812 | 0.814 | 0.814 | -0.002 | 0.045 |
| MLE dim (zero-shot) | 0.749 | 0.734 | 0.734 | 0.015 | 0.025 |
| Length only (sanity baseline) | 0.664 | 0.661 | 0.661 | 0.003 | 0.0 |

## AUROC by split

| detector | in_domain | loo_dom:finance | loo_dom:medicine | loo_dom:reddit_eli5 | loo_dom:wiki_csai |
|---|---|---|---|---|---|
| All topology [GBM] | 0.997 | 0.998 | 0.998 | 0.995 | 0.964 |
| Attention topology [LR] | 0.999 | 0.997 | 1.0 | 0.994 | 0.987 |
| Binoculars | 0.957 | 0.954 | 0.939 | 0.972 | 0.965 |
| Embedding topology [LR] | 0.959 | 0.97 | 0.987 | 0.951 | 0.928 |
| Fast-DetectGPT | 0.922 | 0.931 | 0.878 | 0.946 | 0.932 |
| H1 loop count (zero-shot) | 0.881 | 0.906 | 0.944 | 0.863 | 0.836 |
| H1 total persistence (zero-shot) | 0.913 | 0.927 | 0.969 | 0.89 | 0.879 |
| Hybrid topo+likelihood [GBM] | 1.0 | 1.0 | 1.0 | 0.997 | 0.997 |
| Length only (sanity baseline) | 0.664 | 0.611 | 0.911 | 0.641 | 0.482 |
| Likelihood feats [LR] | 0.995 | 0.999 | 0.999 | 0.994 | 0.997 |
| MLE dim (zero-shot) | 0.749 | 0.567 | 0.863 | 0.788 | 0.719 |
| PHD (zero-shot) | 0.812 | 0.862 | 0.846 | 0.778 | 0.77 |
| RoBERTa fine-tuned (supervised) | 0.999 | 0.995 | 1.0 | 0.997 | 0.987 |
| log-likelihood | 0.992 | 0.999 | 1.0 | 0.993 | 0.994 |
| log-rank | 0.993 | 0.999 | 1.0 | 0.993 | 0.995 |
