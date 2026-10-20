# The one falsifiable claim

On MNIST with a 784-256-128-10 MLP, 20 epochs, Adam, the same cubic schedule and the same
masking code for every method, **at 90% sparsity magnitude pruning keeps 98.27% test accuracy
(std 0.09) and my second-order Taylor criterion keeps 98.01% (std 0.17), averaged over seeds
0-4. The paired difference is -0.27 points, bootstrap 95% CI [-0.33, -0.20], and magnitude wins
on 5 of 5 seeds.** Adding regrowth to my criterion closes most of the gap (98.18%, difference
-0.10 [-0.16, -0.03]) but does not flip it.

So the importance criterion I derived does not beat magnitude here. Raw numbers are in
`pareto.csv`, the statistics in `baseline_stats.csv`.

The claim is wrong if rerunning `python -m experiments.part4_sweep` gives a 90% paired CI that
contains 0 or sits above 0. With 5 seeds the smallest possible exact permutation p-value is
2/32 = 0.0625, so I lean on the paired CI and the 5/5 win count rather than on p.
