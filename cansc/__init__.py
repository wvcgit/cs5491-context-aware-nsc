"""Context-aware NSC (CA-NSC) for the CS5491 project.

The package implements one change to Neural Spectral Capacity (NSC): the FFN
term of layer l is multiplied by w_l = 1 / (1 + alpha * C_l), where C_l is the
raw FFN capacity of the preceding layers divided by the hidden width.

Modules
-------
context_nsc     the scoring change (pure functions over per-layer NSC terms)
records         the archive record type used by the evaluation
stats           rank statistics: average ranks, Spearman, Kendall tau-b
pairs           same-width, near-equal-size pair comparisons (size control)
protocol        the pre-registered alpha grid rule and selection rule
evaluate        development-split evaluation pipeline
checks          randomized property checks of the scoring change
toy             the hand-computable FFN reordering example
course_adapter  binding to the course NSC starter (must be bound to run on data)
report_tex      writes the numbers used by the LaTeX report
"""

__version__ = "0.1.0-milestone"
