# Cross-check with the unmodified course evaluator

`CANSC_ALPHA=0.7 python3 starter/evaluate.py --split development --student student_ca_nsc.py` (score = CA-NSC at the selected alpha, ablation = alpha 0).

- |original_nsc_rho|: 0.0e+00
- |ablation_rho_vs_course_row|: 0.0e+00
- |ablation_rho_vs_alpha0_row|: 0.0e+00
- |student_rho_vs_selected_row|: 0.0e+00
- |student_tau_vs_selected_row|: 0.0e+00
- size-controlled pairs: course evaluator 180, ours 180, starter README 180
- student size-controlled counts equal ours: {'concordant': True, 'discordant': True, 'score_only_ties': True, 'quality_only_ties': True, 'both_ties': True}
- original-NSC ranking block identical to expected_development.json: True (reference rho 0.969745)

course_evaluator_development.json also holds the evaluator's budgeted-selection output; it is not analysed in the milestone.
