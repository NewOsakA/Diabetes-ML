# What can we conclude, and what remains open?

**The project demonstrates measurable tradeoffs in imbalanced classification. It does not establish a reliable prediabetes detector or a clinically validated tool.**

## What the evidence supports

- Experiment 1's weighted MLP improved macro F1 and diabetes recall on the same held-out rows, at a cost to accuracy and false positives.
- Experiment 2's broader search produced only modest additional classification gains over the fixed weighted recipe under nested evaluation.
- Independent calibration improved average probability-based scores in paired checks.
- Extra complexity, synthetic sampling and a recall-focused decision objective did not automatically improve the desired outcomes.

## What the evidence does not establish

| Limitation | Why it matters |
|---|---|
| Prediabetes precision and recall are low | A model can improve a global score while remaining weak for a specific class |
| Survey status is self-reported | Labels reflect the source survey; we do not have laboratory confirmation in this file |
| Data are from 2015 US survey responses | Results may not transfer to another time, country or data-collection process |
| Some features are coarse categories | Different people and statuses can share the same recorded answers |
| No person identifiers | We cannot prove person-level independence from feature profiles alone |
| No survey-design weights | The processed sample is not a survey-weighted national estimate |
| Limited evaluation repetitions | Three outer folds and three neural seeds do not cover all sources of uncertainty |
| Prior exploration of this dataset | Development-stage evaluation is not a substitute for a genuinely independent external test |

Class weights and tuned offsets change what mistakes the model makes. They do not manufacture missing information.

## Useful next research steps

1. **Validate on an independently collected dataset.** This is the strongest next check of whether the findings transfer.
2. **Define the intended decision and acceptable tradeoff before optimizing.** More recall with many false positives may not serve that decision.
3. **Investigate label and feature ambiguity.** Examine which classes share similar recorded profiles, without automatically declaring repeated answers erroneous.
4. **Consider richer, independently collected inputs where appropriate.** The present file lacks measurements such as glucose or HbA1c; we cannot attach them accurately to these rows without a valid linked source.
5. **Expand uncertainty and subgroup analysis.** Include more predeclared repetitions and appropriate confidence estimates, with adequate subgroup sample sizes.
6. **Run matched training ablations.** Change one component at a time when the goal is to attribute a gain specifically to weighting, sampling or regularization.

Keep future experiments separate from the historical test comparison. Changing the random split does not create a genuinely new external dataset.

## How to describe the project accurately

> We compared models for three-class diabetes status classification using BRFSS health indicators. Imbalance-aware training improved some detection metrics, while nested development evaluation showed modest gains from broader tuning and persistent difficulty identifying prediabetes.

Avoid describing the model as a diabetes diagnosis system, a forecast of future illness, or “highly accurate” based only on its majority-dominated accuracy.

**Next:** [FAQ and glossary →](FAQ-and-Glossary)
