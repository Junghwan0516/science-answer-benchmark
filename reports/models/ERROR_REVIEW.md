# Error review: when the model predicts correct

The five records below are the first false-correct contradiction from each of the first five affected development questions in ID order. This is a deterministic case review, not a representative sample of all errors. The original labels are preserved. The scores are from the frozen, temperature-scaled model.

| Question | Model score for correct | What the text shows | Working diagnosis |
| --- | ---: | --- | --- |
| EM_46 | 44.9% | The answer identifies the square salt crystals and their markings, but fails to state the needle shape and appears to deny that the calcite formed shapes. | A correct detail can coexist with an opposing or unclear claim. |
| LP_15c | 39.5% | Under the same load setup, “farther takes less effort” and “closer takes more effort” express the same relationship. | Possible annotation inconsistency; mark for review rather than relabel. |
| ME_17b | 57.3% | The reference requires a closed circuit in the new switch position. The answer says the circuit would open. | Opposite circuit state despite shared terminology. |
| ME_66b | 48.1% | The reference says aluminum itself conducts. The answer adds a false requirement for iron. | Incorrect necessary condition mixed with an initial affirmative. |
| ME_78b | 41.0% | The reference says the motor keeps running via a separate path. The answer asserts a series arrangement and predicts that the motor stops. | Incorrect circuit topology and opposite outcome. |

The examples motivate a controlled comparison using original statements, role reversals, state reversals, and equivalent paraphrases. Treat these five development cases as exploratory; measure new features on independently prepared contrast pairs before deciding whether they improve robustness.

[Machine-generated case list](development_error_examples.csv) · [Confusion matrices and exposure audit](error_audit.json)
