# Case Study: Bugs Found and Fixed — Student Stress Level Prediction

This project went through a structured review process — every claim was checked
against actual code and data rather than taken at face value. Eight distinct bugs
were found and fixed along the way. Documented here in the order they were caught.

---

## Bug 1: Broken column-reorder loop (dead defensive code)

**Symptom:** `app.py` had a loop meant to fill in any missing expected columns
with 0 before reordering:

```python
for col in expected_columns:
    input_df = input_df[expected_columns]
    if col not in input_df.columns:
        input_df[col] = 0
```

**Root cause:** The reorder line (`input_df[expected_columns]`) ran *before* the
missing-column check, on every iteration. If any expected column was actually
missing, this raised `KeyError` immediately — the fallback logic below it could
never execute. The "defensive" code provided zero actual defense; it only
appeared to work because the hardcoded input dict already matched
`expected_columns` exactly.

**Fix:** Removed the loop entirely. Since the feature set is fixed and fully
known at request time, a single line is correct and honest about what it does:

```python
input_df = input_df[expected_columns]
```

**Lesson:** Code that "works" only because the unhappy path never gets
exercised isn't defensive — it's untested. Trace what a loop actually does on
its first iteration before trusting it.

---

## Bug 2: Student_Type encoding mismatch between training and serving

**Symptom:** `app.py` hardcoded an encoding for the `Student_Type` feature:

```python
STUDENT_TYPE_MAP = {"School": 0, "College": 1, "Working Student": 2}
```

**Root cause:** The actual training notebook used `sklearn.LabelEncoder`, which
assigns codes alphabetically: `college=0, school=1, working_student=2`. The
hand-copied dict in `app.py` swapped School and College relative to what the
model was actually trained on. This was silent — no error, no crash — every
prediction for a School or College user was computed with the wrong feature
value, indefinitely, and nothing in manual testing would have caught it because
the bug lived entirely in serving code, not in the model.

**Fix:** The fitted `LabelEncoder` object itself is now saved
(`studentType_encoder.pkl`) and loaded in `app.py`, so encoding is guaranteed
identical between training and inference — there is no second, independently
maintained copy of the mapping to drift out of sync.

**Lesson:** Never hand-transcribe a fitted transformer's behavior into a second
piece of code from memory. Persist and load the object itself.

---

## Bug 3: Test set scaled independently of the training set

**Symptom:** In the original notebook:

```python
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)   # fit_transform, not transform
```

**Root cause:** A second, separate `StandardScaler` was fit on the test set's
own mean/standard deviation instead of reusing the scaler fit on training data.
Reported test metrics therefore did not describe the actual deployed pipeline
(which correctly used the train-fitted scaler for the saved `.pkl`).

**Fix:** Changed to `scaler.transform(x_test)`. Verified by direct comparison
that the impact on this dataset was small (Accuracy 0.828 → 0.829, F1 0.688 →
0.687) — but the fix was made regardless, since the near-zero delta was
specific to this dataset and model, not a general property of the bug.

**Lesson:** A bug that happens not to move the numbers this time is still a
bug. Don't let a benign outcome discourage fixing the actual defect.

---

## Bug 4: Physically impossible values in the raw dataset

**Symptom:** `df.describe()` showed `Study_Hours` with a minimum of -1.43 and
`Attendance` with a range of -5.0 to 120.0.

**Root cause:** 21 rows had negative `Study_Hours` and 112 rows had
`Attendance` outside the 0–100% range — data entry errors, not legitimate
outliers. Neither the training notebook nor the Flask form's input validation
accounted for this.

**Fix:** Added an explicit filter after `dropna()`, applied to the full dataset
before the train/test split (safe because it encodes a fixed physical
constraint, not a statistic derived from the data):

```python
df = df[(df['Study_Hours'] >= 0) & (df['Attendance'].between(0, 100))]
```

This removed 88 rows (0.5% of the cleaned dataset) with no measurable shift in
class balance (70.1%/29.9% before and after).

**Lesson:** `.describe()` output should be read as a diagnostic, not treated as
the definition of a valid input range. Impossible values in training data can
also end up copy-pasted into UI validation bounds if you're not careful — check
both.

---

## Bug 5: Two parallel, inconsistent train/test pipelines (variable shadowing)

**Symptom:** The notebook briefly contained two separate blocks: a correct one
building lowercase `x_train`/`x_test` with train-only-fitted encoding, followed
immediately by a second block rebuilding a *different* split into capitalized
`X_train`/`X_test` from a stale, unencoded `df_encode` copy. All downstream
modeling used the second, broken block. The fitted `LabelEncoder` from the
first block was saved to disk despite never having touched the data the model
was actually trained on.

**Root cause:** Copy-pasting an updated fix into a notebook without deleting
the code it was meant to replace. Because the two blocks used different
variable names, Python raised no immediate error — the mistake only surfaced
downstream when scaling failed on a still-string-typed column.

**Fix:** Removed the leftover `df_encode` copy and the redundant first block.
Consolidated the entire pipeline (load → clean → split → encode → scale →
train → save) into a single, linear, one-pass flow with no duplicate variable
names.

**Lesson:** A notebook that reads correctly top-to-bottom on screen says
nothing about what actually executed, in what order, in the current kernel.
When in doubt, restart the kernel and run all cells in one pass before trusting
any output.

---

## Bug 6: Leakage inside cross-validation via pre-scaled input

**Symptom:** Model comparison code scaled `x_train`/`x_test` once, outside the
`GridSearchCV` loop, then passed the already-scaled arrays into
`GridSearchCV(cv=5, ...)`.

**Root cause:** The scaler's mean/standard deviation were computed from the
*entire* training set before any of the 5 internal CV folds were created. Each
fold's held-out validation data was therefore scaled using statistics that
partly derived from itself — a subtler version of the classic
fit-scaler-on-test-data leak, occurring one level deeper inside the CV loop.

**Fix:** Wrapped each model in a `sklearn.Pipeline` (`StandardScaler` + the
classifier) and passed raw, unscaled `x_train`/`x_test` into `GridSearchCV`.
Scaling is now refit independently within each fold.

**Lesson:** "I already scaled it, so I don't need to scale it again inside the
loop" is exactly backwards for cross-validation. Any preprocessing with fitted
state (scalers, encoders, imputers) belongs inside the pipeline that
`GridSearchCV` cross-validates, not applied once beforehand.

---

## Bug 7: Comparing tuned and untuned models as if they were equivalent

**Symptom:** An early comparison table ranked Logistic Regression and KNN
(default hyperparameters, single train/test split, no cross-validation)
alongside five other models that had gone through `GridSearchCV(cv=5)` with
proper pipelines.

**Root cause:** Two different evaluation methodologies were mixed into one
"comparison," making the ranking meaningless — a model could look worse simply
for not having been tuned, not because it was actually worse.

**Fix:** Re-ran every model (7 total) through the identical `Pipeline` +
`GridSearchCV(cv=5, scoring='f1')` process before ranking them.

**Lesson:** A comparison is only valid if every entry went through the same
process. "I already have a number for that model" isn't the same as "I have a
comparable number for that model."

---

## Bug 8: Reference to an unloaded object, masked by an overly broad `except`

**Symptom:** After fixing Bug 2, `app.py` correctly called
`student_type_encoder.transform(...)` but never actually loaded
`student_type_encoder` from disk. Every form submission failed with
`NameError`, but the user only ever saw a generic "Invalid input" message.

**Root cause:** A `try/except Exception` block around the prediction logic
swallowed the `NameError` and re-displayed it as if it were a user input
mistake, disguising a code bug as a data problem.

**Fix:** Added the missing `joblib.load("studentType_encoder.pkl")` call
alongside the other three artifact loads at startup.

**Lesson:** A catch-all `except Exception` is convenient for surfacing bad user
input, but it will just as happily hide a real programming error behind the
same friendly message. When something "isn't working," read the actual
exception text before assuming the user did something wrong — this project's
own broad except clause made that harder than it needed to be.

---

## Model selection note (not a bug, but worth recording)

Once all 7 models were compared fairly (Bug 7's fix), four of them — Random
Forest, SVM, Logistic Regression, and XGBoost — landed within 0.0023 F1 of each
other on cross-validated score, a gap smaller than run-to-run noise. Ranking by
single-split test F1 instead of CV score would have pointed to a different
"winner" depending on which single split was used.

**Logistic Regression was selected**, not because it scored highest (it
didn't, on either metric, by a meaningful margin), but because among
statistically indistinguishable options it is the cheapest to serve
(native, calibrated `predict_proba`), the simplest to maintain, and the only
one whose coefficients could be directly cross-checked against the EDA
correlations as an independent sanity check — which they were, and all eight
matched in sign.

One additional finding from that check: `Student_Type`'s coefficient in the
full model is effectively zero (-0.0037), despite a real spread in raw stress
rates by student type (school 20.5%, college 31.3%, working student 38.2%).
This indicates Student_Type is largely a proxy for other included features
(likely Exam_Pressure and Study_Hours) rather than an independent predictor —
structurally similar to the MonthlyIncome/JobLevel multicollinearity finding
in the earlier Attrition project.

---

## Final, honestly-reported model performance

Evaluated on a held-out 20% test split (3,345 rows), after all fixes above:

| Metric | Value |
|---|---|
| Accuracy | 0.7955 |
| F1 (High Stress) | 0.7067 |
| Recall (High Stress) | 0.8047 |
| Precision (High Stress) | 0.63 |
| Majority-class baseline accuracy | 0.7037 |

The model beats the majority-class baseline by roughly 9 points of accuracy,
and — more importantly, since accuracy alone is misleading on a 70/30 imbalanced
target — goes from F1 = 0 (baseline can't identify any high-stress case at all)
to F1 = 0.71.

**Known limitation, stated plainly rather than buried:** precision on the
"High Stress" class is 0.63 — roughly 37% of students flagged as high-stress
are false positives. This is an acceptable trade-off for an informational
signal a student can freely disregard, but would need to be disclosed clearly
if this prediction were ever used to trigger any actual intervention (e.g.
automatic counselor outreach).

**Known limitation on feature composition:** `Exam_Pressure` (a self-reported
1–10 rating) is by a wide margin the strongest single predictor (correlation
0.52, largest model coefficient). Because it's a subjective, adjacent measure
of stress rather than a purely behavioral lifestyle factor, this model is
better described as predicting stress from a mix of self-reported pressure and
lifestyle habits — not purely from lifestyle habits alone, as originally
framed.