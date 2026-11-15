# HireNex Evaluation Benchmark

## Files

- `ground_truth.json` — 90 synthetic CSE/ECE/MECH candidate records
- `recruiter_queries.json` — 12 recruiter queries with expected application IDs
- `latest_evaluation.json` — written after each run (ignored by Git)

## How to run

1. Upload the matching benchmark PDFs (application IDs must match `ground_truth.json`).
2. Convert to Markdown; run Traditional NLP and/or LLM parsing.
3. **Evaluation** tab → **Run evaluation**. Export CSV or JSON if needed.

## Metric definitions

**Parsing (per parser)**
- **Records parsed:** share of benchmark resumes that produced a record.
- **Name / email / phone accuracy:** exact match after normalisation.
- **Skills / certifications P / R / F1:** micro-averaged over all resumes.
- **Experience accuracy:** prediction within 0.25 years of the label. Years are computed from the parsed experience dates exactly as retrieval does (overlaps merged); a stated "N years" in the summary is the fallback.
- **CGPA accuracy:** prediction within 0.1 of the label. The degree-level CGPA is used, not 10th/12th grades.
- **MAE / RMSE:** error size, only on records where both values exist. Missing predictions count against accuracy but are not imputed as errors.
- **Overall score:** plain mean of skills F1, certification F1, average contact accuracy, experience accuracy and CGPA accuracy. Custom — label it as such in reports.

**Retrieval (mean over queries that ran)**
- **Precision / recall / F1:** over the full returned list vs. the expected set. Fixed-cutoff P@10 / R@10 are not used: with 4–30 relevant candidates per query they cap a perfect system at 40% / 33%.
- **MRR:** reciprocal rank of the first relevant candidate.
- **nDCG@10:** ranking quality of the top 10 (binary relevance).
- **Latency:** seconds per query for query understanding + search (answer text is not generated during evaluation).

**Conversion**
- Success rate, converted / total, average seconds per resume.

## Benchmark corrections in v5

- *"CGPA above 8.5"* expected 31 candidates, including one with CGPA exactly 8.5. "Above" is strictly greater, so APP000028 was removed (30 expected).

The benchmark is synthetic; for formal reporting use a reviewed, representative labeled dataset with a held-out test set.
