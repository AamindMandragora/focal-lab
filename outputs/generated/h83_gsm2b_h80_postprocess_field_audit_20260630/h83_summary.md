# H83 GSM-2B H80 postprocess field audit
Created: 2026-06-30T06:27:48.312427+00:00
This was CPU-only: **0** model calls, **0** GPU calls, **0** billed API calls, and no score artifact edits.
## Result
- Existing H80 structured artifact candidate count: **0** across group count **0**.
- Direct report has `full_output` field present: **True**; spans found in `full_output`: **94**.
- Scanning all text fields found **467** visible span records across **49** groups.
- Conservative machine-readable expression heuristic found **243** spans across **49** groups.
- Prediction confirmed: **False**.

## Field summary
- `full_output`: spans **94**, groups **26**, parseable-ish **75** across **21** groups.
- `helper_trace[101].detail`: spans **3**, groups **1**, parseable-ish **1** across **1** groups.
- `helper_trace[103].detail`: spans **6**, groups **2**, parseable-ish **2** across **2** groups.
- `helper_trace[106].detail`: spans **3**, groups **1**, parseable-ish **1** across **1** groups.
- `helper_trace[109].detail`: spans **3**, groups **1**, parseable-ish **1** across **1** groups.
- `helper_trace[111].detail`: spans **3**, groups **1**, parseable-ish **1** across **1** groups.
- `helper_trace[15].detail`: spans **54**, groups **18**, parseable-ish **18** across **18** groups.
- `helper_trace[164].detail`: spans **3**, groups **1**, parseable-ish **1** across **1** groups.
- `helper_trace[167].detail`: spans **3**, groups **1**, parseable-ish **1** across **1** groups.
- `helper_trace[206].detail`: spans **3**, groups **1**, parseable-ish **1** across **1** groups.
- `helper_trace[24].detail`: spans **3**, groups **1**, parseable-ish **1** across **1** groups.
- `helper_trace[262].detail`: spans **3**, groups **1**, parseable-ish **1** across **1** groups.
- `helper_trace[27].detail`: spans **3**, groups **1**, parseable-ish **1** across **1** groups.
- `helper_trace[298].detail`: spans **3**, groups **1**, parseable-ish **1** across **1** groups.
- `helper_trace[299].detail`: spans **3**, groups **1**, parseable-ish **1** across **1** groups.
- `helper_trace[303].detail`: spans **3**, groups **1**, parseable-ish **1** across **1** groups.
- `helper_trace[33].detail`: spans **3**, groups **1**, parseable-ish **1** across **1** groups.
- `helper_trace[494].detail`: spans **6**, groups **2**, parseable-ish **2** across **2** groups.
- `helper_trace[495].detail`: spans **3**, groups **1**, parseable-ish **1** across **1** groups.
- `helper_trace[584].detail`: spans **12**, groups **4**, parseable-ish **4** across **4** groups.
- `helper_trace[585].detail`: spans **3**, groups **1**, parseable-ish **1** across **1** groups.
- `helper_trace[587].detail`: spans **3**, groups **1**, parseable-ish **1** across **1** groups.
- `helper_trace[589].detail`: spans **3**, groups **1**, parseable-ish **1** across **1** groups.
- `scored_output`: spans **94**, groups **26**, parseable-ish **75** across **21** groups.
- `task_guidance`: spans **147**, groups **49**, parseable-ish **49** across **49** groups.

## Interpretation
Scanning all text fields recovered some candidates, but not enough to treat H80 as a clean candidate-pool success.

Next: Use field-aware postprocessing plus stricter bare-expression candidate generation; do not rerun the exact H80 body unchanged.

Credential key-name scan hits: **0**.
