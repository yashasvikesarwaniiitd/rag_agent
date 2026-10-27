# Answer eval - claude-haiku-5-5, effort low - 2026-10-28

| Measure | Result |
|---|---|
| Correct (expected fact present) | 17/20 |
| Declined a question it should answer | 1/20 |
| Faithful (every claim supported, per judge) | 17/19 judged |
| On topic (per judge) | 19/19 judged |
| Declined when it should | 2/2 |
| Total cost | $0.0126 |
| Median latency | 3808 ms |

| Question | Style | Result | Missing fact | Faithful | On topic | Note |
|---|---|---|---|---|---|---|
| ma-recred-cycle | policy | answered ✓ | - | 5/5 | yes |  |
| ma-recred-cycle-plain | plain | declined ✗ | - | - | - | sources do not cover the question |
| medicaid-uniform-policy | policy | answered ✓ | - | 5/5 | yes |  |
| medicaid-nondiscrimination-plain | plain | answered ✓ | - | 4/4 | yes |  |
| medicaid-excluded-1128 | policy | answered ✓ | - | 1/3 | yes | The core answer is supported, but the related-requirement claim misstates the source as 'title V' rather than 'this title' and adds an unsup |
| ma-initial-cred-scope | policy | answered ✓ | - | 10/10 | yes |  |
| ma-provisional-plain | plain | answered ✓ | - | 7/7 | yes |  |
| ma-cred-exempt | policy | answered ✓ | - | 3/3 | yes |  |
| primary-source-definition | policy | answered ✓ | - | 5/5 | yes |  |
| ma-recred-board-cert-plain | plain | answered ✓ | - | 2/2 | yes |  |
| ma-non-psv-items | policy | answered ✓ | - | 6/6 | yes | The answer is largely supported and addresses the question, though it mislabels the secondary-source items as a clear list when sources only |
| ma-termination-rights-plain | plain | answered ✓ | - | 10/10 | yes |  |
| ma-oig-exclusion | policy | answered ✗ | Inspector General | 4/4 | yes |  |
| medicaid-revalidation | policy | answered ✓ | - | 4/4 | yes |  |
| medicaid-fingerprints-plain | plain | answered ✓ | - | 1/1 | yes |  |
| medicaid-120-day-agreement | policy | answered ✓ | - | 2/2 | yes |  |
| small-practice-burden-plain | plain | answered ✓ | - | 9/9 | yes | The answer is well supported and addresses the question, though it asserts the 'most acute' point for CHWs by analogy rather than directly. |
| medicare-revalidation-cycle | policy | answered ✓ | - | 8/8 | yes | The answer is accurate and addresses both revalidation frequencies, though citing source 2 and 5 for the 5-year rule is fine and the accredi |
| medicare-revalidation-dme-plain | plain | answered ✓ | - | 3/4 | yes | The final claim that physicians fall under 'all other providers' is an inference not stated in the sources, though the core answer is suppor |
| ma-consultation-plain | plain | answered ✗ | consult | 3/3 | yes |  |
| decline-off-topic | decline | declined ✓ | - | - | - | off-topic: best match 0.47 is below the 0.60 cutoff |
| decline-determination | decline | declined ✓ | - | - | - | sources do not cover the question |

Faithful/on-topic come from a Haiku judge grading Haiku's answers, which tends to be lenient; read the unsupported claims, not just the counts.
