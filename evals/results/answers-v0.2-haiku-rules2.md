# Answer eval - claude-haiku-5-5, effort low - 2026-10-28

| Measure | Result |
|---|---|
| Correct (expected fact present AND on topic) | 17/20 |
| Declined a question it should answer | 2/20 |
| Faithful (every claim supported, per judge) | 17/18 judged |
| On topic (per judge) | 17/18 judged |
| Declined when it should | 2/2 |
| Total cost | $0.0111 |
| Median latency | 3216 ms |

| Question | Style | Result | Missing fact | Faithful | On topic | Note |
|---|---|---|---|---|---|---|
| ma-recred-cycle | policy | answered ✓ | - | 1/1 | yes |  |
| ma-recred-cycle-plain | plain | declined ✗ | - | - | - | sources do not cover the question |
| medicaid-uniform-policy | policy | answered ✓ | - | 2/2 | yes |  |
| medicaid-nondiscrimination-plain | plain | answered ✓ | - | 2/2 | yes |  |
| medicaid-excluded-1128 | policy | answered ✓ | - | 1/1 | yes |  |
| ma-initial-cred-scope | policy | answered ✓ | - | 7/7 | yes |  |
| ma-provisional-plain | plain | answered ✓ | - | 7/7 | yes |  |
| ma-cred-exempt | policy | answered ✓ | - | 3/3 | yes |  |
| primary-source-definition | policy | answered ✓ | - | 5/5 | yes |  |
| ma-recred-board-cert-plain | plain | answered ✓ | - | 2/2 | yes |  |
| ma-non-psv-items | policy | answered ✗ | - | 4/4 | **no** | The answer names malpractice and DEA/CDS as the items, but the question asks which items must be verified without primary source verificatio |
| ma-termination-rights-plain | plain | answered ✓ | - | 7/7 | yes | The answer covers notice and appeal rights for dropping a physician, though it mixes in the separate declining-to-include rule and does not  |
| ma-oig-exclusion | policy | answered ✓ | - | 4/4 | yes |  |
| medicaid-revalidation | policy | answered ✓ | - | 4/4 | yes |  |
| medicaid-fingerprints-plain | plain | answered ✓ | - | 1/1 | yes |  |
| medicaid-120-day-agreement | policy | answered ✓ | - | 3/3 | yes |  |
| small-practice-burden-plain | plain | answered ✓ | - | 10/10 | yes |  |
| medicare-revalidation-cycle | policy | answered ✓ | - | 3/3 | yes |  |
| medicare-revalidation-dme-plain | plain | answered ✓ | - | 4/5 | yes | The answer is mostly supported, but its inference that the 5-year rule covers physicians goes beyond the sources, which never mention doctor |
| ma-consultation-plain | plain | declined ✗ | - | - | - | sources do not cover the question |
| decline-off-topic | decline | declined ✓ | - | - | - | off-topic: best match 0.47 is below the 0.60 cutoff |
| decline-determination | decline | declined ✓ | - | - | - | sources do not cover the question |

Faithful/on-topic come from a Haiku judge grading Haiku's answers, which tends to be lenient; read the unsupported claims, not just the counts.
