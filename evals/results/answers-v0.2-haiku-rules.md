# Answer eval - claude-haiku-5-5, effort low - 2026-10-28

| Measure | Result |
|---|---|
| Correct (expected fact present) | 18/20 |
| Declined a question it should answer | 0/20 |
| Faithful (every claim supported, per judge) | 19/20 judged |
| On topic (per judge) | 17/20 judged |
| Declined when it should | 2/2 |
| Total cost | $0.0124 |
| Median latency | 3418 ms |

| Question | Style | Result | Missing fact | Faithful | On topic | Note |
|---|---|---|---|---|---|---|
| ma-recred-cycle | policy | answered ✓ | - | 3/3 | yes | The answer correctly states the at-least-every-3-years recredentialing requirement and its claims are supported by the sources, though it is |
| ma-recred-cycle-plain | plain | answered ✓ | - | 3/3 | **no** | The question asks about doctors in a Medicare Advantage plan's network, but the answer addresses general Medicare fee-for-service enrollment |
| medicaid-uniform-policy | policy | answered ✓ | - | 5/5 | yes |  |
| medicaid-nondiscrimination-plain | plain | answered ✓ | - | 4/4 | yes |  |
| medicaid-excluded-1128 | policy | answered ✓ | - | 2/2 | yes |  |
| ma-initial-cred-scope | policy | answered ✓ | - | 8/8 | yes |  |
| ma-provisional-plain | plain | answered ✓ | - | 5/5 | yes |  |
| ma-cred-exempt | policy | answered ✓ | - | 3/3 | yes |  |
| primary-source-definition | policy | answered ✓ | - | 7/8 | yes | The answer is mostly supported, but its claim that NPDB is 'not a primary source in the strict sense' is an inference the sources do not sta |
| ma-recred-board-cert-plain | plain | answered ✓ | - | 3/3 | yes |  |
| ma-non-psv-items | policy | answered ✓ | - | 6/6 | **no** | The answer names only malpractice insurance and DEA/CDS, but the sources list the secondary-source items as the 'other credentialing require |
| ma-termination-rights-plain | plain | answered ✓ | - | 7/7 | yes |  |
| ma-oig-exclusion | policy | answered ✓ | - | 7/7 | yes |  |
| medicaid-revalidation | policy | answered ✓ | - | 2/2 | yes |  |
| medicaid-fingerprints-plain | plain | answered ✓ | - | 1/1 | yes |  |
| medicaid-120-day-agreement | policy | answered ✓ | - | 3/3 | yes |  |
| small-practice-burden-plain | plain | answered ✗ | burden | 7/7 | yes |  |
| medicare-revalidation-cycle | policy | answered ✓ | - | 3/3 | yes |  |
| medicare-revalidation-dme-plain | plain | answered ✓ | - | 5/5 | yes |  |
| ma-consultation-plain | plain | answered ✗ | consult | 6/6 | **no** | The answer says the sources do not address whether plans must seek physician input, and instead describes notice and appeal requirements, wh |
| decline-off-topic | decline | declined ✓ | - | - | - | off-topic: best match 0.47 is below the 0.60 cutoff |
| decline-determination | decline | declined ✓ | - | - | - | sources do not cover the question |

Faithful/on-topic come from a Haiku judge grading Haiku's answers, which tends to be lenient; read the unsupported claims, not just the counts.
