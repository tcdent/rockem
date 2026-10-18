# 027 — Model-free drafts from the prompt's own n-grams (SuffixDecoding)

**Status:** Open
**Area:** decode
**Gain:** workload-dependent; large on agentic and code output (up to 5.3× reported)

## Proposal
Suffix trees over the prompt and earlier output propose drafts with no model; combined with the
MTP draft, verified the same way. Cost-truncate the draft where the marginal expert-union cost
exceeds the expected acceptance.

## Source
SuffixDecoding (NeurIPS 2025, arXiv 2411.04975); EVICT (arXiv 2605.00342) for cost truncation.
DFlash2 is non-commercial (CC BY-NC-ND) and out.

## History
- 2026-10-17: proposed.

## Next
Measure n-gram hit rates on recorded agentic sessions before building.
