# One semantic target. The build pipeline reads exactly these three lines.
# The build fails if no acceptance threshold meets the contract at the minimum coverage.
Semantic Target: extract latency + user id
Fallback: legacy parser
Contract: <=1% wrong on accepted inputs; confidence 95%; min coverage 50%
