# tl_agg.awk: the timeline lines of a run summed per (card, tokens a step, kernel): µs a launch, launches a step
/timeline : card/ {
  i = index($0, "card "); j = index($0, " tokens"); key = substr($0, i, j - i + 7)
  steps[key]++
  rest = substr($0, index($0, "count): ") + 8)
  n = split(rest, a, ", ")
  for (k = 1; k <= n; k++) { split(a[k], b, " "); split(b[2], c, "/"); us[key "|" b[1]] += c[1]; cnt[key "|" b[1]] += c[2] }
}
END {
  for (k in us) { split(k, p, "|"); printf "%s %-16s %7.1f us/launch  %5.1f launches/step  %7.0f us/step\n", p[1], p[2], us[k] / cnt[k], cnt[k] / steps[p[1]], us[k] / steps[p[1]] }
}
