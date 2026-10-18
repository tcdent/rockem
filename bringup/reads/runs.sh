#!/usr/bin/env bash
# The newest prow runs of the node: `runs.sh [n]` (default 10). Which build, which box, which cmdline.
set -u; . "$(dirname "$0")/_run.sh"
q "SELECT substr(run, 1, 8) AS run, strftime(started::TIMESTAMPTZ, '%m-%d %H:%M') AS started, host, state,
          \"build.commit\" AS commit, \"build.dirty\" AS dirty, image, ctx,
          regexp_extract(cmdline, 'rockem\.resident=[0-9]+') AS resident,
          regexp_extract(cmdline, 'rockem\.bundle=[^ ]+') AS bundle
   FROM runs ORDER BY started DESC LIMIT ${1:-10}"
