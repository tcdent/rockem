#!/usr/bin/env python3
"""`bundle_relayout.py <bundle dir>`: a bundle written before wire v5 (one `[model]` table, files at the root) into
the v5 layout the builders write now (docs/bundle.md): a `[[model]]` entry carrying the recipe and geometry, the
model's files under a directory named for it, `[files]` keyed `<name>/<file>`. A rename within one filesystem, so
the weights do not move. Idempotent: a bundle already in the layout is left alone."""
import os, re, sys
root = sys.argv[1]
text = open(os.path.join(root, "manifest.toml")).read()
if "[[model]]" in text:
    print(f"{root}: already a model list"); sys.exit(0)
head, rest = text.split("[model]", 1)
model, rest = rest.split("[files]", 1)
files, comment = rest.split("# checkpoint.toml", 1) if "# checkpoint.toml" in rest else (rest, "")
name = re.search(r'name\s*=\s*"([^"]+)"', model).group(1)
top = [l for l in head.strip().splitlines() if l.strip()]
entry = [l for l in model.strip().splitlines() if l.strip()]
# the entry: name first, then the recipe and geometry that were top-level, then the wire's constants
entry = [l for l in entry if l.startswith("name")] + top + [l for l in entry if not l.startswith("name")]
keyed = []
for l in files.strip().splitlines():
    m = re.match(r'"([^"]+)"\s*=\s*(.*)', l.strip())
    if not m:
        continue
    f, sha = m.groups()
    src, dst = os.path.join(root, f), os.path.join(root, name, f)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(src):
        os.rename(src, dst)
    keyed.append(f'"{name}/{f}" = {sha}')
out = "[[model]]\n" + "\n".join(entry) + "\n\n[files]\n" + "\n".join(keyed) + "\n"
if comment:
    out += "\n# checkpoint.toml" + comment
open(os.path.join(root, "manifest.toml"), "w").write(out)
print(f"{root}: {name}, {len(keyed)} files under {name}/")
