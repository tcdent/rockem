#!/usr/bin/env python3
"""`lab_bundle.py <production bundle> <staging bundle> <lab dir>`: a bundle directory for a lab boot that lists the
production bundle's models and the staging bundle's one — the model directories symlinked (the weights do not move), one
manifest with every entry, every [files] line and every checkpoint comment. Production is untouched; rockem.bundle=
/bundle/<lab dir> on a lab cmdline boots it (bringup/box/lab_boot.sh)."""
import os, sys

def parts(text):
    """(models text, [files lines], [checkpoint comment blocks]) of a manifest — bundle_merge.py's reading."""
    models, rest = text.split("[files]", 1)
    pieces = rest.split("# checkpoint.toml")
    files = [l for l in pieces[0].strip().splitlines() if l.strip()]
    comments = ["# checkpoint.toml" + p.rstrip("\n") for p in pieces[1:]]
    return models.rstrip(), files, comments

def write(path, models, files, comments):
    open(path, "w").write(models.rstrip() + "\n\n[files]\n" + "\n".join(files) + "\n\n" + "\n\n".join(comments) + "\n")

prod, staging, lab = sys.argv[1:4]
os.makedirs(lab, exist_ok=True)
m1, f1, c1 = parts(open(os.path.join(prod, "manifest.toml")).read())
m2, f2, c2 = parts(open(os.path.join(staging, "manifest.toml")).read())
for src in (prod, staging):
    for d in sorted(os.listdir(src)):
        if os.path.isdir(os.path.join(src, d)):
            link = os.path.join(lab, d)
            if os.path.lexists(link):
                os.remove(link)
            os.symlink(os.path.relpath(os.path.join(src, d), lab), link)
            print(f"{link} -> {os.readlink(link)}")
write(os.path.join(lab, "manifest.toml"), m1.rstrip() + "\n\n" + m2.strip(), f1 + f2, c1 + c2)
open(os.path.join(lab, "CACHEDIR.TAG"), "w").write("Signature: 8a477f597d28d172789f06886806bc55\n")
print(os.path.join(lab, "manifest.toml"))
