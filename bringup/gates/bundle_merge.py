#!/usr/bin/env python3
"""`bundle_merge.py <bundle dir> <staging bundle dir>`: move the staging bundle's one model into the bundle — its
directory, its `[[model]]` entry (with `[model.audio]` if it has one), its `[files]` lines and its checkpoint comment —
so the bundle lists both (docs/bundle.md). The reverse is `--split <bundle dir> <model name> <staging dir>`: the model
back out into a staging bundle of its own. A rename within one filesystem: the weights do not move."""
import os, shutil, sys

def parts(text):
    """(models text, [files lines], [checkpoint comment blocks]) of a manifest."""
    models, rest = text.split("[files]", 1)
    pieces = rest.split("# checkpoint.toml")
    files = [l for l in pieces[0].strip().splitlines() if l.strip()]
    comments = ["# checkpoint.toml" + p.rstrip("\n") for p in pieces[1:]]
    return models.rstrip(), files, comments

def write(path, models, files, comments):
    open(path, "w").write(models.rstrip() + "\n\n[files]\n" + "\n".join(files) + "\n\n" + "\n\n".join(comments) + "\n")

if sys.argv[1] == "--split":
    bundle, name, staging = sys.argv[2], sys.argv[3], sys.argv[4]
    models, files, comments = parts(open(os.path.join(bundle, "manifest.toml")).read())
    entries = ["[[model]]" + e for e in models.split("[[model]]") if e.strip()]
    mine = [e for e in entries if f'name = "{name}"' in e]
    assert len(mine) == 1, f"{name}: not in the bundle"
    keep = [e for e in entries if e not in mine]
    my_files = [l for l in files if l.startswith(f'"{name}/')]
    my_comments = [c for c in comments if f"({name})" in c]
    os.makedirs(staging, exist_ok=True)
    shutil.move(os.path.join(bundle, name), os.path.join(staging, name))
    write(os.path.join(staging, "manifest.toml"), "\n\n".join(mine), my_files, [c.replace(f" ({name})", "", 1) for c in my_comments])
    write(os.path.join(bundle, "manifest.toml"), "\n\n".join(keep), [l for l in files if l not in my_files], [c for c in comments if c not in my_comments])
    print(f"{name} out of {bundle} into {staging}")
else:
    bundle, staging = sys.argv[1], sys.argv[2]
    models, files, comments = parts(open(os.path.join(bundle, "manifest.toml")).read())
    s_models, s_files, s_comments = parts(open(os.path.join(staging, "manifest.toml")).read())
    name = s_models.split('name = "', 1)[1].split('"', 1)[0]
    assert f'name = "{name}"' not in models, f"{name}: already in {bundle}"
    shutil.move(os.path.join(staging, name), os.path.join(bundle, name))
    comments += [c.replace("# checkpoint.toml", f"# checkpoint.toml ({name})", 1) for c in s_comments]
    write(os.path.join(bundle, "manifest.toml"), models + "\n\n" + s_models.strip(), files + s_files, comments)
    os.remove(os.path.join(staging, "manifest.toml"))
    if not os.listdir(staging):
        os.rmdir(staging)
    print(f"{name} merged into {bundle}")
