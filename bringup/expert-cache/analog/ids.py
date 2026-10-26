# ids.py: the 18k coding-session prompt (bringup/box/requests/cody-user.json) as the API renders and encodes it for
# GLM-5.3-Flash, one id a word, to data/analog/cody.ids (the analog dump's teacher-forced text).
import json, sys
from rockem.bundle import Bundle
from rockem.edge import Edge
b = Bundle("/home/slomp/Work/rockem/bundles/glm-5.3-flash-q4rtn")
m = b.model("glm-5.3-flash") if hasattr(b, "model") else b.models[0]
e = Edge(m)
d = json.load(open("/home/slomp/Work/rockem/bringup/box/requests/cody-user.json"))
ids = e.encode(e.render(d["messages"]))
open("/home/slomp/Work/rockem/bringup/expert-cache/data/analog/cody.ids", "w").write(" ".join(map(str, ids)))
print(len(ids), ids[:8])
