# ids_doc.py: the second text of the analog study — ~/Work/localai/model-architecture-survey.md (technical prose and
# tables, not a code session) as one user message, rendered and encoded by the API's edge, to data/analog/doc.ids.
from rockem.bundle import Bundle
from rockem.edge import Edge
b = Bundle("/home/slomp/Work/rockem/bundles/glm-5.3-flash-q4rtn")
e = Edge(b.model("glm-5.3-flash") if hasattr(b, "model") else b.models[0])
text = open("/home/slomp/Work/localai/model-architecture-survey.md").read()
ids = e.encode(e.render([{"role": "user", "content": "Here is a survey document. Read it and summarize its conclusions.\n\n" + text}]))
open("/home/slomp/Work/rockem/bringup/expert-cache/data/analog/doc.ids", "w").write(" ".join(map(str, ids)))
print(len(ids))
