"""Copy a GGUF, adding/overriding bool metadata keys:  python gguf_add_bool.py in.gguf out.gguf key=true [key=false ...]
Lets an existing mmproj opt in to the Ossicle audio keys (or any other bool key) without reconverting."""
import sys
sys.path.insert(0, "/mnt/data/projects/llama.cpp/gguf-py")
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType

src, dst, *kv = sys.argv[1:]
add = {k: v.lower() == "true" for k, v in (x.split("=") for x in kv)}
r = GGUFReader(src)
arch = r.fields["general.architecture"].contents()
w = GGUFWriter(dst, arch)
for name, f in r.fields.items():
    if name.startswith("GGUF.") or name == "general.architecture" or name in add:
        continue
    t = f.types[0]
    if t == GGUFValueType.ARRAY:
        w.add_array(name, f.contents())
    elif t == GGUFValueType.STRING:
        w.add_string(name, f.contents())
    elif t == GGUFValueType.BOOL:
        w.add_bool(name, bool(f.contents()))
    else:
        w.add_key_value(name, f.contents(), t)
for k, v in add.items():
    w.add_bool(k, v)
for t in r.tensors:
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in r.tensors:
    w.write_tensor_data(t.data)
w.close()
print("wrote", dst, "added", add)
