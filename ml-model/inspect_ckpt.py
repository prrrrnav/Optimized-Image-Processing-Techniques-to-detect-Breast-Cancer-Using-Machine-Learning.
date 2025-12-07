# inspect_ckpt.py
import torch, json
ckpt_path = "model.pth"   # adjust if your file is elsewhere
ckpt = torch.load(ckpt_path, map_location="cpu")
print("TYPE:", type(ckpt))
if isinstance(ckpt, dict):
    print("keys:", list(ckpt.keys()))
    for k in ckpt.keys():
        if hasattr(ckpt[k], "keys"):
            print(f"--- {k} keys: {list(ckpt[k].keys())[:20]}")
        else:
            print(f"--- {k}: {type(ckpt[k])}")
else:
    # maybe a state_dict itself
    print("state_dict keys sample:", list(ckpt.keys())[:40])
