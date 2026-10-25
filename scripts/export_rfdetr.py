#!/usr/bin/env python3
"""Export an Apache-2.0 RF-DETR-Seg checkpoint to ONNX for reco-anon and
write the manifest (<model>.onnx.json) the CLI requires.

    python3 -m venv .venv && . .venv/bin/activate
    pip install "rfdetr>=1.3" onnx onnxsim
    python scripts/export_rfdetr.py --size medium --out models/rfdetr-seg-medium.onnx
    python scripts/export_rfdetr.py --task det --size small --out models/rfdetr-det-small.onnx

`--task det` exports a detection-only model (boxes, no masks): heads are then
derived from the top of each person box. Much faster, see README.

Only the Apache-2.0 sizes are allowed here (nano/small/medium/large).
RF-DETR XL/2XL and the `rfdetr_plus` models use a different licence.
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import tempfile

APACHE_SIZES = {
    "seg": {
        "nano": ["RFDETRSegNano"],
        "small": ["RFDETRSegSmall"],
        "medium": ["RFDETRSegMedium", "RFDETRSegPreview"],
        "large": ["RFDETRSegLarge"],
    },
    "det": {
        "nano": ["RFDETRNano"],
        "small": ["RFDETRSmall"],
        "medium": ["RFDETRMedium"],
        "large": ["RFDETRLarge"],
    },
}


def load_class(task, size):
    import rfdetr

    for name in APACHE_SIZES[task][size]:
        cls = getattr(rfdetr, name, None)
        if cls is not None:
            return name, cls
    sys.exit(f"rfdetr {getattr(rfdetr, '__version__', '?')} has none of {APACHE_SIZES[task][size]}; upgrade rfdetr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(APACHE_SIZES), default="seg")
    ap.add_argument("--size", choices=sorted(APACHE_SIZES["seg"]), default="medium")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    name, cls = load_class(args.task, args.size)
    model = cls()
    tmp = tempfile.mkdtemp()
    model.export(output_dir=tmp)
    onnx_files = sorted(glob.glob(os.path.join(tmp, "**", "*.onnx"), recursive=True))
    if not onnx_files:
        sys.exit(f"export produced no .onnx in {tmp}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    shutil.copy(onnx_files[0], args.out)

    import onnx

    m = onnx.load(args.out)
    outs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]
    ins = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]
    print("inputs :", ins)
    print("outputs:", outs)
    if args.task == "seg" and not any(len(s) == 4 for _, s in outs):
        print("WARNING: no mask output found; heads will be derived from boxes only")

    sha = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    manifest = {
        "id": f"rfdetr-{name}",
        "sha256": sha,
        "license": "Apache-2.0 (code and COCO weights, roboflow/rf-detr)",
        "kind": "person-seg" if args.task == "seg" else "person-det",
        # RF-DETR COCO checkpoints index logits by COCO category id: person = 1
        "person_class": 1,
        "source": f"rfdetr.{name} via model.export()",
    }
    with open(args.out + ".json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {args.out} and {args.out}.json (sha256 {sha[:16]}…)")


if __name__ == "__main__":
    main()
