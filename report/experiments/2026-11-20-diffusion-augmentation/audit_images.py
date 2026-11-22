"""Check a separate SQLite image archive and render paired originals and variants."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps


def audit(args):
    with sqlite3.connect((args.store / "synthetic.sqlite").resolve().as_uri()+"?mode=ro", uri=True) as db:
        catalog = pd.read_sql_query("SELECT * FROM samples ORDER BY parent_id", db)
    accepted = catalog[catalog.status == "accepted"]
    if not len(accepted):
        raise ValueError("No accepted images to verify")
    assert accepted.parent_id.is_unique and accepted.synthetic_id.is_unique
    assert (accepted.synthetic_id == -accepted.parent_id).all()
    assert (accepted.is_synthetic == 1).all() and (accepted.is_diffusion == 1).all()
    assert (accepted.parent_split == "train").all()
    meta = pd.read_parquet(args.data_root / "meta/smapshot.parquet",
                           columns=["id","latitude","longitude","date_shot_min","date_shot_max"]).set_index("id")
    original = meta.loc[accepted.parent_id]
    split = pd.read_csv(args.split_csv, usecols=["id","split"]).set_index("id")
    assert split.loc[accepted.parent_id, "split"].eq("train").all()
    assert not split.index.isin(accepted.synthetic_id).any()
    for key in ("latitude","longitude"):
        np.testing.assert_allclose(original[key], accepted[key], rtol=0, atol=1e-10)
    for key in ("date_shot_min","date_shot_max"):
        raw = pd.to_datetime(original[key], errors="coerce", utc=True, format="mixed").reset_index(drop=True)
        copied = pd.to_datetime(accepted[key], errors="coerce", utc=True, format="mixed").reset_index(drop=True)
        assert raw.equals(copied) or ((raw == copied) | (raw.isna() & copied.isna())).all()
    for row in accepted.itertuples():
        path = args.store / row.image_path
        assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == row.image_sha256
        with Image.open(path) as im:
            im.verify()
    sample = accepted.sample(min(8,len(accepted)),random_state=29).reset_index(drop=True)
    tile_w, tile_h = 214, 165
    sheet = Image.new("RGB",(tile_w*4,(tile_h+23)*((len(sample)+1)//2)),color="white")
    draw = ImageDraw.Draw(sheet)
    for i,row in enumerate(sample.itertuples()):
        y = (i//2)*(tile_h+23)
        x = (i%2)*tile_w*2
        parent = args.images_root/f"{int(row.parent_id)//1000:04d}"/f"{int(row.parent_id)}.jpg"
        for j,(p,title) in enumerate(((parent,"Archive"),(args.store/row.image_path,"Diffusion"))):
            with Image.open(p) as im:
                small = ImageOps.contain(ImageOps.exif_transpose(im).convert("RGB"),(tile_w-5,tile_h-4))
                sheet.paste(small,(x+j*tile_w+(tile_w-small.width)//2, y+(tile_h-small.height)//2))
            draw.text((x+j*tile_w+5,y+tile_h+2),f"{title}   {row.parent_id}",fill="black")
    sheet.save(args.preview)
    report = {"counts":{str(k):int(n) for k,n in catalog.status.value_counts().items()},
              "accepted_verified":len(accepted),"preview_parent_ids":sample.parent_id.tolist(),
              "unique_seeds":int(accepted.seed.nunique()),"model_revisions":accepted.model_revision.unique().tolist()}
    args.report.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("store","images-root","data-root","split-csv","preview","report"):
        parser.add_argument("--"+name,type=Path,required=True)
    audit(parser.parse_args())
