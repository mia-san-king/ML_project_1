import csv
import gzip
import json
import os
import sys

import pytest

from models.t6_router import bundle


@pytest.fixture
def archived(tmp_path, monkeypatch):
    v1 = tmp_path / "v1"
    v1.mkdir()
    entries = {}
    for key, name in bundle.LEGACY_FILES.items():
        path = v1 / name
        path.write_bytes(key.encode())
        entries[key] = {"name": name, "sha256": bundle._sha256(path)}
    source = v1 / "T6.json"
    source.write_text(json.dumps({"model": "T6", "version": 1, "rule": bundle.LEGACY_RULE,
                                  "files": entries, "real_training_photos": 5}))
    gate = tmp_path / "fitted.joblib"
    gate.write_bytes(b"published gate")
    pinned = {key: entries[key]["sha256"] for key in bundle.FROZEN_SHA256 if key != "gate"}
    monkeypatch.setattr(bundle, "FROZEN_SHA256", {**pinned, "gate": bundle._sha256(gate)})
    return source, gate, tmp_path / "v2"


@pytest.mark.parametrize("hardlinks", [True, False])
def test_upgrade_cli_creates_complete_bundle_and_preserves_archive(archived, monkeypatch, hardlinks):
    source, gate, output = archived
    archived_hash = bundle._sha256(source)
    if not hardlinks:
        def unavailable(*args):
            raise OSError("hardlinks unavailable")
        monkeypatch.setattr(os, "link", unavailable)
    monkeypatch.setattr(sys, "argv", ["bundle", "--from-bundle", str(source), "--gate", str(gate),
                                      "--output", str(output)])
    bundle.main()
    manifest = output / "T6.json"
    description = json.loads(manifest.read_text())
    assert (description["version"], description["rule"], description["real_training_photos"]) == (2, bundle.RULE, 5)
    assert bundle._sha256(source) == archived_hash
    paths = bundle.load_bundle(manifest)
    assert set(paths) == set(bundle.FILES)
    assert paths["gate"].read_bytes() == gate.read_bytes()
    for key, original in bundle.load_bundle(source).items():
        assert os.path.samefile(original, paths[key]) == hardlinks
    with pytest.raises(FileExistsError):
        bundle.upgrade_bundle(from_bundle=source, gate=gate, output=output)


def test_rejects_wrong_gate_before_publication_and_modified_gate_at_inference(archived):
    source, gate, output = archived
    wrong = gate.with_suffix(".wrong")
    wrong.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="gate"):
        bundle.upgrade_bundle(from_bundle=source, gate=wrong, output=output)
    assert not output.exists()
    manifest = bundle.upgrade_bundle(from_bundle=source, gate=gate, output=output)
    (output / "gate.joblib").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="gate"):
        bundle.load_bundle(manifest)


def test_frozen_digests_match_published_experiment():
    report = bundle.ROOT / "report/experiments/2026-11-22-local-models"
    gate = json.loads((report / "t6/geographic_gate.json").read_text())["gate_sha256"]
    with gzip.open(report / "manifests/assets.tsv.gz", "rt", encoding="utf-8") as f:
        assets = {row[1]: row[3] for row in csv.reader(f, delimiter="\t")}
    expected = {key: assets["data/local_runs/t6_aug8192/bundle/" + name]
                for key, name in bundle.LEGACY_FILES.items() if key in bundle.FROZEN_SHA256}
    assert bundle.FROZEN_SHA256 == {**expected, "gate": gate}
