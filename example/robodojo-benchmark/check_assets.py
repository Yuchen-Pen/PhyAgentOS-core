"""Check real X5 URDF/mesh and selected-task layouts without launching Isaac Sim."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from preflight import load_config


def check(config):
    assets = (Path(config["robodojo_root"]) / "Assets").resolve()
    receipt = json.loads((assets.parent / "assets-ready.json").read_text())
    if receipt.get("status") != "ready" or receipt.get("assets_root") != str(assets):
        raise ValueError("full asset preparation has not finished")
    robot = assets / "Robots/x5"
    robot_config = yaml.safe_load((robot / "robot_config.yml").read_text())
    urdf = (robot / robot_config["urdf_path"]).resolve()
    if not urdf.is_relative_to(assets):
        raise ValueError("URDF escaped the configured asset root")
    tree = ET.parse(urdf)
    references = set()
    for mesh in tree.findall(".//mesh"):
        name = mesh.attrib["filename"]
        if "://" in name:
            raise ValueError("resolve external/package URDF mesh URI before declaring assets checked")
        path = (urdf.parent / name).resolve()
        if not path.is_relative_to(assets) or not path.is_file() or path.stat().st_size == 0:
            raise ValueError("missing or external X5 mesh: " + name)
        references.add(path)
    if not references:
        raise ValueError("X5 URDF contains no mesh references")
    layout_root = assets / "Eval_Layout/RoboDojo/arx_x5" / str(config["layout_seed"])
    layouts = sorted(layout_root.glob(config["suite"] + "_*.json"))
    if not layouts:
        raise ValueError("configured task has no layouts")
    for layout in layouts:
        if not isinstance(json.loads(layout.read_text()), dict):
            raise ValueError("invalid task layout: " + layout.name)
    return {
        "status": "passed", "assets_root": str(assets), "asset_files": receipt["files"],
        "asset_revision": receipt["revision"], "urdf": str(urdf), "unique_mesh_references": len(references),
        "suite": config["suite"], "layout_seed": config["layout_seed"], "valid_layout_files": len(layouts),
        "simulation_started": False, "usd_scene_loading_verified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args()
    print(json.dumps(check(load_config(args.config)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
