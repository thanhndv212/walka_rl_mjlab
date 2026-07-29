"""One-time URDF -> MJCF conversion for the Walka biped.

Source: a URDF + OBJ meshes export (`jackbot_v1.urdf`, `meshes/*.obj`),
the geometric counterpart of the USD asset used in the IsaacLab-based
walka_lab project. Point this at that export directory:

    uv run python tools/convert_urdf_to_mjcf.py /path/to/v1_newjointlim

It writes src/assets/robots/walka/xmls/walka.xml and copies the meshes into
src/assets/robots/walka/xmls/assets/.

MuJoCo parses the URDF natively (mujoco.MjSpec.from_file), auto-computing
body mass/inertia from mesh volume with its default density since the
source URDF has no <inertial> tags — masses here (~28 kg total) are a
starting point, not CAD-verified values. The raw parse also flattens the
URDF's jackbot -> base_link -> pelvis chain (all zero-offset "fixed"
joints) straight into worldbody, since none of those are real joints; this
script re-wraps that flattened subtree in a body with a freejoint so the
robot has an actual floating base, and adds the pieces a URDF has no
concept of but mjlab's built-in velocity task requires:
  - geom names containing "collision" (walka_constants.py's CollisionCfg
    matches on ".*collision.*" and disables collision on anything that
    doesn't match)
  - footL/footR sites (TerrainHeightSensorCfg, foot_clearance, foot_slip)
  - an IMU site + gyro/velocimeter/accelerometer sensors, and a
    subtreeangmom sensor (mdp.builtin_sensor / the angular_momentum
    reward read these directly off the compiled MuJoCo model, matching
    mjlab's own g1.xml convention exactly)

Verified against a live env: both Walka-Rough and Walka-Flat build, reset,
and step (100 steps of random small actions, no NaNs) with the output of
this script in place.
"""

import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

REPO_ROOT = Path(__file__).resolve().parent.parent
XML_DIR = REPO_ROOT / "src" / "assets" / "robots" / "walka" / "xmls"
ASSETS_DIR = XML_DIR / "assets"
DST = XML_DIR / "walka.xml"


def convert(urdf_dir: Path) -> None:
    urdf_path = urdf_dir / "jackbot_v1.urdf"
    mesh_dir = urdf_dir / "meshes"
    if not urdf_path.exists():
        raise FileNotFoundError(f"Expected {urdf_path}")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for obj_file in mesh_dir.glob("*.obj"):
        shutil.copy(obj_file, ASSETS_DIR / obj_file.name)

    spec = mujoco.MjSpec.from_file(str(urdf_path))
    spec.compile()  # Fail fast if the raw URDF itself doesn't compile.
    root = ET.fromstring(spec.to_xml())
    root.set("model", "walka")

    compiler = root.find("compiler")
    compiler.set("meshdir", "assets")

    for mesh in root.find("asset").findall("mesh"):
        # meshdir="assets" now supplies the directory; files live flat there.
        mesh.set("file", mesh.get("file").split("/")[-1])

    worldbody = root.find("worldbody")

    # Everything under worldbody (one geom + pelvisL/pelvisR/abdomen bodies)
    # is the pelvis subtree that the URDF's fixed jackbot/base_link/pelvis
    # chain flattened straight into world. Move it all into a new floating
    # body with a freejoint.
    old_children = list(worldbody)
    for child in old_children:
        worldbody.remove(child)

    pelvis_body = ET.SubElement(worldbody, "body")
    pelvis_body.set("name", "pelvis")
    pelvis_body.set("pos", "0 0 0.832")
    ET.SubElement(pelvis_body, "freejoint", name="floating_base_joint")
    for child in old_children:
        pelvis_body.append(child)

    # Name every geom "<body>_collision" so CollisionCfg's
    # geom_names_expr=(".*collision.*",) in walka_constants.py matches them
    # (mjlab disables collision on any geom that doesn't match).
    for geom in pelvis_body.findall("geom"):
        geom.set("name", "pelvis_collision")
    seen: dict[str, int] = {}
    for body_elem in pelvis_body.iter("body"):
        name = body_elem.get("name")
        idx = seen.get(name, 0)
        for geom in body_elem.findall("geom"):
            suffix = f"_{idx}" if idx > 0 else ""
            geom.set("name", f"{name}_collision{suffix}")
            idx += 1
        seen[name] = idx

    # Foot sites for TerrainHeightSensorCfg / foot_clearance / foot_slip,
    # placed near the sole center (Cube_001/Cube_004.obj local bounding box:
    # x in [-0.117, 0.226], z in [-0.099, 0.022]), mirroring how g1.xml puts
    # a site on its ankle_roll body.
    for body_elem in pelvis_body.iter("body"):
        if body_elem.get("name") in ("footL", "footR"):
            ET.SubElement(
                body_elem, "site", name=body_elem.get("name"), pos="0.05 0 -0.099"
            )

    # IMU site + sensors (gyro/velocimeter/accelerometer + subtreeangmom),
    # matching g1.xml's convention exactly — mjlab's built-in velocity
    # observations and the angular_momentum reward read these directly off
    # the compiled model ("robot/imu_lin_vel", "robot/imu_ang_vel",
    # "robot/root_angmom"), and a URDF has no equivalent concept. No real
    # IMU mounting offset is known from the source data, so the site sits
    # at the pelvis origin rather than guessing a physical offset.
    ET.SubElement(pelvis_body, "site", name="imu_in_pelvis", size="0.01", pos="0 0 0")
    sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "gyro", name="imu_ang_vel", site="imu_in_pelvis")
    ET.SubElement(sensor, "velocimeter", name="imu_lin_vel", site="imu_in_pelvis")
    ET.SubElement(sensor, "accelerometer", name="imu_lin_acc", site="imu_in_pelvis")
    ET.SubElement(sensor, "subtreeangmom", name="root_angmom", body="pelvis")

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(DST, encoding="unicode", xml_declaration=False)

    # Fail fast if the restructured XML doesn't compile.
    mujoco.MjSpec.from_file(str(DST)).compile()
    print(f"wrote {DST}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} /path/to/v1_newjointlim")
    convert(Path(sys.argv[1]))
