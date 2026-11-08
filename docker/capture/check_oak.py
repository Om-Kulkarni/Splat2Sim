#!/opt/r2s-venv/bin/python
"""`capture.sh --check` for the OAK: device present, USB link speed, product/sensors, calibration model.

Exit codes: 0 ok, 2 no device / cannot open it, 3 USB link too slow, 4 depthai missing.
Writes a JSON summary to the path in $R2S_CHECK_JSON (used by run.sh for capture_info.json).
"""
import json
import os
import sys

try:
    import depthai as dai
except ImportError:
    print("ERROR: the depthai python package is missing from the image (rebuild: R2S_REBUILD=1 ./scripts/capture.sh --check)", file=sys.stderr)
    sys.exit(4)

ROLLING = {"IMX378", "IMX214", "IMX582", "IMX577", "IMX296"}
GLOBAL = {"OV9282", "OV9782", "AR0234", "OV7251"}


def fail(msg, code=2):
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(code)


devs = dai.Device.getAllAvailableDevices()
if not devs:
    fail("no OAK device detected.\n"
         "  - plug the OAK-D Pro W into a USB 3 port with the supplied USB 3 cable; check 'lsusb | grep 03e7'\n"
         "  - the camera is only visible to one process: stop any other depthai/ROS driver using it\n"
         "  - run this script with the container --privileged and /dev mounted (capture.sh does that)\n"
         "  - if it was just unplugged/crashed, replug it")
print(f"depthai {dai.__version__}; devices: " + ", ".join(f"{d.deviceId} ({d.state.name})" for d in devs))

try:
    dev = dai.Device()
except Exception as e:  # noqa: BLE001
    fail(f"cannot open the OAK device: {e}\n  - X_LINK_DEVICE_NOT_FOUND / already in use: unplug and replug, close other users of the camera")

with dev:
    info = {"driver": "depthai_ros_driver_v3", "depthai": dai.__version__, "serial": dev.getDeviceId()}
    cal = dev.readCalibration()
    ed = cal.getEepromData()
    info["product"] = ed.productName or dev.getDeviceName()
    info["board"] = ed.boardName
    speed = dev.getUsbSpeed().name
    info["usb_speed"] = speed
    print(f"product: {info['product']}   board: {ed.boardName} rev {ed.boardRev}   serial: {info['serial']}")
    print(f"USB link: {speed}")
    sensors = {}
    for f in dev.getConnectedCameraFeatures():
        sensors[f.socket.name] = f.sensorName
        shutter = "rolling shutter" if f.sensorName in ROLLING else "global shutter" if f.sensorName in GLOBAL else "shutter ?"
        print(f"  {f.socket.name}: {f.sensorName}  max {f.width}x{f.height}  ({shutter})")
    info["sensors"] = sensors
    for sock in (dai.CameraBoardSocket.CAM_A, dai.CameraBoardSocket.CAM_B):
        try:
            model = cal.getDistortionModel(sock).name
            coeffs = cal.getDistortionCoefficients(sock)
            info[f"calibration_{sock.name}"] = {"model": model, "n_coeffs": len(coeffs)}
            print(f"  calibration {sock.name}: model {model}, {len(coeffs)} distortion coeffs, max |k| {max(abs(c) for c in coeffs):.2f}")
        except Exception as e:  # noqa: BLE001
            print(f"  calibration {sock.name}: not readable ({e})")
    try:
        imu = dev.getConnectedIMU()
        info["imu"] = imu
        print(f"IMU: {imu} (firmware {dev.getIMUFirmwareVersion()})")
    except Exception as e:  # noqa: BLE001
        info["imu"] = None
        print(f"WARN: could not query the IMU ({e})")

    rc = 0
    if "OAK-D-PRO-W" not in str(info["product"]).upper().replace(" ", "-"):
        print("WARN: not an OAK-D Pro W (protocol and docs assume it; the pipeline itself reads intrinsics from the camera)")
    if not info.get("imu"):
        print("WARN: no IMU on this device: RTAB-Map runs without IMU (poses.use_imu=false)")
    if speed not in ("SUPER", "SUPER_PLUS"):
        print(f"ERROR: camera enumerated as USB {speed}. 1280x720 colour + depth + IMU needs USB 3 (SUPER). "
              "Use a different port / the supplied short USB 3 cable, no hub.", file=sys.stderr)
        rc = 3
    if "IMX378" in sensors.values() and rc == 0:
        print("note: colour sensor is a rolling shutter: keep exposure short and walk slowly")

out = os.environ.get("R2S_CHECK_JSON")
if out:
    with open(out, "w") as fh:
        json.dump(info, fh, indent=2)
if rc == 0:
    print("check OK")
sys.exit(rc)
