"""ingest stage: Capture (ROS 2 MCAP / librealsense .bag / TUM folder) -> ``runs/<scene>/frames/``.

Deliberately free of pydantic/ROS imports so it runs in the slim ``ingest`` container.
"""
