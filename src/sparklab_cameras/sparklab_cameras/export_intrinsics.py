import sys

import pyrealsense2 as rs
import yaml

def export_all_cameras_to_yaml(output_filename="realsense_cameras.yaml"):
    # Initialize context to discover all connected hardware
    context = rs.context()
    devices = context.query_devices()
    
    if len(devices) == 0:
        print("Error: No Intel RealSense devices detected.")
        return

    config_data = {"total_cameras": len(devices), "cameras": {}}

    print(f"Found {len(devices)} connected camera(s). Initializing streams...")

    # Iterate through each physical camera unit
    for idx, dev in enumerate(devices):
        serial_number = dev.get_info(rs.camera_info.serial_number)
        model_name = dev.get_info(rs.camera_info.name)
        camera_key = f"camera_{idx}_{serial_number}"
        
        print(f"Processing [{idx + 1}/{len(devices)}]: {model_name} (S/N: {serial_number})")

        # Configure pipeline explicitly for this specific serial number
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_number)
        
        # Start profile to fetch active streams and active camera settings
        try:
            profile = pipeline.start(config)
        except Exception as e:
            print(f"  Failed to start pipeline for {serial_number}: {e}")
            continue

        camera_entry = {
            "model": model_name,
            "serial_number": serial_number,
            "streams": {}
        }

        # Query all active video streams (Depth, Color, Infrared, etc.)
        active_streams = profile.get_streams()
        for stream_profile in active_streams:
            if stream_profile.is_video_stream_profile():
                video_profile = stream_profile.as_video_stream_profile()
                stream_type = str(video_profile.stream_type()).split('.')[-1]
                stream_idx = video_profile.stream_index()
                stream_name = f"{stream_type}_{stream_idx}" if stream_idx > 0 else stream_type
                
                # Fetch intrinsic metrics
                intrinsics = video_profile.get_intrinsics()
                
                camera_entry["streams"][stream_name] = {
                    "resolution": {
                        "width": intrinsics.width,
                        "height": intrinsics.height
                    },
                    "focal_length": {
                        "fx": float(intrinsics.fx),
                        "fy": float(intrinsics.fy)
                    },
                    "principal_point": {
                        "ppx": float(intrinsics.ppx),
                        "ppy": float(intrinsics.ppy)
                    },
                    "distortion": {
                        "model": str(intrinsics.model).split('.')[-1],
                        "coefficients": [float(c) for c in intrinsics.coeffs]
                    }
                }

        # Save to total configuration object and halt pipeline for next device
        config_data["cameras"][camera_key] = camera_entry
        pipeline.stop()

    # Write dictionary payload directly to the YAML target
    with open(output_filename, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        
    print(f"\nSuccess! Camera data exported successfully to '{output_filename}'")

def main() -> None:
    export_all_cameras_to_yaml(sys.argv[1] if len(sys.argv) > 1 else "intrinsics.yaml")


if __name__ == "__main__":
    main()
