from .video_capture import VCDevice


def main():
    # Create the video capture device
    dev = VCDevice()
    # Run the device
    dev.run()
