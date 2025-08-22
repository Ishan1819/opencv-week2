import cv2
import numpy as np
import threading
import time
from collections import deque

# more reliable HTTP video streams 
STREAMS = [
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4"
]

# shared frames with loading text
frames = []
for i in range(4):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, f"Loading Stream {i+1}...", (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    frames.append(frame)

lock = threading.Lock()
stream_status = ["Connecting..."] * 4


def is_frame_blurry(frame, threshold=100):
    """Check if frame is blurry using Laplacian variance."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


def is_frame_covered(frame, dark_thresh=40, bright_thresh=220, percent=0.75):
    """Check if frame is covered (too dark/too bright)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    total_pixels = gray.size
    dark_pixels = np.sum(gray < dark_thresh)
    bright_pixels = np.sum(gray > bright_thresh)

    if dark_pixels / total_pixels > percent:
        return True
    if bright_pixels / total_pixels > percent:
        return True
    return False


def fetch_stream(idx, url):
    """Fetch frames from HTTP video stream with motion + integrity checks."""
    global stream_status
    print(f"Starting stream {idx+1}: {url}")

    # Create motion detector
    back_sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50, detectShadows=True)
    motion_threshold = 5000  # pixels threshold for motion

    # Track compromised frames
    history = deque(maxlen=30)  # last 30 frames

    while True:  # Loop to restart video
        try:
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                print(f"❌ Failed to open stream {idx+1}")
                # Create error frame
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, f"Stream {idx+1} FAILED", (200, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                with lock:
                    frames[idx] = error_frame
                    stream_status[idx] = "FAILED TO CONNECT"
                time.sleep(5)
                continue

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Stream {idx+1} connected: {width}x{height}, {total_frames} frames @ {fps}fps")

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"🔄 Stream {idx+1} ended, restarting...")
                    break  # Restart the video

                frame_count += 1
                frame = cv2.resize(frame, (640, 480))

                # motion detection
                fg_mask = back_sub.apply(frame)
                moving_pixels = cv2.countNonZero(fg_mask)

                if moving_pixels > motion_threshold:
                    cv2.putText(frame, "Motion Detected", (150, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.rectangle(frame, (50, 100), (590, 380), (0, 0, 255), 2)
                    
                # blurr detection
                compromised = False
                if is_frame_blurry(frame):
                    cv2.putText(frame, "Blur Detected", (150, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    compromised = True
                    
                # covering detection
                if is_frame_covered(frame):
                    cv2.putText(frame, "Covered/Blocked", (150, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    compromised = True

                history.append(1 if compromised else 0)
                if len(history) == history.maxlen and sum(history) / len(history) > 0.75:
                    cv2.putText(frame, "Camera Compromised", (120, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                # Adding stream info
                cv2.putText(frame, f"Stream {idx+1}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 460),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Update shared frame
                with lock:
                    frames[idx] = frame.copy()
                    stream_status[idx] = f"Playing {frame_count}/{total_frames}"

                if fps > 0:
                    time.sleep(1.0 / fps * 0.5)  # play at half speed for demo (it can be 1.0)

            cap.release()

        except Exception as e:
            print(f"Stream {idx+1} error: {e}")
            time.sleep(5)


def main():
    print("Starting Multi-Stream Viewer with Motion + Integrity Check...")
    print(f"OpenCV Version: {cv2.__version__}")

    # start threads to on the stream
    threads = []
    for i, url in enumerate(STREAMS):
        t = threading.Thread(target=fetch_stream, args=(i, url))
        t.daemon = True
        t.start()
        threads.append(t)

    print("Press 'q' to quit, 's' for status")
    start_time = time.time()

    while True:
        with lock:
            top = np.hstack((frames[0], frames[1]))
            bottom = np.hstack((frames[2], frames[3]))
            grid = np.vstack((top, bottom))

            # adding timestamp to get the time
            runtime = int(time.time() - start_time)
            cv2.putText(grid, f"Runtime: {runtime}s", (10, grid.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Multi-Stream (Motion + Integrity)", grid)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("\nStream Status:")
            for i, status in enumerate(stream_status):
                print(f" Stream {i+1}: {status}")

    cv2.destroyAllWindows()
    print("Viewer closed")


if __name__ == "__main__":
    main()
