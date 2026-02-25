import cv2
from collections import deque
import time
import numpy as np

ip_address = "insert here"
rtsp_url = f"rtsp://{ip_address}:8554/live.sdp"

# Store up to ~7.5 seconds of samples
MAX_SAMPLES = 200  
MIN_INTERVAL = 300 # 300 ms

cap = cv2.VideoCapture(rtsp_url)

# Rolling buffer of (brightness, timestamp_ms)
samples = deque(maxlen=MAX_SAMPLES)

def get_average_brightness(frame):
    # frame is RGB
    r = frame[:, :, 0]
    g = frame[:, :, 1]
    return (r.mean() + g.mean()) / 255.0  # normalize 0..1

# Graph settings

GRAPH_WIDTH = 600
GRAPH_HEIGHT = 200
MIDLINE = GRAPH_HEIGHT // 2

# Create blank graph
graph = np.zeros((GRAPH_HEIGHT, GRAPH_WIDTH, 3), dtype=np.uint8)

# Track previous y‑position for line drawing
prev_y = MIDLINE

def draw_waveform(graph, value):
    global prev_y

    # Scroll graph left by 1 pixel
    graph[:, :-1] = graph[:, 1:]
    graph[:, -1] = (0, 0, 0)

    scale = 3000  # adjust sensitivity
    y = int(MIDLINE - value * scale)

    y = max(0, min(GRAPH_HEIGHT - 1, y))

    cv2.line(graph, (GRAPH_WIDTH - 2, prev_y), (GRAPH_WIDTH - 1, y), (0, 255, 0), 1)

    prev_y = y
    return graph

def compute_bpm(samples):
    global graph
    if len(samples) < 3:
        return None

    # Extract brightness only
    values = [v for v, t in samples]
    avg = sum(values) / len(values)

    # Detect downward crossings through the average
    crossings = []
    for i in range(1, len(samples)):
        prev_val, prev_t = samples[i-1]
        curr_val, curr_t = samples[i]

        if curr_val < avg and prev_val > avg:
            crossings.append(curr_t)

    if len(crossings) < 2:
        return None

    # Compute average interval between crossings
    intervals = [
        crossings[i] - crossings[i-1]
        for i in range(1, len(crossings))
    ]

    # Waveform value = brightness - average
    wave_value = -(samples[-1][0] - avg)

    # Draw waveform
    graph = draw_waveform(graph, wave_value)
    cv2.imshow("Pulse Waveform", graph)

    avg_interval = sum(intervals) / len(intervals)  # ms
    bpm = 60000 / avg_interval
    return bpm


while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not received")
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    brightness = get_average_brightness(rgb)
    timestamp = time.time() * 1000  # ms

    samples.append((brightness, timestamp))

    # Compute average brightness over buffer
    bpm = compute_bpm(samples)
    if bpm:
        print("BPM:", round(bpm))

    cv2.imshow("RTSP Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
