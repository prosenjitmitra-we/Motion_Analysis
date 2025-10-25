import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from pathlib import Path
from collections import defaultdict, deque

# Page configuration
st.set_page_config(
    page_title="Object Detection & Counter",
    page_icon="✈️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .big-metric {
        font-size: 28px;
        font-weight: bold;
        color: #1f77b4;
    }
    .detection-box {
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("✈️ Advanced Object Detection & Counter")
st.markdown(
    "Detect and count moving objects like jets, missiles, vehicles, people, and more using AI-powered object detection.")

# Sidebar settings
st.sidebar.header("⚙️ Detection Settings")

# Detection mode
detection_mode = st.sidebar.radio(
    "Detection Method",
    ["YOLO (AI-Based)", "Motion-Based"],
    help="YOLO detects specific objects like jets, vehicles. Motion-based detects any movement."
)

# Object selection for YOLO
if detection_mode == "YOLO (AI-Based)":
    st.sidebar.markdown("### 🎯 Select Objects to Detect")

    # COCO dataset classes (what YOLOv3 can detect)
    all_classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    # Popular presets
    preset = st.sidebar.selectbox(
        "Quick Presets",
        ["Custom", "Aerial Objects (jets, planes)", "Vehicles (cars, trucks, buses)",
         "People & Animals", "All Objects"]
    )

    if preset == "Aerial Objects (jets, planes)":
        selected_classes = ["airplane"]
    elif preset == "Vehicles (cars, trucks, buses)":
        selected_classes = ["car", "truck", "bus", "motorcycle", "bicycle"]
    elif preset == "People & Animals":
        selected_classes = ["person", "bird", "cat", "dog", "horse"]
    elif preset == "All Objects":
        selected_classes = all_classes
    else:
        selected_classes = st.sidebar.multiselect(
            "Choose objects to detect",
            all_classes,
            default=["airplane", "car", "person"]
        )

    confidence_threshold = st.sidebar.slider(
        "Detection Confidence (%)",
        min_value=10,
        max_value=100,
        value=50,
        step=5,
        help="Higher = fewer false positives, but may miss objects"
    )

    nms_threshold = st.sidebar.slider(
        "Overlap Threshold (NMS)",
        min_value=0.1,
        max_value=0.9,
        value=0.4,
        step=0.1,
        help="Removes duplicate detections"
    )

else:  # Motion-Based
    min_contour_area = st.sidebar.slider(
        "Minimum Motion Area (pixels)",
        min_value=500,
        max_value=10000,
        value=2000,
        step=100
    )

    var_threshold = st.sidebar.slider(
        "Detection Sensitivity",
        min_value=10,
        max_value=100,
        value=40,
        step=5
    )

# Common settings
show_preview = st.sidebar.checkbox("Show Video Preview", value=True)
preview_interval = st.sidebar.slider(
    "Preview Frame Interval",
    min_value=5,
    max_value=50,
    value=15,
    help="Show every Nth frame"
)

# File uploader
uploaded_file = st.file_uploader(
    "📁 Upload Video File",
    type=['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'],
    help="Upload video containing objects you want to detect and count"
)


class ObjectTracker:
    """Simple object tracker using centroid tracking"""

    def __init__(self, max_disappeared=30):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.counted_objects = set()

    def register(self, centroid, class_name):
        self.objects[self.next_object_id] = {
            'centroid': centroid,
            'class': class_name
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        """detections: list of (centroid, class_name)"""
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for centroid, class_name in detections:
                self.register(centroid, class_name)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [obj['centroid'] for obj in self.objects.values()]

            detection_centroids = [d[0] for d in detections]

            # Calculate distances
            D = np.zeros((len(object_centroids), len(detection_centroids)))
            for i, oc in enumerate(object_centroids):
                for j, dc in enumerate(detection_centroids):
                    D[i, j] = np.linalg.norm(np.array(oc) - np.array(dc))

            # Find minimum distances
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > 100:  # Max distance threshold
                    continue

                object_id = object_ids[row]
                self.objects[object_id]['centroid'] = detection_centroids[col]
                self.objects[object_id]['class'] = detections[col][1]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(len(object_centroids))) - used_rows
            unused_cols = set(range(len(detection_centroids))) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(detection_centroids[col], detections[col][1])

        return self.objects


def load_yolo():
    """Load YOLO model - using a lightweight version for Streamlit Cloud"""
    try:
        # Download YOLO files if not present
        weights_url = "https://pjreddie.com/media/files/yolov3-tiny.weights"
        config_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
        names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

        # For simplicity, we'll use OpenCV's DNN with pre-configured paths
        # In production, you'd download these files
        st.info("⚠️ YOLO model files needed. Using motion detection instead.")
        return None, None
    except:
        return None, None


def detect_objects_yolo(frame, net, classes, selected_classes, conf_threshold, nms_threshold):
    """Detect objects using YOLO"""
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold / 100:
                if classes[class_id] in selected_classes:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold / 100, nms_threshold)

    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            centroid = (x + w // 2, y + h // 2)
            class_name = classes[class_ids[i]]
            detections.append((centroid, class_name))

    return detections, boxes, confidences, class_ids, indices


def process_video_motion(video_path, settings):
    """Process video using motion detection"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=settings['var_threshold'],
        detectShadows=True
    )

    tracker = ObjectTracker(max_disappeared=20)
    object_counts = defaultdict(int)
    total_processed = 0

    progress_bar = st.progress(0)
    status_text = st.empty()
    preview_placeholder = st.empty() if settings['show_preview'] else None

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_processed += 1
        frame_count += 1

        fgmask = fgbg.apply(frame)
        blur = cv2.GaussianBlur(fgmask, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for c in contours:
            if cv2.contourArea(c) < settings['min_contour_area']:
                continue

            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                detections.append(((cX, cY), "moving_object"))

        objects = tracker.update(detections)

        # Count unique objects
        for obj_id, obj_data in objects.items():
            if obj_id not in tracker.counted_objects:
                tracker.counted_objects.add(obj_id)
                object_counts["moving_object"] += 1

        # Draw on frame
        for obj_id, obj_data in objects.items():
            cx, cy = obj_data['centroid']
            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
            cv2.putText(frame, f"ID:{obj_id}", (cx - 20, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        progress = total_processed / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing: Frame {total_processed}/{total_frames} | Objects: {len(objects)}")

        if settings['show_preview'] and preview_placeholder and frame_count % settings['preview_interval'] == 0:
            cv2.putText(frame, f"Total Counted: {object_counts['moving_object']}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(frame, f"Currently Tracking: {len(objects)}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preview_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()

    return {
        'total_frames': total_processed,
        'fps': fps,
        'duration': duration,
        'width': width,
        'height': height,
        'object_counts': dict(object_counts),
        'total_unique_objects': sum(object_counts.values())
    }


# Main processing
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        st.info(f"📁 File: {uploaded_file.name} ({uploaded_file.size / (1024 * 1024):.2f} MB)")

        if st.button("🚀 Start Detection & Counting", type="primary", use_container_width=True):
            st.markdown("---")

            with st.spinner("🔍 Detecting and counting objects..."):
                if detection_mode == "YOLO (AI-Based)":
                    st.warning("⚠️ YOLO mode requires model files. Using motion-based detection instead.")
                    settings = {
                        'min_contour_area': 2000,
                        'var_threshold': 40,
                        'show_preview': show_preview,
                        'preview_interval': preview_interval
                    }
                    results = process_video_motion(tmp_path, settings)
                else:
                    settings = {
                        'min_contour_area': min_contour_area,
                        'var_threshold': var_threshold,
                        'show_preview': show_preview,
                        'preview_interval': preview_interval
                    }
                    results = process_video_motion(tmp_path, settings)

            if results:
                st.success("✅ Detection Complete!")
                st.markdown("---")

                st.header("📊 Detection Results")

                # Main metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎯 Total Objects Detected", results['total_unique_objects'])
                with col2:
                    st.metric("⏱️ Duration", f"{results['duration']:.1f}s")
                with col3:
                    st.metric("🎞️ Total Frames", f"{results['total_frames']:,}")
                with col4:
                    st.metric("📹 FPS", f"{results['fps']:.1f}")

                # Object breakdown
                st.markdown("### 🎯 Object Counts")
                for obj_type, count in results['object_counts'].items():
                    st.markdown(f"**{obj_type.replace('_', ' ').title()}:** {count} unique objects")

                # Summary
                st.markdown("### 📝 Summary")
                st.info(f"""
                ✅ Successfully processed **{results['total_frames']:,} frames**  
                ✅ Detected and counted **{results['total_unique_objects']} unique moving objects**  
                ✅ Video resolution: **{results['width']}x{results['height']}**
                """)

            else:
                st.error("❌ Error processing video.")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

else:
    st.info("👆 Upload a video file to start detecting and counting objects")

    st.markdown("### 📖 How It Works")
    st.markdown("""
    1. **Upload** your video (jets, missiles, vehicles, etc.)
    2. **Select** detection method and objects to track
    3. **Adjust** sensitivity settings
    4. **Count** unique objects automatically

    **Perfect for:**
    - ✈️ Aerial surveillance (jets, drones, missiles)
    - 🚗 Traffic monitoring (vehicles, pedestrians)
    - 🏭 Production line counting
    - 🔒 Security footage analysis
    - 🐾 Wildlife tracking
    """)
