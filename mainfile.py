import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Motion Detection Analyzer",
    page_icon="🎥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .big-metric {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🎥 Motion Detection Analyzer")
st.markdown("Upload a video to detect and count motion events using advanced background subtraction.")

# Sidebar for settings
st.sidebar.header("⚙️ Detection Settings")

min_contour_area = st.sidebar.slider(
    "Minimum Motion Area (pixels)",
    min_value=500,
    max_value=5000,
    value=1500,
    step=100,
    help="Larger values = less sensitive to small movements"
)

var_threshold = st.sidebar.slider(
    "Detection Sensitivity",
    min_value=10,
    max_value=100,
    value=60,
    step=5,
    help="Lower values = more sensitive to motion"
)

history = st.sidebar.slider(
    "Background Learning (frames)",
    min_value=100,
    max_value=1000,
    value=500,
    step=50,
    help="Number of frames used to build background model"
)

show_preview = st.sidebar.checkbox("Show Video Preview", value=True)
preview_interval = st.sidebar.slider(
    "Preview Frame Interval",
    min_value=10,
    max_value=100,
    value=30,
    help="Show every Nth frame during processing"
)

# File uploader
uploaded_file = st.file_uploader(
    "Choose a video file",
    type=['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'],
    help="Upload your video file (max 200MB recommended)"
)


def process_video(video_path, settings):
    """Process video and detect motion"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    # Initialize background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=settings['history'],
        varThreshold=settings['var_threshold'],
        detectShadows=True
    )

    # Statistics
    motion_counter = 0
    motion_frames = 0
    total_processed = 0
    motion_timestamps = []

    # Progress tracking
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

        # Apply background subtraction
        fgmask = fgbg.apply(frame)

        # Remove noise
        blur = cv2.GaussianBlur(fgmask, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        motion_areas = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < settings['min_contour_area']:
                continue
            motion_detected = True
            motion_areas.append(area)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if motion_detected:
            motion_counter += 1
            motion_frames += 1
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            motion_timestamps.append({
                'time': current_time,
                'frame': total_processed,
                'areas': motion_areas
            })

        # Update progress
        progress = total_processed / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing: Frame {total_processed}/{total_frames} ({progress * 100:.1f}%)")

        # Show preview every N frames
        if settings['show_preview'] and preview_placeholder and frame_count % settings['preview_interval'] == 0:
            # Add motion indicator
            status_color = (0, 255, 0) if motion_detected else (128, 128, 128)
            cv2.circle(frame, (30, 30), 15, status_color, -1)
            cv2.putText(frame, f"Frame: {total_processed}", (60, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Convert BGR to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preview_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()

    # Calculate final statistics
    motion_percentage = (motion_frames / total_processed * 100) if total_processed > 0 else 0

    return {
        'total_frames': total_processed,
        'motion_frames': motion_frames,
        'motion_events': motion_counter,
        'motion_percentage': motion_percentage,
        'fps': fps,
        'duration': duration,
        'width': width,
        'height': height,
        'motion_timestamps': motion_timestamps
    }


# Main processing
if uploaded_file is not None:
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        # Display video info
        st.info(f"📁 File: {uploaded_file.name} ({uploaded_file.size / (1024 * 1024):.2f} MB)")

        # Process button
        if st.button("🚀 Start Motion Detection", type="primary"):
            st.markdown("---")

            with st.spinner("Processing video..."):
                settings = {
                    'min_contour_area': min_contour_area,
                    'var_threshold': var_threshold,
                    'history': history,
                    'show_preview': show_preview,
                    'preview_interval': preview_interval
                }

                results = process_video(tmp_path, settings)

            if results:
                st.success("✅ Processing Complete!")
                st.markdown("---")

                # Display results
                st.header("📊 Motion Analysis Results")

                # Metrics in columns
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Frames", f"{results['total_frames']:,}")
                with col2:
                    st.metric("Frames with Motion", f"{results['motion_frames']:,}")
                with col3:
                    st.metric("Motion Events", f"{results['motion_events']:,}")
                with col4:
                    st.metric("Motion Percentage", f"{results['motion_percentage']:.2f}%")

                # Video properties
                st.markdown("### 🎬 Video Properties")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write(f"**Duration:** {results['duration']:.2f}s")
                with col2:
                    st.write(f"**FPS:** {results['fps']:.2f}")
                with col3:
                    st.write(f"**Resolution:** {results['width']}x{results['height']}")
                with col4:
                    st.write(f"**Total Size:** {results['total_frames']} frames")

                # Motion timeline
                if results['motion_timestamps']:
                    st.markdown("### ⏱️ Motion Timeline")

                    # Create timeline data
                    times = [mt['time'] for mt in results['motion_timestamps'][:100]]  # Limit to first 100

                    if times:
                        st.write(f"First 100 motion events (of {len(results['motion_timestamps'])} total):")

                        # Show as expandable table
                        with st.expander("View detailed motion events"):
                            for i, mt in enumerate(results['motion_timestamps'][:100]):
                                st.write(
                                    f"Event {i + 1}: Time {mt['time']:.2f}s | Frame {mt['frame']} | Areas: {len(mt['areas'])}")

                # Download results
                st.markdown("### 💾 Export Results")

                # Create CSV content
                csv_content = "Event,Time(s),Frame,Motion_Areas\n"
                for i, mt in enumerate(results['motion_timestamps']):
                    csv_content += f"{i + 1},{mt['time']:.2f},{mt['frame']},{len(mt['areas'])}\n"

                st.download_button(
                    label="📥 Download Motion Report (CSV)",
                    data=csv_content,
                    file_name=f"motion_report_{uploaded_file.name}.csv",
                    mime="text/csv"
                )
            else:
                st.error("❌ Error processing video. Please try another file.")

    finally:
        # Cleanup temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

else:
    # Instructions when no file is uploaded
    st.info("👆 Please upload a video file to begin motion detection analysis.")

    st.markdown("### 📖 How it works:")
    st.markdown("""
    1. **Upload** your video file using the file uploader above
    2. **Adjust** detection settings in the sidebar if needed
    3. **Click** the "Start Motion Detection" button
    4. **View** real-time processing and results
    5. **Download** the motion report as CSV

    **Tips:**
    - Increase "Minimum Motion Area" to ignore small movements
    - Decrease "Detection Sensitivity" for more sensitive detection
    - Enable "Show Video Preview" to see processing in real-time
    """)

    st.markdown("### 🎯 Use Cases:")
    st.markdown("""
    - **Security & Surveillance**: Detect movement in camera footage
    - **Sports Analysis**: Track player movements
    - **Wildlife Monitoring**: Detect animal activity
    - **Traffic Analysis**: Count vehicles or pedestrians
    - **Quality Control**: Monitor production lines
    """)