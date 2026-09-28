import streamlit as st
import tempfile
import os

from lip_sync_detector import detect_lip_sync


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Lip-Sync Fraud Detector",
    page_icon="🕵🏼‍♂️",
    layout="wide"
)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# TITLE
# ============================================================

st.title("🕵🏼‍♂️ Lip-Sync Fraud Detector")

st.write(
    "Detect potential lip-sync manipulation and "
    "audio-video mismatch."
)

st.markdown("---")


# ============================================================
# UPLOAD VIDEO
# ============================================================

uploaded_file = st.file_uploader(
    "📁 Upload a video",
    type=["mp4", "avi", "mov", "mkv"]
)


# ============================================================
# VIDEO ANALYSIS
# ============================================================

if uploaded_file is not None:

    st.subheader("Uploaded Video")

    st.video(uploaded_file)

    st.markdown("---")

    if st.button(
        "🔍 Analyze Video",
        use_container_width=True
    ):

        temp_path = None

        try:

            # ------------------------------------------------
            # Save uploaded video temporarily
            # ------------------------------------------------

            file_extension = os.path.splitext(
                uploaded_file.name
            )[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=file_extension
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                temp_path = temp_file.name


            # ------------------------------------------------
            # Run detector
            # ------------------------------------------------

            with st.spinner(
                "Analyzing audio and lip movement..."
            ):

                detector_output = detect_lip_sync(
                    temp_path
                )


            # =================================================
            # READ DETECTOR OUTPUT
            # =================================================

            sync_score = None
            detected_lag = 0
            detector_result = None


            if isinstance(detector_output, dict):

                detector_result = detector_output.get(
                    "result",
                    detector_output.get(
                        "status",
                        None
                    )
                )

                sync_score = detector_output.get(
                    "sync_score",
                    detector_output.get(
                        "score",
                        detector_output.get(
                            "synchronization_score",
                            None
                        )
                    )
                )

                detected_lag = detector_output.get(
                    "detected_lag",
                    detector_output.get(
                        "lag",
                        0
                    )
                )


            elif isinstance(
                detector_output,
                (tuple, list)
            ):

                if len(detector_output) >= 1:
                    detector_result = detector_output[0]

                if len(detector_output) >= 2:
                    sync_score = detector_output[1]

                if len(detector_output) >= 3:
                    detected_lag = detector_output[2]


            elif isinstance(
                detector_output,
                (int, float)
            ):

                sync_score = float(
                    detector_output
                )


            if sync_score is None:

                raise ValueError(
                    "The detector did not return "
                    "a synchronization score."
                )


            sync_score = float(sync_score)


            # =================================================
            # SAME THRESHOLD AS DETECTOR
            # =================================================

            DETECTION_THRESHOLD = 60.0

            is_matched = (
                sync_score >= DETECTION_THRESHOLD
            )


            # =================================================
            # RESULT
            # =================================================

            if is_matched:

                display_result = (
                    "MATCHED / GENUINE"
                )

                status = "Matched"

                explanation = (
                    "The audio and mouth-movement signals "
                    "show reasonable temporal alignment. "
                    "No significant lip-sync mismatch "
                    "was detected."
                )

                st.success(
                    f"✅ {display_result}"
                )

            else:

                display_result = (
                    "LIP-SYNC FRAUD / TAMPERED"
                )

                status = "Mismatch"

                explanation = (
                    "The audio and visual signals show "
                    "low temporal correlation. This may "
                    "indicate possible lip-sync manipulation "
                    "or video-audio mismatch."
                )

                st.error(
                    f"⚠️ {display_result}"
                )


            # =================================================
            # SCORE
            # =================================================

            st.metric(
                "Synchronization Score",
                f"{sync_score:.2f}%"
            )


            # =================================================
            # METRICS
            # =================================================

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Detected Lag",
                    f"{detected_lag} frames"
                )

            with col2:

                st.metric(
                    "Status",
                    status
                )


            # =================================================
            # EXPLANATION
            # =================================================

            st.markdown(
                "### 🧠 Analysis Explanation"
            )

            st.info(
                explanation
            )


            # =================================================
            # THRESHOLD
            # =================================================

            st.caption(
                "Detection threshold: 60% | "
                "Score ≥ 60% = MATCHED | "
                "Score < 60% = POSSIBLE MISMATCH"
            )


        except Exception as e:

            st.error(
                "❌ An error occurred while "
                "analyzing the video."
            )

            st.code(
                str(e)
            )


        finally:

            # ------------------------------------------------
            # Remove temporary file
            # ------------------------------------------------

            if (
                temp_path is not None
                and os.path.exists(temp_path)
            ):

                try:
                    os.remove(temp_path)

                except Exception:
                    pass


# ============================================================
# INFORMATION
# ============================================================

else:

    st.info(
        "👆 Upload a video above to begin analysis."
    )

    st.markdown(
        "### How it works"
    )

    st.write(
        """
        1. Upload a video containing a visible speaker.
        2. Extract the audio signal.
        3. Detect mouth movement using facial landmarks.
        4. Compare audio activity with mouth movement.
        5. Calculate the synchronization score.
        6. Classify the video using the 60% threshold.
        """
    )