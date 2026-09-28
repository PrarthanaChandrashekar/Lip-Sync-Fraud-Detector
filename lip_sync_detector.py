import cv2
import numpy as np
import librosa
import tempfile
import os

from scipy.signal import correlate


# ============================================================
# AUDIO EXTRACTION
# ============================================================

def extract_audio(video_path):

    from moviepy import VideoFileClip

    temp_audio = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )

    temp_audio.close()

    video = VideoFileClip(video_path)

    if video.audio is None:
        video.close()
        return None

    video.audio.write_audiofile(
        temp_audio.name,
        logger=None
    )

    video.close()

    return temp_audio.name


# ============================================================
# AUDIO ANALYSIS
# ============================================================

def analyze_audio(audio_path):

    y, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    # Remove very low-level background noise
    y = librosa.util.normalize(y)

    # RMS energy
    rms = librosa.feature.rms(
        y=y,
        frame_length=1024,
        hop_length=256
    )[0]

    # Smooth the audio envelope
    window = 5

    if len(rms) > window:

        rms = np.convolve(
            rms,
            np.ones(window) / window,
            mode="same"
        )

    rms = normalize_signal(rms)

    return rms, sr


# ============================================================
# MOUTH LANDMARK ANALYSIS
# ============================================================

def analyze_video(video_path):

    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    mp_face_mesh = mp.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    mouth_signal = []

    frame_count = 0

    # MediaPipe mouth landmarks
    #
    # 13  = upper inner lip
    # 14  = lower inner lip
    # 61  = left mouth corner
    # 291 = right mouth corner

    TOP_LIP = 13
    BOTTOM_LIP = 14
    LEFT_CORNER = 61
    RIGHT_CORNER = 291

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Process every second frame
        if frame_count % 2 != 0:
            frame_count += 1
            continue

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:

            mouth_signal.append(0)

            frame_count += 1
            continue

        landmarks = (
            results.multi_face_landmarks[0]
            .landmark
        )

        top = landmarks[TOP_LIP]
        bottom = landmarks[BOTTOM_LIP]

        left = landmarks[LEFT_CORNER]
        right = landmarks[RIGHT_CORNER]

        # Vertical mouth opening
        vertical_distance = np.sqrt(
            (top.x - bottom.x) ** 2 +
            (top.y - bottom.y) ** 2
        )

        # Mouth width
        mouth_width = np.sqrt(
            (left.x - right.x) ** 2 +
            (left.y - right.y) ** 2
        )

        # Normalize mouth opening by mouth width.
        # This reduces the effect of face distance
        # from the camera.

        if mouth_width > 0:

            mouth_opening = (
                vertical_distance /
                mouth_width
            )

        else:

            mouth_opening = 0

        mouth_signal.append(
            mouth_opening
        )

        frame_count += 1

    cap.release()

    face_mesh.close()

    if len(mouth_signal) < 10:

        return None, None

    mouth_signal = np.array(
        mouth_signal,
        dtype=np.float32
    )

    mouth_signal = normalize_signal(
        mouth_signal
    )

    effective_fps = fps / 2

    return mouth_signal, effective_fps


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_signal(signal):

    signal = np.asarray(
        signal,
        dtype=np.float32
    )

    if len(signal) == 0:
        return signal

    minimum = np.percentile(
        signal,
        5
    )

    maximum = np.percentile(
        signal,
        95
    )

    if maximum - minimum < 1e-8:

        return np.zeros_like(signal)

    signal = (
        (signal - minimum) /
        (maximum - minimum)
    )

    return np.clip(
        signal,
        0,
        1
    )


# ============================================================
# RESAMPLING
# ============================================================

def resample_signal(
    signal,
    target_length
):

    if len(signal) == target_length:

        return signal

    if len(signal) < 2:

        return np.zeros(
            target_length
        )

    old_time = np.linspace(
        0,
        1,
        len(signal)
    )

    new_time = np.linspace(
        0,
        1,
        target_length
    )

    return np.interp(
        new_time,
        old_time,
        signal
    )


# ============================================================
# SYNC ANALYSIS
# ============================================================

def calculate_sync_score(
    audio_signal,
    video_signal
):

    if (
        audio_signal is None
        or video_signal is None
    ):

        return 0.0, 0

    # Match lengths
    target_length = max(
        100,
        min(
            len(audio_signal),
            len(video_signal)
        )
    )

    audio = resample_signal(
        audio_signal,
        target_length
    )

    video = resample_signal(
        video_signal,
        target_length
    )

    audio = normalize_signal(audio)
    video = normalize_signal(video)

    # Remove DC component
    audio = (
        audio - np.mean(audio)
    )

    video = (
        video - np.mean(video)
    )

    # --------------------------------------------------------
    # ZERO-LAG CORRELATION
    # --------------------------------------------------------

    audio_std = np.std(audio)
    video_std = np.std(video)

    if (
        audio_std < 1e-6
        or video_std < 1e-6
    ):

        return 0.0, 0

    zero_lag = np.corrcoef(
        audio,
        video
    )[0, 1]

    if np.isnan(zero_lag):

        zero_lag = 0

    # --------------------------------------------------------
    # LIMITED TEMPORAL SEARCH
    # --------------------------------------------------------
    #
    # We only allow a realistic lip-sync delay.
    # We DO NOT allow hundreds of arbitrary samples.

    correlation = correlate(
        audio,
        video,
        mode="full"
    )

    center = len(video) - 1

    # Maximum allowed lag:
    # approximately 1 second on each side

    max_lag = min(
        int(target_length * 0.10),
        50
    )

    start = max(
        0,
        center - max_lag
    )

    end = min(
        len(correlation),
        center + max_lag + 1
    )

    limited = correlation[
        start:end
    ]

    denominator = (
        np.linalg.norm(audio) *
        np.linalg.norm(video)
    )

    if denominator < 1e-8:

        return 0.0, 0

    limited = (
        limited /
        denominator
    )

    best_index = np.argmax(
        limited
    )

    best_correlation = (
        limited[best_index]
    )

    lag = (
        start +
        best_index -
        center
    )

    # --------------------------------------------------------
    # COMBINE ZERO-LAG AND BEST-LAG
    # --------------------------------------------------------

    best_correlation = float(
        np.clip(
            best_correlation,
            -1,
            1
        )
    )

    zero_lag = float(
        np.clip(
            zero_lag,
            -1,
            1
        )
    )

    # Convert correlation to 0-100
    best_score = (
        best_correlation + 1
    ) / 2

    zero_score = (
        zero_lag + 1
    ) / 2

    # Give more importance to actual
    # zero-lag synchronization.

    final_score = (
        0.65 * zero_score +
        0.35 * best_score
    )

    final_score *= 100

    return float(
        np.clip(
            final_score,
            0,
            100
        )
    ), int(lag)


# ============================================================
# MAIN DETECTOR
# ============================================================

def detect_lip_sync(video_path):

    audio_path = extract_audio(
        video_path
    )

    if audio_path is None:

        return {
            "status": "ERROR",
            "message": "No audio track found.",
            "score": 0,
            "lag": 0
        }

    try:

        # ----------------------------------------------------
        # AUDIO
        # ----------------------------------------------------

        audio_signal, audio_sr = (
            analyze_audio(
                audio_path
            )
        )

        # ----------------------------------------------------
        # VIDEO / MOUTH
        # ----------------------------------------------------

        video_signal, video_fps = (
            analyze_video(
                video_path
            )
        )

        if video_signal is None:

            return {
                "status": "ERROR",
                "message": (
                    "No detectable face or "
                    "mouth landmarks found."
                ),
                "score": 0,
                "lag": 0
            }

        # ----------------------------------------------------
        # SYNC SCORE
        # ----------------------------------------------------

        score, lag = (
            calculate_sync_score(
                audio_signal,
                video_signal
            )
        )

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        # More conservative threshold

        if score >= 60:

            result = (
                "MATCHED / GENUINE"
            )

            message = (
                "The audio activity and "
                "mouth movement show "
                "reasonable temporal "
                "alignment."
            )

        else:

            result = (
                "LIP-SYNC FRAUD / TAMPERED"
            )

            message = (
                "The audio and mouth movement "
                "show a significant temporal "
                "mismatch."
            )

        return {
            "status": "SUCCESS",
            "result": result,
            "score": round(
                score,
                2
            ),
            "lag": lag,
            "message": message
        }

    except Exception as e:

        return {
            "status": "ERROR",
            "message": str(e),
            "score": 0,
            "lag": 0
        }

    finally:

        if os.path.exists(
            audio_path
        ):

            os.remove(
                audio_path
            )