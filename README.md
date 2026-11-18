# 🎭 Lip-Sync Fraud and Video-Audio Mismatch Detection

## 📌 Project Overview

The **Lip-Sync Fraud and Video-Audio Mismatch Detection System** is an AI-based application designed to identify potential inconsistencies between a speaker's lip movements and the corresponding audio in a video.

The system analyzes audio activity and visual mouth movements, calculates a synchronization score, and classifies the uploaded video as either **MATCHED / GENUINE** or **LIP-SYNC FRAUD / TAMPERED** based on a predefined threshold.

The application is developed using Python and Streamlit, providing a simple interface for uploading and analyzing videos.

## 🎯 Objectives

- Detect potential mismatches between video and audio.
- Analyze facial mouth movements using facial landmarks.
- Extract and analyze audio activity.
- Calculate an audio-video synchronization score.
- Display the detection result through an interactive dashboard.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| OpenCV | Video processing |
| MediaPipe Face Mesh | Facial and mouth landmark detection |
| Librosa | Audio signal analysis |
| NumPy | Numerical computations |
| MoviePy | Audio extraction from video |
| SciPy | Signal processing and correlation |

## ⚙️ How It Works

1. **Video Upload:** The user uploads a video in a supported format.
2. **Audio Extraction:** The audio track is extracted from the uploaded video.
3. **Audio Analysis:** Audio activity is analyzed using signal-processing techniques.
4. **Face and Mouth Detection:** MediaPipe Face Mesh identifies facial landmarks and tracks mouth movements.
5. **Synchronization Analysis:** Audio activity and mouth movement signals are compared to estimate their temporal alignment.
6. **Result Classification:** A synchronization score is calculated and compared against a 60% threshold.

## 📊 Detection Results

The application produces one of the following results:

- **MATCHED / GENUINE:** Synchronization score is greater than or equal to 60%.
- **LIP-SYNC FRAUD / TAMPERED:** Synchronization score is below 60%.

The dashboard displays:
- Synchronization Score
- Detected Lag
- Detection Status
- Analysis Explanation

## 📁 Project Structure

```text
lip_sync_fraud_detector/
│
├── app.py
├── lip_sync_detector.py
├── requirements.txt
└── README.md
```

## 💻 Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Folder

```bash
cd lip_sync_fraud_detector
```

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🚀 Usage

1. Open the application.
2. Upload a video containing a visible speaker.
3. Click **Analyze Video**.
4. Wait for the audio and visual analysis to complete.
5. View the synchronization score, detected lag, and classification result.

## ⚠️ Limitations

This project is a prototype based on audio activity and mouth-movement synchronization analysis. It does not establish whether a video is genuinely manipulated or authentic.

Results may be affected by background noise, silent speech, multiple speakers, poor video quality, facial visibility, and other recording conditions. The 60% threshold is a prototype classification threshold and should be validated using a representative dataset before real-world deployment.

## 🔮 Future Enhancements

- Improve synchronization analysis using advanced audio-video models.
- Evaluate the system using larger genuine and manipulated video datasets.
- Improve detection accuracy across different recording conditions.
- Add confidence calibration and more detailed visual explanations.
- Explore deep-learning-based lip-sync manipulation detection.

## 👩‍💻 Author

**Developed as an AI/ML project.**

