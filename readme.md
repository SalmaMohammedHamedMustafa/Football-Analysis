# SoccerVisionAI
# Soccer Video Tracking & Analytics

**A  modular pipeline for soccer analytics from video.**
 Combines efficient object detection/tracking, camera movement estimation, color-based team identification, and analytics including ball possession and speed/distance calculations , optimized for memory-efficient streaming.

## Table of Contents

- [SoccerVisionAI](#soccervisionai)
- [Soccer Video Tracking \& Analytics](#soccer-video-tracking--analytics)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Pipeline Flow](#pipeline-flow)
  - [Core Technologies](#core-technologies)
  - [Major Components / Class Breakdown](#major-components--class-breakdown)
  - [`Tracker`](#tracker)
  - [`TeamAssigner`](#teamassigner)
  - [`BallControlTracker`](#ballcontroltracker)
  - [`CameraMovementEstimator`](#cameramovementestimator)
  - [`ViewTransformer`](#viewtransformer)
  - [`SpeedAndDistance_Estimator`](#speedanddistance_estimator)
  - [Helper Modules](#helper-modules)
  - [Requirements](#requirements)
  - [Setup \& Installation](#setup--installation)
  - [File Structure](#file-structure)
  - [Customization](#customization)

## Project Overview

This project provides **full video analytics for soccer using a single camera feed**. The pipeline delivers:

- Ball, player & referee detection and tracking (with YOLOv8 + SORT-style tracking)
- Team assignment via jersey color clustering (unsupervised)
- Camera movement compensation using optical flow
- Real field (court) coordinate mapping via homography
- Ball possession stats (proximity-based)
- Speed and distance estimation (player & ball)
- Rich, live video annotation overlays (visualization ready)

It is **memory-efficient**: works in a streaming mode without loading the full video into RAM.

## Pipeline Flow

1. **Preprocessing & Initialization**
   - Load YOLOv8 model, set detection thresholds.
   - Prepare transformation matrices for pixel-to-court mapping.
   - Initialize camera movement and speed/distance modules.
2. **Detection and Tracking**
   - **Ball Detection:** YOLO runs in prediction mode on video, outputting ball bboxes per frame.
      Pandas-based interpolation fills in missing detections.
   - **Player/Referee Tracking:** YOLO in **track** mode for persistent IDs across frames.
3. **Camera Motion Compensation**
   - Optical flow estimates per-frame camera movement.
   - All positions/bboxes are adjusted accordingly for accurate statistics.
4. **Team Assignment**
   - On frames with enough players, unsupervised color clustering splits detected players into two teams.
5. **Possession & Analytics**
   - Ball control logic picks the nearest player per frame and determines possession, applying smoothing and change detection.
   - Speed/distance calculations leverage the best available coordinate space (court, camera-adjusted, or pixel).
6. **Annotation & Video Output**
   - Visual overlays:
     - Players by **team color** and unique track ID
     - Ball with special triangle marker; interpolated balls drawn in distinct color
     - Camera movement visualization (overlay)
     - Possession summary
   - Final annotated video output is streamed out frame by frame, compatible with standard media players.
7. **Comprehensive Logging & Reporting**
   - Console and terminal output at every major processing step: stats, warning flags, and summaries for detection quality, speed sanity, team identification, movement, etc.

## Core Technologies

- **YOLOv8 (via Ultralytics Python package):**
   For object detection/tracking (players, ball, referee), benefiting from best-in-class speed/accuracy.
- **OpenCV:**
   All image operations (I/O, color, drawing annotations, feature extraction, optical flow for camera movement).
- **scikit-learn (KMeans):**
   Unsupervised clustering for team recognition using jersey color histograms.
- **Numpy / Pandas:**
   Efficient array manipulation for tracking, interpolation, and measurements.

## Major Components / Class Breakdown

## `Tracker`

- **Main orchestrator**.
- Handles video reading, invoking YOLO for detection/tracking operations, fuses outputs, manages per-frame processing (camera movement, position adjustment, coordinate transformation).
- Includes methods for annotation, statistics, result caching/loading, and output.
- Coordinates all the submodules below.

## `TeamAssigner`

- **Assigns team identity by jersey color unsupervised clustering** (with KMeans).
- Extracts player region from detected bbox (focus on jersey/chest), converts section to feature vector, applies 2-cluster KMeans, and memorizes mapping of detected players to their team.
- Assigns each player a team ID and provides team color (for visual overlays).

## `BallControlTracker`

- **Calculates real-time ball possession**.
- For every frame, the closest player (within a threshold) is considered "in possession".
- Includes smoothing to minimize flicker and tracks possession changes, duration, and histories.
- Provides summary stats and overlay drawing for the annotated video.

## `CameraMovementEstimator`

- **Optical flow based camera shake/pan compensation.**
- Finds strong points to track on bordering image regions and, per frame, applies OpenCV’s Lucas-Kanade flow.
- Computes per-frame X/Y movement and applies this to all detection positions for true motion (player/ball/camera separation).

## `ViewTransformer`

- **Maps pixel positions to real-world (field/court) coordinates** via OpenCV’s perspective transform.
- Uses hardcoded or customizable source (image) and target (field) calibration points.
- Enables speed/distance in meters or km/h when possible.

## `SpeedAndDistance_Estimator`

- **Calculates speed (km/h) and cumulative distance (meters) for all tracked objects.**
- Applies filtering for noise, outliers, implausible jumps, and respects class- (player/ball/referee) specific motion thresholds.
- Prefers to use transformed (real-world) coords if possible, but can fall back to camera/pixel spaces.

## Helper Modules

- **`bbox_utils.py` / `my_utils.py`:**
   Functions for bounding box geometry (center, height, width, distance), helping all modules work together.

## Requirements

- Python 3.8+
- [opencv-python](https://pypi.org/project/opencv-python/)
- [ultralytics](https://pypi.org/project/ultralytics/) (for YOLOv8)
- [scikit-learn](https://scikit-learn.org/)
- numpy
- pandas

## Setup & Installation

1. **Clone the repository:**

   ```
   bashgit clone https://github.com/yourname/soccer-video-analytics.git
   cd soccer-video-analytics
   ```

2. **Install dependencies:**

   make sure to install install the correct CPU/GPU variant - open the requirements.txt for this

   ```
   bash
   pip install -r requirements.txt
   ```

3. **Add YOLO weights:**

   - Place the trained YOLOv8 weights (e.g., `best.pt`) in the root directory.

4. **Add your soccer video:**

   ---

   

## File Structure

| File                              | Purpose                                                      |
| --------------------------------- | ------------------------------------------------------------ |
| `main.py`                         | Entry point; runs tracker pipeline and configures I/O, thresholds |
| `tracker.py`                      | Pipeline hub: video I/O, detection, tracking, calls annotation, manages analytics |
| `team_assigner.py`                | KMeans-based team clustering, jersey color logic             |
| `ball_control_tracker.py`         | Ball possession determination, smoothing, indicator annotation |
| `camera_movement_estimator.py`    | Optical flow for camera movement compensation, feature extraction |
| `view_transformer.py`             | Field homography (pixel ↔ real-world meters)                 |
| `speed_and_distance_estimator.py` | Object (player, ball) speed/distance stats and filtering     |
| `bbox_utils.py` / `my_utils.py`   | Helper functions for bbox geometry, distances, coordinate transforms |

## Customization

- **YOLO model:**
   Train or fine-tune your own `.pt` weights for better ball/player classes.
- **Team assignment:**
   Adjust jersey area sampling in `team_assigner.py` (e.g., if teams' shorts are similarly colored).
- **Possession logic:**
   Edit distance and smoothing params in `ball_control_tracker.py`.
- **Speed/Distance:**
   Calibrate `view_transformer.py` pixel↔meter mapping for your field/camera setup.
- **Annotation:**
   Tweak drawing styles for improved visual clarity in overlays.