# OpenCV Real-Time Streaming and Processing - Week 2

## Introduction

This project demonstrates real-time multi-stream video processing with motion detection and camera health monitoring using OpenCV.

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/Ishan1819/opencv-week2.git
   cd opencv-week2
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the program:
   ```bash
   python main.py
   ```

## Features

- Multi-Stream Viewer (2x2 grid for 4 streams)
- Motion Detection using background subtraction
- Camera Health Monitoring (blur/blocked detection)
- Real-time annotations on video
- Error handling for failed streams
- Runtime and stream status display

## Outputs

- 2x2 stream window with real-time motion alerts
- Console logs for stream status
- Camera compromised warnings on faulty feeds
