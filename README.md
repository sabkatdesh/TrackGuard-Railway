# TrackGuard-Railway
"Real-time passenger tracking system for revenue protection in rail systems" (Highlights both monitoring and financial safeguards)
This AI-powered system uses computer vision to count people entering and exiting train stations, designed to combat ticket fraud and corruption in railway systems.

Key Features
🚶 Real-time people counting using YOLOv8 object detection

🔄 DeepSORT for persistent object tracking

📐 Geometric line-crossing logic for accurate entry/exit detection

📊 Real-time visualization with OpenCV

⚙️ Configurable virtual lines and detection parameters

🎯 90%+ accuracy in real-world conditions

Railway-People-Counter/
├── main.py                    # Complete standalone implementation
├── sort.py                    # SORT tracker (required dependency)
├── requirements.txt           # Python dependencies
└── README.md                  # Documentation

📌 Overview
RailGuard BD is an intelligent computer vision system designed to combat ticket fraud and corruption in Bangladesh's railway stations. Using cutting-edge YOLOv8 object detection and DeepSORT tracking, it accurately counts passengers entering/exiting stations while filtering out non-passengers like well-wishers.

🛠️ Technical                              Stack
Component	                                 Technology
Object                                     Detection	YOLOv8
Object Tracking	                           DeepSORT
Computer Vision	                           OpenCV
Core Language	                             Python 3.8+
GPU Acceleration	                         CUDA (optional)
