# UAV Deconfliction System

This project implements a drone deconfliction system that checks for spatiotemporal conflicts between a primary drone's flight path and other drones operating in a shared airspace. It provides automated conflict detection, visualization, and reporting.

---

## Project Structure

```
drone_deconfliction_sys/
├── data/
│   ├── primary_mission.json
│   └── simulated_flights.json
├── main.py
├── src/
│   ├── mission_parser.py
│   ├── conflict_checker.py
│   ├── visualizer.py              
│   ├── drone_plotly_viz.py        
│   └── test_conflicts.py
├── requirements.txt
├── README.md
└── reflection.md
```

---

## Getting Started

### 1. Clone and enter the project folder

```bash
git clone <your_repo_url>
cd uav_deconfliction
```

### 2. Set up virtual environment

```bash
python -m venv drone_env
drone_env\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 3. Run the system

```bash
python main.py
```

---

## Running Tests

This project includes a complete unit test suite using Python’s `unittest` framework.

### Testing covers:

- 3D distance calculation
- Waypoint interpolation logic
- Conflict detection for edge and typical cases
- Closest approach logic
- Safety threshold classification
- File loading validation
- Active drone filtering logic

### To run tests (with readable output):

```bash
python test_conflicts.py
```

---

## Dependencies

```txt
colorama==0.4.6
contourpy==1.3.2
cycler==0.12.1
fonttools==4.58.1
kiwisolver==1.4.8
matplotlib==3.10.3
narwhals==1.41.0
numpy==2.2.6
packaging==25.0
pandas==2.2.3
pillow==11.2.1
plotly==6.1.2
pyparsing==3.2.3
python-dateutil==2.9.0.post0
pytz==2025.2
scipy==1.15.3
six==1.17.0
tqdm==4.67.1
tzdata==2025.2
```

---


