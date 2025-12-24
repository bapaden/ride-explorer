# ride-explorer
Analyze FIT files from cycling activities

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Visualize a ride (opens a matplotlib window):

```bash
python3 analyze_ride.py --fit_file path/to/activity.fit --system_mass 85
```

Optional power-balance parameters can be supplied to visualize how drivetrain
efficiency and resistive coefficients affect the ride:

```bash
python3 analyze_ride.py --fit_file path/to/activity.fit --system_mass 85 --cda 0.32 --crr 0.004 --eta 0.97
```

Save the plot instead of showing it (useful for headless environments):

```bash
python3 analyze_ride.py --fit_file path/to/activity.fit --output ride.png --no-show --system_mass 85
```
