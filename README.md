# Radar Stud Detection — UNet Segmentation

## What This Does

Finds studs in walls using radar scans. Ultimate goal was differentiating 2x6s, 2x8s, and 2x10s. A UNet takes 3D radar data, slices it into 200+ XZ cross-sections per scan, and predicts whether each pixel is occupied by a stud. This allowed for faster computation and provided far more data for training. Served as an intital proof of concept, ultimately more data was needed with a greater diversity of stud placements.

## Data & Labels

Radar returns a 3D voxel grid with 2 channels (amplitude + secondary). Labels are binary masks marking stud locations as rectangular slabs through the volume. Below: the XY amplitude view with ground-truth rectangles, and the full 3D label grid.

![XY Amplitude with Labels](Stud%26Radar%20Return%20Visualization.png)
![3D Label Grid](Label%20Example.png)

## How It Works

This is a 4-level UNet, turns 2-channel XZ slice into 2-class segmentation. Trained with combined cross-entropy + Dice loss and a 5:1 class weighting.

## Results

The data/results are displayed on X and Z axes, which is a top view looking down the walls cavity. The first image of each set are the raw returned amplitude points, the remaining two images are a comparsion of the model with the best training loss, and the last epoch saved. Training took ~8 hours on laptop gpu. First 2 results are from a 3 stud scan, last is from a 2 stud scan.

![Y=100](Raw%20vs.%20Prediction%20Comparsion%20y%3D100.png)
![Y=122](Raw%20vs.%20Prediction%20Comparsion%20y%3D122.png)
![Y=180](Raw%20vs.%20Prediction%20Comparsion%20y%3D180.png)
