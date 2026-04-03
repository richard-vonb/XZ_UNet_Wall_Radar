# Radar Stud Detection — UNet Segmentation

## What This Does

Finds studs in walls using radar scans. A UNet takes 3D radar data, slices it into 200+ XZ cross-sections per scan, and predicts whether each pixel is "stud" or "not stud." More training data from fewer scans, faster iteration.

## Data & Labels

Radar returns a 3D voxel grid with 2 channels (amplitude + secondary). Labels are binary masks marking stud locations as rectangular slabs through the volume. Below: the XY amplitude view with ground-truth rectangles, and the full 3D label grid.

![XY Amplitude with Labels](Stud_Radar_Return_Visualization.png)
![3D Label Grid](Label_Example.png)

## How It Works

`StudUNet` is a lightweight 4-level UNet — 2-channel XZ slice in, 2-class segmentation map out. Trained with combined cross-entropy + Dice loss and a 5:1 class weight so the model doesn't just predict "no stud" everywhere and call it a day.

## Results

Model 1 nails crisp, confident stud columns. Model 2 gets the right locations but bleeds probability near the wall base. Three sample slices shown below.

![Y=100](Raw_vs__Prediction_Comparsion_y_100.png)
![Y=122](Raw_vs__Prediction_Comparsion_y_122.png)
![Y=180](Raw_vs__Prediction_Comparsion_y_180.png)
