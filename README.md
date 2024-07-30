# ALPR Project

This project focuses on Automatic License Plate Recognition (ALPR) data. Here are the steps we followed:

1. Combined the ALPR data from different regions into a single folder called "raw".
2. Formatted the label box coordinates to xyxy format and resized the images to 640x640.
3. Split the data into three sets: train, val, and test.
4. Saved the split data to files to work with the YOLO library.
5. Fine-tuned a YOLOv10 model on the data to predict only the bounding boxes for license plates.
6. Used the predicted bounding boxes to crop the original images.
7. Applied the TrOCR model to the cropped images to extract possible license plate numbers.

This pipeline allows us to accurately recognize license plates using ALPR data.
