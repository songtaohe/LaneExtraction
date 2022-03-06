# Lane-Level Street Map Extraction from Aerial Imagery
## Abstract
Digital maps with lane-level details are the foundation of many applications. However, creating and maintaining digital maps especially maps with lane-level details, are labor-intensive and expensive. In this work, we propose a mapping pipeline to extract lane-level street maps from aerial imagery automatically. Our mapping pipeline first extracts lanes at non-intersection areas, then it enumerates all the possible turning lanes at intersections, validates the connectivity of them, and extracts the valid turning lanes to complete the map. We evaluate the accuracy of our mapping pipeline on a dataset consisting of four U.S. cities, demonstrating the effectiveness of our proposed mapping pipeline and the potential of scalable mapping solutions based on aerial imagery.


## View the dataset and annotate new images

Please check the instructions in [code/hdmapeditor](code/hdmapeditor). 

## Create training data

We have the raw dataset in the [dataset](dataset) folder. To train the models, we have to first create the necessary data, e.g., the lane segmentation.

```bash
cd code
python3 create_training_data.py
```

## Train the models

Train the lane-and-direction extraction model.

```bash
cd code/laneAndDirectionExtraction
python train.py resnet34v3
```

Train the turning lane validation model.

```bash
cd code/turningLaneExtraction
python train.py
```

Train the turning lane extraction model.

```bash
cd code/turningLaneExtraction
python train.py
```

