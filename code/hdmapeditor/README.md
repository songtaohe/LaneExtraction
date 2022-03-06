# Annotate Lane Maps

Use main.py to annotate images,
```bash
python3 main.py image_file [pickle_file]
```
Here, *image_file* is the image file you need to annotate. The annotation result is stored in *pickle_file*. *pickle_file* is optional - if it is not provided, main.py creates a new pickle file based on the name of the *image_file*.

For example, to visualize the annotations in tile 0 of our dataset,
```bash
python3 main.py ../../dataset/sat_0.jpg
```

## Basic Keys

**'w,a,s,d'** pan the map.

**'1,2,3'** activate three zoom levels.

**'e'** switch between 'editing' mode and 'drawing' mode.

**'q'** switch the operation target between 'link' and 'way'. Here, ways correspond to the lanes at non-intersection area, and links correspond to the turning lanes at intersections. 

**'x'** toggle 'delete' mode.

**'r'** switch between 'drawing' mode and 'erase' mode. CAUTION: we don't support 'undo' for erase and delete modes yet.

**'4,5,6'** change the size of the 'eraser'. 

**'z'** undo.

**'m'** change layers.

**'f'** create bezier curves for turning lanes.

**'c'** select a turning lane.

**'ESC'** exit the editor.

## Basic Annotation Procedures

**Create a regular lane:** Make sure the editor is in 'ready_to_draw | way | Layer 0' status. Click on a sequence of locations to draw a polyline. Double click the last node to stop drawing.

**Create a turning lane:** Make sure the editor is in 'ready_to_draw | link | Layer 0' status. Click on a terminal node at the intersection. Draw a polyline for the turning lane. Click another terminal node to stop drawing. 

**Editing a node:** Switch to 'edit' mode. Click a node, and place the node at a new location by clicking it again.

**Deleting a node:** Switch to 'delete' mode. Click a node twice to delete. 









