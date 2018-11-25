# XRayJigsawPuzzle
Used some code from https://github.com/bbrattoli/JigsawPuzzlePytorch

Learning image representations on unannotated Chest Xray images using the method described in Noroozi and Favaro to gain improvements in classification tasks. Here we they use the pretext task of solving jigsaw puzzles to pre-train the convolutional neural network. Chest x-ray images from the x-ray14 database were used.

![xray jigsaw](/docs/xray_doc.png "Xray Jigsaw")



Steps:

```
Run JigsawTrain.py and save the weights.

Run xNetTrain.py using the saved weights to fine tune the classification network.

```
