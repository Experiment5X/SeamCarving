# Seam Carving
This technique can resize images by removing the pixels that are least important in the scene. The main parts of the image are still preserved during a resize. This idea was published in 2007 in [this article](https://dl-acm-org.ezproxy.rit.edu/citation.cfm?id=1276390).

Before

![Before](https://i.imgur.com/mMqzsjM.jpg)

After

![After](https://i.imgur.com/PrG0c8f.png)

## How it Works

The pixels of least importance are found by first computing the entropy for each pixel in the image. The higher the entropy, the more important the pixel is. The entropy is calculated by summing the difference in pixel intensity for the immediate neighbors. 

Now that the entropy has been calculated it is possible to find the pixels to remove. When cropping, only a single pixel is removed from each row/column so that the image remains rectangular and each of the pixels that are removed must be touching each other so that the content of the image remains intact. The goal at this step is to find a path or a seam of least entropy from one side of the image to the other. 

Below is an example of a seam calculated for the input image. All of these pixels would be removed during a crop.

![Horizontal Seam](https://i.imgur.com/KSsFs8J.png)
![Vertical Seam](https://i.imgur.com/a6VCLYG.png)

## Usage
This program allows you to crop an image in either direction by a certain amount of pixels. The seams can also be drawn onto the image instead of cropping them.

Install dependencies:
```bash
pip install -r requirements.txt
```

To crop an image 50 pixels using horizontal seams:
```bash
python3 seam_carving.py crop landscape.jpg -x -o h_cropped.png -p 50
```

To crop an image using vertical seams
```bash
python3 seam_carving.py crop landscape.jpg -y -o v_cropped.png -p 50
```

To draw the horizontal seam
```bash
python3 seam_carving.py draw landscape.jpg -x -o v_cropped.png -p 50
```

Full Usage
```bash
$ python3 seam_carving.py --help
usage: seam_carving.py [-h] [-p PIXEL_COUNT] [-o OUT_FILE] [-y] [-x] command

Compute and remove seams from images to crop them

positional arguments:
  command               draw|crop

optional arguments:
  -h, --help            show this help message and exit
  -p PIXEL_COUNT, --pixel-count PIXEL_COUNT
                        Number of pixels to remove in one direction
  -o OUT_FILE, --out-file OUT_FILE
                        The file to write the final picture to
  -y, --vertical-slices
                        Remove vertical slices from the image to crop its
                        width
  -x, --horizontal-slices
                        Remove horizontal slices from the image to crop its
                        height. This is the default
```