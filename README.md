# Plate Generator

Generates license plate images to be used for training object detectors. I used this to generate license plate images for training YOLOv5 license plate detectors because the license plate data sets I had were insufficient.

Unfortunately I don't think I'm allowed to upload the fonts here, meaning you won't be able to run this right away. But if anyone wants to use this and has any questions, just ask.


## Usage
```
usage: plate_generator.py [-h] [--plate PLATE] [--amount AMOUNT] --settings SETTINGS [--warp] [--debug] [--show_bbox] [--noise] [--dataset_folder DATASET_FOLDER] --output OUTPUT [--low-res] [--square] [--letterbox] [--blur] [--multi-thread] [--lines] [--adjust]

optional arguments:
  -h, --help            show this help message and exit
  --plate PLATE         Generate THIS specific plate
  --amount AMOUNT       How many plates to generate
  --settings SETTINGS   The path to the settings file
  --warp                Apply random warp to the image.
  --debug               Images will be shown and not saved. Use this for getting the character placement and boxes right.
  --show_bbox           Show bounding boxes when debug is enabled
  --noise               How much noise to apply. Default: 0
  --dataset_folder DATASET_FOLDER
                        The folder in which the dataset should be created
  --output OUTPUT       Where to write the data
  --low-res             Randomly make images lower resolution
  --square              Stretch images to be square
  --letterbox           Letterbox images to be square
  --blur                Apply blur of random intensity
  --multi-thread        Apply blur of random intensity
  --lines               Draw random lines over the images
  --adjust              Adjust boxes to better match the bounds of the characters
```

## Generate a specific plate and show it
`python plate_generator.py --plate "åäö-123" --settings /home/roboto/Documents/Dippen/tools/plate_generator/settings/autobabahn/basic_plate.yaml --dataset_folder /home/roboto/Documents/Dippen/data --output gentestlol --warp --debug --show_bbox`


## Generate 500 license plates using the settings found in settings/autobabahn/basic_plate.yaml
`python plate_generator.py --amount 500 --settings /home/roboto/Documents/Dippen/tools/plate_generator/settings/autobabahn/basic_plate.yaml --dataset_folder /home/roboto/Documents/Dippen/data --output gentestlol --warp --noise --blur --low-res --letterbox --lines --adjust --debug --show_bbox`
