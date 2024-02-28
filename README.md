# Semantic segmentation model for 4 class: green, water, non-green, cloud.

## Usage

```
python main.py --weight_path <path to model weights>
--image_path <path to input image>
--save_path <path to save output image>
--strength <float value, amplifier for small region>
```
or use the default value with all the file is in the corresponding folder and strength = 3
```
python main.py
```

## Detail

Input:

+ 4 band color image (r, g, b, nir) of size >= (512, 512)

Output:

+ saved prediction image with 4 possible value:

  + 0: Green
  + 1: Water
  + 2: Non-green
  + 3: Cloud

Free parameter:

- Amplifier strength (default to 3.0): positive or negative float value to amplify class signal for small region, bigger mean more small region appearing.

# Architecture used:

+ A U2Netlite model as the main architecture, trained on the original and a dilated version of each label to predict both label.

+ A small pretrained pixel-based model to process color signal with its output directly feed into each RSU block.

+ 2 convolutional layer of size 9x9x4 after the predicted dilated label.

+ Note: The architecture as in the image have 8 input channel as it was designed to have 2 separated 4 channel input: the original image to feed into the pixel based color model,
and an augmented image learn to feed into the U2net to learn the context with the color signal already exist.

![AT_architecture](https://github.com/curiosity29/Green-Cover/assets/62107278/706f3898-122c-425f-9bb8-891f50cc5d70)
