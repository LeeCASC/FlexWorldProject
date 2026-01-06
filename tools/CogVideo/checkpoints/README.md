---
license: mit
---

This model performs image-to-video generation based on the paper [FlexWorld: Progressively Expanding 3D Scenes for Flexible-View Synthesis](https://arxiv.org/abs/2503.13265).

Project page: https://ml-gsai.github.io/FlexWorld

Code: https://github.com/ml-gsai/FlexWorld

## Usage Example

A basic example of generating a static scene video given an image and a camera trajectory:

```bash
# You can utilize our CamPlanner class to freely construct the desired trajectory at line 13 in the `video_generate.py` file.
python video_generate.py --input_image_path ./assets/room.png --output_dir ./results-single-traj
```
