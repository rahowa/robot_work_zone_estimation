<p align="center"><img src=https://github.com/rahowa/robot_work_zone_estimation/blob/ar_base/assets/logo.png  alt="AWESOME LOGO" height="300"/></p>

# Robot workzone estimation
> An invariant to the camera position algorithm that helps to estimate the working areas of industrial robot.


## Table of Contents
- [Installation from source](#installation-from-source)
- [Start](#start)
- [Camera calibration](#camera-calibration)
- [Usage](#usage)
- [License](#license)

## Installation from source
> Clone repo to your local machine
```shell 
$ git clone https://github.com/rahowa/robot_work_zone_estimation.git
```

> Setup dependencies via pip
```shell
$ pip install -r requrements.txt
```

## Start
There are two possible ways to estimate workzone with markers. Thre first one is to use predifined Aruco markers (recommended way). Another way is to define your own custom marker and calibrate parameters for zone estimation (in progress).
### With Aruco markers
``` shell 
$ streamlit run configure.py
```

### With custom marker
``` shell 
$ streamlit run configure_custom.py
```

## Camera calibration
For better accuracy it's recommended to calibrate camera before doing marker pose estimation.

> If you already have images with calibration chessboards from your camera

``` shell
$ python calibrate_camera -p /path/to/images/ -x num_horizontal_squares -y num_vertical_saqures -m cam_height -w cam_width
```

> or if you want to create dataset and calibrate

``` shell
$ python calibrate_camera -t -p /path/to/images/ -x num_horizontal_squares -y num_vertical_saqures -m cam_height -w cam_width
```

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**