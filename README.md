# face.py

face.py is a Python package for simple way face landmarks extracting from [mediapipe](https://github.com/google/mediapipe/tree/master) tool.

### Preface

I created this tool to easily extract facial landmark coordinates.
The package allows quick access to the characteristic points of the eyes, irises, eyebrows and lips.
Additionally, two methods of the Face class allow you to check whether the eyes and mouth are open or closed.

### Potential usage of the package

**Winky**
The potential application of this tool immediately comes to mind. I continue to work on the keyboard controled
with eyes. The keyboard will allow to fully control computer as with a regular keyboard.

**BlinkBoard**
Another simple project is a communication keyboard for people with disabilities. By eyes blinking and mouth moving
selecting characters from the keyboard displayed on the monitor will be possible. This give a passibility to build entire sentences
which will enable a disabled person who cannot speak and move to have a conversation. The above work is in progress...

Another potential application is a sleep detector for car drivers. The detector will warn with a sharp signal
sound that the driver has his eyes closed.

### Action presentation

To demonstrate how face.py works, I created a small window application app.py.
You need a webcam for testing. This app shows all the current capabilities of Face.py packages.

### Dependencies

To use face.py, you need to install the [mediapipe](https://github.com/google/mediapipe/tree/master) package and its dependencies.
The easiest way is to install it using pip from the Windows console.

```
pip install mediapipe
```

Additionally, the OpenCV, Tkinter, pillow and numpy packages are needed to run the app.py test application.
OpenCV and numpy install automatically as mediapipe dependencies.
Tkinter is a package that supports window applications written in Python.

```
pip install opencv-python

pip install tk

pip install pillow
```

### License

Anyone interested can use this tiny tool under the MIT license. I don't know how this relates to the mediapipe package license.
I wish you fruitful work and interesting projects.