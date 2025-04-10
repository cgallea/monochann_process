


# monochann_process
Processing and Interpretation of Hi-Res Single-Channel Seismic Data
Using filter and tranforms, with csv export of mapped reflectors 

Tethys is a program for processing and interpreting high-resolution single-channel seismic data. It was developed to assist in the visualization and mapping of seismic reflectors from resonant (Chirp) and impulsive (Boomer, Sparker) source equipment.
From SEG-Y data, it allows the selection and testing of several filters for different analyses in search of the best solution for mapping reflectors in noisy environments. The best filtered results can be interpreted individually and superimposed on other filters, enabling the best seismic interpretation.
The mapped reflectors can be exported in vector format in DXF and CSF formats. The results of each filter can also be exported as an image for later illustrations.

To run the program we must first install the python dependencies:

pip install numpy

pip install math 

pip install os

pip install sys

pip install matplotlib

pip install matplotlib.pyplot

pip install matplotlib.widgets

pip install matplotlib.backends.backend_tkagg

pip install tkinter

pip install segyio

pip install pandas

pip install scipy

pip install dxfwrite

pip install utm


With the python language already installed, it can be executed with the command:
python tethys-X.py  (where X is the latest version of this repository)



If you prefer, you can convert this script into standalone executable files (.exe) on Windows:

Open a command prompt or terminal and run the following command to install PyInstaller:

pip install pyinstaller

Use the `cd` command to navigate to the directory where your Python script is located.

cd path\to\your\script

Run PyInstaller with the following command:

pyinstaller --onefile your_script.py

Once PyInstaller has finished, you will find a `dist` directory in your script’s directory. 
Inside the `dist` directory, you will see the standalone executable file with the same name as your script but with a `.exe` extension.


