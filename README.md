pip install numpy
pip install math 
pip install os
pip install matplotlib
pip install matplotlib.pyplot
pip install matplotlib.widgets
pip install matplotlib.figure 
pip install matplotlib.backends.backend_tkagg
pip install tkinter
pip install segyio
pip install pandas
pip install scipy
pip install dxfwrite
pip install utm
pip install csv
pip install ezdxf
pip install dxfwrite 
pip install utm


<<<<<<< HEAD
With the python language already installed, it can be executed with the command:
python tethys-X.py  (where X is the latest version of this repository)

=======

With the python language already installed, it can be executed with the command:
python tethys-X.py  (where X is the latest version of this repository)


If you prefer, you can convert this script into standalone executable files (.exe) on Windows:

Open a command prompt or terminal and run the following command to install PyInstaller:

pip install pyinstaller

Use the `cd` command to navigate to the directory where your Python script is located.

cd path\to\your\script

Run PyInstaller with the following command:

pyinstaller --onedir --noconsole --clean    tethys-X.py

Once PyInstaller has finished, you will find a `dist` directory in your script’s directory. 
Inside the `dist` directory, you will see the standalone executable file with the same name as your script but with a `.exe` extension.