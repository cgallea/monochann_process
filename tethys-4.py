import numpy as np
import math
import os, sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button as Btn
from matplotlib.widgets import Slider
from matplotlib.widgets  import RectangleSelector
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import segyio
import pandas as pd

from scipy import signal
from scipy import interpolate
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import make_interp_spline
from scipy.ndimage import shift
from scipy import ndimage
import scipy.linalg as la
from scipy.linalg import toeplitz
from scipy.signal.windows import triang
from scipy.signal import convolve2d as conv2
from scipy.fft import fft, ifft, fftfreq

from skimage import img_as_float
from skimage import exposure
from skimage.exposure import rescale_intensity
from skimage import filters

from dxfwrite import DXFEngine as dxf
import utm

from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter import messagebox

# from tkdocviewer import *
import PyPDF2

#===============================================================

def select(window, px, py):

        def check_cbox(event):
            
            def assig(var1, var2, var3, var4, var5):
                global sigdata, sigcmin, sigcmax, sigpal, sigfilt

                sigdata = var1  
                sigcmin = var2
                sigcmax = var3
                sigpal = var4
                sigfilt = var5
                
            try:

                if cbox.get() == 'Raw Data':
                    filt = ' over Raw Data'
                    assig(sig, cmin, cmax, palet, filt)

                if cbox.get() == 'Band Pass':
                   filt = ' over Band Pass'
                   assig(BP, cminBP, cmaxBP, palet, filt)

                if cbox.get() == 'Spectr Whiten':
                    filt = ' over Spectral Whitening'
                    assig(WT, cminWT, cmaxWT, palet, filt) 

                if cbox.get() == 'Parm Equaliz':
                    filt = ' over Parametric Equalizer' 
                    assig(EQ, cminEQ, cmaxEQ, palet, filt)

                if cbox.get() == 'AGC':
                    filt = ' over AGC'
                    assig(AGC, cminAGC, cmaxAGC, palet, filt)
                    
                if cbox.get() == 'Manual Gain':
                    filt = ' over Manual Gain'
                    assig(MG, cminMG, cmaxMG, palet, filt)

                if cbox.get() == 'STA/LTA':
                    filt = ' over STA/LTA'
                    assig(STA, cminSTA, cmaxSTA, palet, filt)

                if cbox.get() == 'Chirp Ricker':
                    filt = ' over Ricker Convolution'
                    assig(RIK, cminRIK, cmaxRIK, palet, filt)

                if cbox.get() == 'Deconvolution':
                    filt = ' over Deconvolution'
                    assig(bDEC, cminbDEC, cmaxbDEC, palet, filt)

                if cbox.get() == 'Envelope (Inst Amplit)':
                    filt = ' over Envelope (Instant Amplitude)'
                    assig(IA, cminIA, cmaxIA, palet, filt)

                if cbox.get() == 'Inst Frequency':
                    filt = ' over Instant Frequency'
                    assig(IF, cminIF, cmaxIF, palet, filt)

                if cbox.get() == 'Phase Cosine':
                    assig(PC, cminPC, cmaxPC, palet, filt)
                    filt = ' over Phase Cosine'

                if cbox.get() == 'Laplace':
                    filt = ' over Laplace filter'
                    assig(LAP, cminLAP, cmaxLAP, palet, filt)

                if cbox.get() == 'x-Deriv':
                    filt = ' over x-Derivative'
                    assig(DX, cminDX, cmaxDX, palet, filt)

                if cbox.get() == 'y-Deriv':
                    filt = ' over y-Derivative'
                    assig(DY, cminDY, cmaxDY, palet, filt)

                if cbox.get() == 'Wiener':
                    filt = ' over Wiener filter'
                    assig(WN, cminWN, cmaxWN, palet, filt)

            except Exception as e:
                messagebox.showerror("Error", "filter not defined")

        options = ['Raw Data','Band Pass', 'Spectr Whiten','Parm Equaliz','AGC','Manual Gain','STA/LTA','Chirp Ricker',
            'Boom Deconv','Inst Amplit','Inst Freq','Phase Cos','Laplace','x-Deriv','y-Deriv','Wiener']

        default_option = StringVar(value=options[0])

        in_data = ''
        # cbox = ttk.Combobox(window, textvariable=default_option, state="readonly", width=13)
        cbox = ttk.Combobox(window, textvariable=in_data, state="readonly", width=13)
        cbox['values'] = options
        cbox['state'] = 'readonly' 
        cbox.place(x=px, y=py) 
        cbox.bind('<<ComboboxSelected>>', check_cbox)
        lbl1 = Label(window, text="Select Data")
        lbl1.place(x=px+15, y=py-22)

# ===================================================================================================

def pal_choice(event):  # divergent pallets
    global palet

    if combox1.get() == 'Greys':
        palet = combox1.get() 
    if combox1.get() == 'seismic':
        palet = combox1.get()
    if combox1.get() == 'bwr':
        palet = combox1.get()
    if combox1.get() == 'coolwarm':
        palet = combox1.get() 
    if combox1.get() == 'Spectral':
        palet = combox1.get()
    if combox1.get() == 'PuOr':
        palet = combox1.get()
    if combox1.get() == 'BrBG':
        palet = combox1.get() 
    if combox1.get() == 'PRGn':
        palet = combox1.get()     
    if combox1.get() == 'RdBu':
        palet = combox1.get()

#===================================================================================================

def Norm(dat):
    return (dat - np.min(dat)) / (np.max(dat) - np.min(dat))

# ===================================================================================================

def pdfReader(file):
    help = Toplevel(root)
    help.geometry("750x450")

    text_help = Text(help, width= 80, height=30)
    text_help.pack(pady=20)
    #Open the PDF File
    pdf_file = PyPDF2.PdfFileReader(file)
    #Select a Page to read
    page = pdf_file.getPage(0)
    #Get the content of the Page
    content = page.extractText()
    #Add the content to TextBox
    text_help.insert(END, content)

# ===================================================================================================

def hilb(hdata, samp_rate):    # Hilbert   - scipy.signal.hilbert

    fs = 1/samp_rate  # frequency
    z= signal.hilbert(hdata) #form the analytical signal
    i_a = np.abs(z) #envelope extraction
    i_p = np.unwrap(np.angle(z))#inst phase
    i_f = np.diff(i_p)/(2*np.pi)*fs #inst frequency    
    
    return i_a, i_f, i_p

# ===================================================================================================

def classic_sta_lta_py(a, nsta, nlta):
    """
    Computes the standard STA/LTA from a given input array a. The length of
    the STA is given by nsta in samples, respectively is the length of the
    LTA given by nlta in samples. Written in Python.

    https://docs.obspy.org/packages/obspy.signal.html

    .. note::

        There exists a faster version of this trigger wrapped in C
        called :func:`~obspy.signal.trigger.classic_sta_lta` in this module!

    :type a: NumPy :class:`~numpy.ndarray`
    :param a: Seismic Trace
    :type nsta: int
    :param nsta: Length of short time average window in samples
    :type nlta: int
    :param nlta: Length of long time average window in samples
    :rtype: NumPy :class:`~numpy.ndarray`
    :return: Characteristic function of classic STA/LTA
    """
    # The cumulative sum can be exploited to calculate a moving average (the
    # cumsum function is quite efficient)
    sta = np.cumsum(a ** 2, dtype=np.float64)

    # Copy for LTA
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[nsta:] = sta[nsta:] - sta[:-nsta]
    sta /= nsta
    lta[nlta:] = lta[nlta:] - lta[:-nlta]
    lta /= nlta

    # Pad zeros
    sta[:nlta - 1] = 0

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny

    return sta / lta

# ===================================================================================================

def whiten(X, method='zca'):
    """
    Whitens the input matrix X using specified whitening method.
    Inputs:
        X:      Input data matrix with data examples along the first dimension
        method: Whitening method. Must be one of 'zca', 'zca_cor', 'pca',
                'pca_cor', or 'cholesky'.

       Joel Marino -  http://joelouismarino.github.io/
    """
    X = X.reshape((-1, np.prod(X.shape[1:])))
    X_centered = X - np.mean(X, axis=0)
    Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
    W = None
    
    if method in ['zca', 'pca', 'cholesky']:
        U, Lambda, _ = np.linalg.svd(Sigma)
        if method == 'zca':
            W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))
        elif method =='pca':
            W = np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T)
        elif method == 'cholesky':
            W = np.linalg.cholesky(np.dot(U, np.dot(np.diag(1.0 / (Lambda + 1e-5)), U.T))).T
    elif method in ['zca_cor', 'pca_cor']:
        V_sqrt = np.diag(np.std(X, axis=0))
        P = np.dot(np.dot(np.linalg.inv(V_sqrt), Sigma), np.linalg.inv(V_sqrt))
        G, Theta, _ = np.linalg.svd(P)
        if method == 'zca_cor':
            W = np.dot(np.dot(G, np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T)), np.linalg.inv(V_sqrt))
        elif method == 'pca_cor':
            W = np.dot(np.dot(np.diag(1.0/np.sqrt(Theta + 1e-5)), G.T), np.linalg.inv(V_sqrt))
    else:
        raise Exception('Whitening method not found.')

    return np.dot(X_centered, W.T)

# ===================================================================================================

class ManualGain(object):

    def __init__(self, fig, ax1, ax2, trc):
        self.xs = []
        self.ys = []
        self.ax1 = ax1
        self.ax2 = ax2
        self.fig = fig
        self.trc = trc
        self.init = True
        self.i = -1
        self.Y = []
        self.escap = True
        
    def mouse_click(self, event):

        self.escap = False # escape nao pressionado ainda
        if not event.inaxes:
            return
        
        #left click
        if event.button == 1:
            if self.init == True:
                self.xs.append(event.xdata)
                self.ys.append(event.ydata)
                self.i+= 1
                self.init = False
            else:
                self.xs.append(event.xdata)
                self.ys.append(event.ydata)                
                line, = self.ax1.plot([self.xs[self.i], self.xs[self.i+1]], [self.ys[self.i], self.ys[self.i+1]], 
                color='r', marker="o")
                line.figure.canvas.draw_idle()
                self.i+= 1

        #right click
        if event.button == 3:
            if len(self.xs) > 0:
                self.xs.pop()
                self.ys.pop()
            #delete last line drawn if the line is missing a point,
            #never delete the original stock plot
            if len(self.xs) % 2 == 1 and len(self.ax.lines) > 1:
                self.ax1.lines.pop()
            #refresh plot
            self.fig.canvas.draw_idle()

    def on_key(self, event):
        
        if event.key == 'escape':

            if self.escap == False:
                # interpolates the stroke between the vertices of the drawn line 
                if self.trc[0] < self.xs[0]: 
                    self.xs.insert(0, 0)
                    self.ys.insert(0, self.trc[0])
                if self.trc[-1] > self.xs[-1]:
                    self.xs.append(len(self.trc))
                    self.ys.append((self.trc[-1]))
                interv = len(self.trc)
                xnew = np.linspace(self.xs[0], self.xs[-1], interv)
                fint = interpolate.UnivariateSpline(self.xs, self.ys)
                y = fint(xnew)
                for n1, n2 in zip(y, self.trc):
                    self.Y.append((n1) * n2)
                self.ax2.plot(self.Y, color='g')
                # plota traco interpolado
                self.fig.canvas.draw_idle()
                self.escap = True  

        if event.key == 'x':
            plt.cla(self.ax1) # erase drawn line
            plt.cla(self.ax2) # delete interpolated trace
            self.fig.canvas.draw_idle()

# ===================================================================================================

def img_show(filename, txt, im, tri, trf, spi, spf, sr, v_min, v_max, colrs, filt, trace_index):
    global ext

    def showfreq(event):

        t = (trf-tri)//2

        # mpl.rcParams['toolbar']= 'None' 
        fig = plt.figure (figsize=(10,2), constrained_layout=True)
        plt.suptitle('Trace '+str(t), fontsize=11)

        ax1 = fig.add_subplot(121)

        if len(sigdata) == 0:
            Xf_mag = np.abs(np.fft.fft(im[t]))
        else:
            Xf_mag = np.abs(np.fft.fft(sigdata[t]))

        freqs = np.fft.fftfreq(len(Xf_mag), d=sr)
        ax1.plot(abs(freqs), Xf_mag)

        ax1.set_xlabel('Hz')
        ax1.set_title('Spectrum Before', fontsize=11)


        ax2 = fig.add_subplot(122)   #, sharey=ax1)

        i = (trf-tri)//2
        Xf_mag = np.abs(np.fft.fft(im[t]))
        freqs = np.fft.fftfreq(len(Xf_mag), d=sr)
        ax2.plot(abs(freqs), Xf_mag)

        ax2.set_xlabel('Hz')
        ax2.set_title('Spectrum After', fontsize=11)

        # multi = MultiCursor(None, (ax1, ax2), useblit=False, color='r', lw=1)
        plt.show(block=False)

    # ----------------------------------------------------------------------------

    fig, ax1 = plt.subplots(figsize=(16, 8))   
    tmi = spi*sr*1000
    tmf = spf*sr*1000
    twt = np.linspace(tmi, tmf)         # array dos tempos
    ext = [tri, trf, twt[-1], twt[0]]   # define extent
    ax1.set_xlabel('Trace number', fontsize=9)
    ax1.set_ylabel('TWT [ms]', fontsize=9)
    plt.grid(color='gray')

    # ----------------------------------------------------------------------------

    if all(x != 0 for x in trace_index):
        
        tindex_ini = trace_index[tri] 
        tindex_end = trace_index[trf-1]    

        ax1_smp = ax1.twiny()  
        ax1_smp.set_xlabel('TraceIndex', fontsize=9)

        ax1_smp.set_xlim(tindex_ini, tindex_end) 
        ax1_smp.set_ylim(twt[-1], twt[0])  

    # ----------------------------------------------------------------------------

    ax1_smp = ax1.twinx()  
    ax1_smp.set_ylabel('Samples', fontsize=9)

    ax1_smp.set_xlim(tri, trf)
    ax1_smp.set_ylim(spf, spi)

    ax1_smp.figure.canvas.draw()
    ax1.set_title(f'{filename}'+'  -  '+txt+filt, fontsize=9)

    # ----------------------------------------------------------------------------

   
    ax1_smp.figure.canvas.draw()
    ax1.set_title(f'{filename}'+'  -  '+txt+filt, fontsize=10)


    vm = np.percentile(im, 95) # im is the filtered image
    img = ax1.imshow(im.T, vmin=-vm, vmax=vm, cmap=colrs, aspect='auto', interpolation='nearest', extent=ext)

    plt.subplots_adjust(top=0.9, bottom=0.08, left=0.04, right=0.91) # define chart area
    #axcolor = 'lightgoldenrodyellow'    
    ax_clr_min = plt.axes([0.96, 0.1, 0.01, 0.8]) # position of the sliders in the created area
    ax_clr_max = plt.axes([0.98, 0.1, 0.01, 0.8])
    s_clr_min = Slider(ax_clr_min, '', v_min, 0, orientation='vertical') # put the slides 
    s_clr_max = Slider(ax_clr_max, '', 0, v_max, orientation='vertical')
 
    showbtn = plt.axes([0.83, 0.94, 0.1, 0.05])   # define button area
    # put the matplotlib button in that area
    btn = Btn(showbtn, 'Show Frequencies', hovercolor='gold')

    btn.on_clicked(showfreq)
    
    cmin = -vm
    cmax = vm
    clr_min = v_min
    clr_max = v_max
    
    def store(C1, C2):
        clr_min = C1
        clr_max = C2
        return clr_min, clr_max
    
    def update(val, s=None):
        global cmin, cmax

        _clr_min = s_clr_min.val # get the cmin value from the slider
        _clr_max = s_clr_max.val # get the cmax value from the slider
        img.set_clim(_clr_max) # cmin, cmax arrow in the image
        img.set_clim([_clr_min, _clr_max]) 
        cmin, cmax = store(_clr_min, _clr_max) # store _clr_min in cmin and _clr_max in cmax
        plt.draw()  # redraws the image with new cmin and cmax
        return cmin, cmax

    s_clr_min.on_changed(update)
    s_clr_max.on_changed(update) 

    plt.show() 
    plt.close()

    return cmin, cmax, colrs

# ===================================================================================================

def prof_viewer(tr, spi, spf, sign): 

    def times(x):
        return x*sr*1000
    def samples(x):
        return x

    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(spf-spi)
    y = sign   # plot a trace
    plt.plot(x, y, label=segyfile + ' - Trace '+str(tr))
    plt.hlines(0, 0, ns, color='k', linestyles='dashdot')
    ax.set_xlabel('Samples', fontsize=9)
    ax.set_ylabel('Signal', fontsize=9)
    ax.legend()
    secax = ax.secondary_xaxis('top', functions=(times, samples))
    secax.set_xlabel('Time (ms)', fontsize=9)
    plt.grid()
    plt.show()

def plot_perf(): 
    tr = nt//2 # middle trace
    prof_viewer(tr, 0, ns, data[tr])  

# ===================================================================================================

def extract_coords(file):
    with segyio.open(file, mode='r+', ignore_geometry=True) as f:
        n_traces = f.tracecount
        #sample_rate = segyio.tools.dt(f) / 1000000
        headers = segyio.tracefield.keys
        # Initialize pd.DataFrame with trace id as index and headers as columns
        trace_headers = pd.DataFrame(index=range(1, n_traces + 1), columns=headers.keys())
        # Fill pd.DataFrame with all header values
        for k, v in headers.items():
            trace_headers[k] = f.attributes(v)[:]
    return trace_headers

# ===================================================================================================

def butter_bandpass(lowcut, highcut, fs, order=5):

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

# ===================================================================================================

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

# ===================================================================================================  
  
def save_segy(imgfile, filtername):     #Save the current filter as a new file

    outfile = filepath[:-4]+'_'+filtername+str('.sgy')

    segyio.tools.from_array2D(outfile, imgfile, 189, 193, 1, sr*1000000, 0)

    # with segyio.open(filepath, ignore_geometry=True) as src:
    #     spec = segyio.tools.metadata(src)
    #     spec.samples = spec.samples[:]
    #     with segyio.create(outfile, spec) as dst:
    #         dst.text[0] = src.text[0]
    #         dst.bin = src.bin
    #         dst.bin.update(hns=len(spec.samples))
    #         dst.header = src.header
    #         dst.trace = imgfile[:]

# ===================================================================================================

def parse_text_header(file):
    '''
    Format segy text header into a readable, clean dict
    '''
    raw_header = segyio.tools.wrap(file.text[0])
    # Cut on C*int pattern
    cut_header = np.split(r'C ', raw_header)[1::]
    # Remove end of line return
    text_header = [x.replace('\n', ' ') for x in cut_header]
    text_header[-1] = text_header[-1][:-2]
    # Format in dict
    clean_header = {}
    i = 1
    for item in text_header:
        key = "C" + str(i).rjust(2, '0')
        i += 1
        clean_header[key] = item
    return clean_header

# ===================================================================================================    

def open_segy_file():

    btn_corrections['state']='disabled'
    btn_freq_filters['state']='disabled'
    btn_ampl_filters['state']='disabled'
    btn_conv_deconv['state']='disabled'
    btn_imag_filters['state']='disabled'
    btn_interp['state']='disabled'

    global data, nt, ns, sr, twt, coords, head, datatype, spec, f, trace_index
    global filepath, segyfile, filename, no_coords, sigdata, logfile

    # clear all vars
    data = None
    nt = None
    ns = None
    sr = None
    twt = None
    coords = None
    head = None
    datatype = None
    sigdata = None

    # select and open file

    filepath = askopenfilename(filetypes=[("SEG-Y", "*.sgy"), ("SEG files", "*.seg"), ("All Files", "*.*")])    
    if not filepath:
        return

    segyfile = os.path.basename(filepath)
    filename = segyfile[:-4]

    with segyio.open(filepath, ignore_geometry=True) as f:
        # Get basic attributes
        n_traces = f.tracecount
        sample_rate = segyio.tools.dt(f) / 1000000
        n_samples = f.samples.size
        twt = f.samples
        data = f.trace.raw[:]  # Get all data into memory (could cause on big files)

        spec = segyio.tools.metadata(f)
        spec.samples = spec.samples[:]

        # Get all header keys
        headers = segyio.tracefield.keys 
        # Initialize pd.DataFrame with trace id as index and headers as columns
        trace_headers = pd.DataFrame(index=range(1, n_traces + 1), columns=headers.keys())
        # Fill pd.DataFrame with all header values
        for k, v in headers.items():
            trace_headers[k] = f.attributes(v)[:]
        trace_heads = extract_coords(filepath)
        coord = trace_heads[['TRACE_SEQUENCE_LINE','SourceX','SourceY', 'LagTimeA', 'ElevationScalar']]
        coords = coord.rename(columns={'TRACE_SEQUENCE_LINE': 'Trace'})
        c0 = coords['Trace'][1]
        coords['Trace'] = coords['Trace']-c0 

        trace_index = trace_headers.FieldRecord.values

        head = segyio.tools.wrap(f.text[0])
        nt = n_traces
        ns = n_samples
        sr = sample_rate
        sig = np.copy(data)

        raw_header = segyio.tools.wrap(f.text[0])

        txt_edit.insert(END, '\t EBCDIC header\n')
        txt_edit.insert(END, raw_header)
        txt_edit.insert(END, '\n')

        txt_edit.insert(END, '\n\t File name: '+segyfile)
        txt_edit.insert(END, '\n\t Number of traces: '+str(nt))
        txt_edit.insert(END, '\n\t Number of samples per trace: '+str(ns))
        txt_edit.insert(END, '\n\t Sample rate: '+str(sr*1000000)+' microseconds')
        txt_edit.insert(END, '\n\t Frequency: '+str(int(1/sr))+' Hz\n\n\n')
        txt_edit.insert(END, f.bin)

    # --------------------------------------------------------------------------------

    if min(data[10]) < 0:
        datatype = 0   # analitic data
        txt_edit.insert(END, '\n')
        txt_edit.insert(END, '\n Analitic Data')
    else:
        datatype = 1     # enveloped data
        txt_edit.insert(END, '\n')
        txt_edit.insert(END, '\n Enveloped Data')

    if coords['SourceX'][10] == 0: 
        no_coords = True
        txt_edit.insert(END, '\n')
        txt_edit.insert(END, '\n Coordinates are Missing')
    else: 
        no_coords = False    

    # --------------------------------------------------------------------------------

    def img_view():
        global tmi, tmf, tr, tri, trf, spi, spf, sig, sigdata, sig1d, cmin, cmax
        
        tri = e_tri.get()
        if tri == '' :
            tri = 0
        else:
            tri = int(tri)

        trf = e_trf.get()
        if trf == '' :
            trf = nt
        else:
            trf = int(trf)

        tmi= e_tmi.get()
        if tmi == '':
            tmi = 0
        else:
            tmi = int(tmi)

        tmf = e_tmf.get()
        if tmf == '':
            tmf = ns*sr*1000
            tmf = int(tmf)
        else:    
            tmf = int(tmf)
            
        spi = int(tmi /sr /1000 )  # initial sample
        spf = int(tmf /sr /1000 )  # final smaple
        tr = int((trf-tri)/2)

        sig = data[tri:trf, spi:spf]   # set working file in selected window
        sigdata = sig                  # sigdata is used in select data and defines the chosen image
        sig1d = np.ravel(sig)

        vm = np.percentile(sig, 99)   # get the percentile from the raw data to start the viewer
        cmin = -vm
        cmax = vm

        tit = 'Working Window over Raw Data'
        sigfilt = ''
        cmin, cmax, cmap = img_show(segyfile, tit, sig, tri, trf, spi, spf, sr, cmin, cmax, palet, sigfilt, trace_index)
        
        btn3['state'] = 'normal' 

    # --------------------------------------------------------------------------------

    def save_limits():
        global scenes
 
        scenes = pd.DataFrame({'img':[],'cmin':[],'cmax':[],'tit':[],'sigfilt':[],'cmap':[]}, dtype=object)
        tit = 'Raw Signal'
        sigfilt = ''
        cmap = palet

        scene = [sigdata, cmin, cmax, tit, sigfilt, cmap]
        scenes.loc[len(scenes)] = scene

        txt_edit.insert(END, '\n')
        txt_edit.insert(END, '\n Working Window: ')
        txt_edit.insert(END, '\n Start Time (ms): '+ str(tmi))
        txt_edit.insert(END, '\n End Time (ms): '+ str(tmf))
        txt_edit.insert(END, '\n Start Sample: '+ str(spi))
        txt_edit.insert(END, '\n End Sample: '+ str(spf))
        txt_edit.insert(END, '\n Start Trace: '+ str(tri))
        txt_edit.insert(END, '\n End Trace: '+ str(trf))
        txt_edit.insert(END, '\n')

        btn_corrections['state'] = 'normal'
        btn_freq_filters['state'] = 'normal'
        btn_ampl_filters['state'] = 'normal'
        btn_conv_deconv['state'] = 'normal'
        btn_imag_filters['state'] = 'normal'
        btn_interp['state'] = 'normal'
        btn_naveg['state'] = 'normal'

        work_window.destroy()

    # --------------------------------------------------------------------------------

    filelog = filename+'.log'
    logfile = open(filelog, 'a')    # create log file
    logfile.write('file opened')

    txt_edit.insert(END, '\n')
    txt_edit.insert(END, '\n '+filename+'.log as created')
    txt_edit.insert(END, '\n')
   
    work_window = Toplevel(root)
    work_window.title("WORKING WINDOW")
    # work_window.iconbitmap("logotet.ico")
    w = 310 # window width
    h = 250 # window height
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    work_window.geometry('%dx%d+%d+%d' % (w, h, x, y))
    work_window.resizable(width=False, height=False)
    work_window.attributes('-toolwindow', True)

    CheckVar = IntVar(value=0)

    lbl1 = Label(work_window, text="Start Time (ms): ")
    lbl1.place(x=70, y=30)
    e_tmi = Entry(work_window, width=10)
    e_tmi.place(x=170, y=30)
    e_tmi.focus()

    lbl2 = Label(work_window, text="End Time (ms): ")
    lbl2.place(x=70, y=50)
    e_tmf = Entry(work_window, width=10)
    e_tmf.place(x=170, y=50)

    lbl3 = Label(work_window, text="Start Trace: ")
    lbl3.place(x=70, y=70)
    e_tri = Entry(work_window, width=10)
    e_tri.place(x=170, y=70)

    lbl4 = Label(work_window, text="End Trace: ")
    lbl4.place(x=70, y=90)
    e_trf = Entry(work_window, width=10)
    e_trf.place(x=170, y=90)

    tr = nt//2
    sigdata = []

    btn1 = Button(work_window, width=12, text ='Image View', command= img_view)
    btn1.place(x=50, y=140)

    btn2 = Button(work_window, width=12, text='Trace View', command= plot_perf)
    btn2.place(x=165, y= 140)

    btn3 = Button(work_window, width=10, text='Save', command= save_limits)
    btn3.place(x=115, y= 180)

    btn3['state'] = 'disable'                         

# ===================================================================================================

def correction():

    def swell_filter():

        def view_img():

            vm = np.percentile(sig, 95)
            cmin = -vm
            cmax = vm
            sigfilt = '' 
            tit = ' - Select Start Time for Bottom Search'

            cmin, cmax, cmap = img_show(segyfile, tit, sigdata, 0, nt, 0, ns, sr, cmin, cmax, palet, sigfilt, trace_index)

        def apply():
            global sigcor, correct, nt, sigdata

            def moving_average(x, w):
                return np.convolve(x, np.ones(w), 'same') / w

            btn3['state'] = 'normal'
            btn4['state'] = 'normal'

            ini_scan  = int(e_ini.get())  # start scan in ms
            smp_ini = int(ini_scan/1000/sr) # start scan in samples

            n_sta = int(e_sta.get())
            n_lta = int(e_lta.get())

            win = int(e_win.get()) # win size for moving average

            # extracts Smax (SAMPLE NUMBER) from the trace
            def extractSmax(trace, sta, lta):

                I = classic_sta_lta_py(trace, sta, lta)
                Ymax = max(I)
                I1 = I.tolist()
                Smax = I1.index(Ymax)
                return Smax
                
            Smax = []

            for tr in range(nt):  
                sigtrace = sigdata[tr, smp_ini:]
                smax = extractSmax(sigtrace, n_sta, n_lta)  #using the ones that worked best
                Smax = np.append(Smax, smax)
            # sample number of the max intensity of each of the nt traces

            fator = 1.5                                # 1.5 is the multiplication factor
            q75, q25 = np.percentile(Smax, [75, 25])   # returns the third and first quartile
            iqr = q75 - q25                            # calculates the iqr(interquartile range)

            lowpass = q25 - (iqr * fator)              # calculates the minimum value to apply to the filter
            highpass = q75 + (iqr * fator)             # calculates the maximum value to apply to the filter

            for i in range(nt):                        # interpolates the range of the outlier
                if Smax[i] > highpass:
                    Smax[i] = Smax[i-1] # use the previous

                if Smax[i] < lowpass:
                    Smax[i] = Smax[i-1] # use the previous 

            # applying the moving average in a window of win samples
            win = 100

            sma = moving_average(Smax, win)

            # start trace value = start trace value+window //2
            # final trace value = final-window trace value //2

            w = win//2
            last = len(sma)-1
            for i in range(w):
                sma[i] = sma[w]
            for i in range(w):
                sma[last-i] = sma[last-w]

            # Creating the cubic spline over the moving average

            nt = len(sma)
            x = np.arange(nt)
            cs = interpolate.CubicSpline(x, sma, axis=0)
            cs_x = np.arange(0, nt)
            cs_y = cs(np.arange(0, nt))

            # difference between the spline and theshold at each point
            # corresponds to the number of samples between the two

            dif = (cs_y - Smax) 
            dif = dif.astype(int)

            ################ METODO 1 - using insert e del)
            
            sigcor = sigdata.tolist()  # convert array to list
            for tr in range(nt):

                if dif[tr] > 0:

                    for i in range(int(dif[tr])):

                        # insert zero at the top of the trace
                        sigcor[tr].insert(i, 0)         
                                    
                        # remove samples from the base of the trace
                        del sigcor[tr][-i]
                
                        
                if dif[tr] >= 0 :
                    for i in range(int(dif[tr])):
                        
                        # remove samples from the top of the trace
                        del sigcor[tr][-i]
                        
                        # insert zeros at the base of the trace
                        sigcor[tr].insert(i, 0)
                        
            sigcor = np.array(sigcor)  # back to array

            tit = ' SWELL Filter - METHOD #1'
            cmin = -np.percentile(sigcor, 95)
            cmax = np.percentile(sigcor, 95)
            sigfilt = '' 
            cmin, cmax, cmap = img_show(segyfile, tit, sigcor, 0, nt, 0, ns, sr, cmin, cmax, palet, sigfilt, trace_index)

            #####################  METODO 2 - using shift()

            shft = []
            for i in range(nt):
                shft.append(shift(sigdata[i,:], dif[i], cval=0))  # opção 2 usando função shift
            correct = np.nan_to_num(shft).astype(int)

            tit = ' SWEll Filter - METHOD #2'
            cmin = -np.percentile(correct, 95)
            cmax = np.percentile(correct, 95)
            sigfilt = ''
            cmin, cmax, cmap = img_show(segyfile, tit, correct, 0, nt, 0, ns, sr, cmin, cmax, palet, sigfilt, trace_index)

        # -----------------------------------------------------------------------------------    

        def save_meth1():   
            global sig

            sig = sigcor    # save meth1 over raw data
            tit = ' SWELL Filter - METHOD #1'
            sigfilt = ''
            # cmin, cmax, cmap = img_show(segyfile, tit, sig, 0, nt, 0, ns, sr, cmin, cmax, palet, sigfilt, trace_index)
            
            btn2['state'] = 'disabled'
            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\n Swell Filter - applied over Raw Data with Method 1')

            log = open(filename+'.log', 'a')
            log.write('\nSwell Filter applied over Raw Data with Method 1')

            btn3['state'] = 'disable'
        
        # -----------------------------------------------------------------------------------

        def save_meth2():
            global sig

            sig = correct    # save meth2 over raw data

            vm = np.percentile(sig, 95)
            cmin = -vm
            cmax = vm
            sigfilt=''
            tit = ' SWEll Filter - METHOD #2'

            btn2['state'] = 'disabled'
            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\nSwell Filter applied over Raw Data with Method 2')

            log = open(filename+'.log', 'a')
            log.write('\nSwell Filter applied over Raw Data with Method 2')

            btn4['state'] = 'disable'

        swell = Toplevel(corrections)
        swell.title("SWELL FILTER")
        # swell.iconbitmap('logotet.ico')
        w = 300 # window width
        h = 390 # window height
        ws = corrections.winfo_screenwidth()
        hs = corrections.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        swell.geometry('%dx%d+%d+%d' % (w, h, x, y))
        swell.resizable(width=False, height=False)
        swell.attributes('-toolwindow', True)


        lbl1 = Label(swell, text='Start Scan of Bottom Tracker (ms)')
        lbl1.place(x=40, y=110)
        ini1 = StringVar(swell, value='10')
        e_ini = Entry(swell, width=3, textvariable= ini1)
        e_ini.place(x=225, y=110)

        lbl2 = Label(swell, text='STA - number of samples')
        lbl2.place(x=40, y=140)
        ini2 = StringVar(swell, value='10')
        e_sta = Entry(swell, width=3, textvariable= ini2)
        e_sta.place(x=180, y=140)

        lbl3 = Label(swell, text='LTA - number of samples')
        lbl3.place(x=40, y=170)
        ini3 = StringVar(swell, value='20')
        e_lta = Entry(swell, width=3, textvariable= ini3)
        e_lta.place(x=180, y=170)

        lbl4 = Label(swell, text="Moving Average window size (traces)")
        lbl4.place(x=30, y=200)
        pr = StringVar(swell, value='100')
        e_win = Entry(swell, width=3, textvariable= pr)
        e_win.place(x=235, y=200)

        btn1 = Button(swell, width=15, text="View Line Image", command= view_img)
        btn1.place(x=80, y=75)      

        btn2 = Button(swell, width=20, text="Apply Swell Correction", command= apply)
        btn2.place(x=60, y=250)

        btn3 = Button(swell, width=20, text="Save with Method 1", command= save_meth1)
        btn3.place(x=60, y=290)

        btn4 = Button(swell, width=20, text="Save with Method 2", command= save_meth2)
        btn4.place(x=60, y=330)

        lbl5 = Label(swell, text='Only applicable with full working window')
        lbl5.place(x=30, y=50)

        select(swell, 87, 30)

# ===================================================================================================

    def DCremoved():

        rdata = np.copy(data)

        def applyDC():
            global dcremove, sp

            raw1d = np.ravel(rdata)
            m = np.mean(raw1d)
            dc = list(map(lambda var: m - var, raw1d)) 
            sp = int(len(raw1d)/nt)
            dcremove = np.reshape(dc, (nt, sp))
            
            vm = np.percentile(dcremove, 95)
            cmin = -vm
            cmax = vm
            sigfilt = '' 
            tit = ' DC removal filter'

            cmin, cmax, cmap = img_show(segyfile, tit, dcremove, 0, nt, 0, ns, sr, cmin, cmax, palet, sigfilt, trace_index)

        def dc_save():
            global data

            data = dcremove # save over raw data
  
            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\n DC remove filter applied over raw data')

            logfile = open(filename+'.log', 'a')
            msg = 'DC remove filter applied over raw data'
            logfile.write('\n'+msg)


        DCremove = Toplevel(corrections)
        DCremove.title("REMOVE DC")
        # DCremove.iconbitmap('logotet.ico')
        w = 280 # window width
        h = 220 # window height
        ws = corrections.winfo_screenwidth()
        hs = corrections.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        DCremove.geometry('%dx%d+%d+%d' % (w, h, x, y))
        DCremove.resizable(width=False, height=False)
        DCremove.attributes('-toolwindow', True)


        lbl1 = Label(DCremove, text='Check if your signal is not centered')
        lbl1.place(x=40, y=20)
        lbl2 = Label(DCremove, text='    around the zero level axis') 
        lbl2.place(x=40, y=40)

        tr = nt//2
        btn1 = Button(DCremove, width=15, text="Prof viewer", command=lambda: prof_viewer(tr, 0, ns, rdata[tr]))
        btn1.place(x=70, y=70)

        btn2 = Button(DCremove, width=10, text ='Apply', command=applyDC)
        btn2.place(x=90, y=120)

        btn3 = Button(DCremove, width=15, text ='Save DCremove', command= dc_save)
        btn3.place(x=70, y=160)

    # ================================================================================

    def spher_diver():
            # aplica a divergencia esferica sobre os dados brutos e reescreve

        def apply_sd():
            global SD

            def spher_diverg(dat, ntraces, nsamples, veloc):

                t = [ x*sr for x in range(nsamples)]
                sigout = np.zeros(dat.shape)
                sd = [1/veloc*x for x in t]

                for k in range(ntraces):
                    sigout[k,:] = dat[k,:]*sd
                    
                return sigout

            vel = 1500   # m/s 
            SD = spher_diverg(sigdata, sigdata.shape[0], sigdata.shape[1], vel)

            vm = np.percentile(SD, 95)   # get percentile from raw data
            cmin = -vm
            cmax = vm
            palet = sigpal
            sigfilt = ''
            tit = 'Spherical Divergence Correction'  

            cmin, cmax, cmap = img_show(segyfile, tit, SD, 0, nt, 0, ns, sr, cmin, cmax, 'gray', sigfilt, trace_index) 
            
            btn2['state'] = 'normal'

        def save_sd():
            global sigdata

            sigdata = SD    # save over raw data

            btn2['state'] = 'disabled'
            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\nSpherical Divergence Correction')

            log = open(filename+'.log', 'a')
            log.write('\nSpherical Divergence Correction')

        spher_div = Toplevel(corrections)
        spher_div.title("SPHERICAL DIVERGENCE")
        # spher_div.iconbitmap('logotet.ico')
        w = 180 # window width
        h = 200 # window height
        ws = corrections.winfo_screenwidth()
        hs = corrections.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        spher_div.geometry('%dx%d+%d+%d' % (w, h, x, y))
        spher_div.resizable(width=False, height=False)
        spher_div.attributes('-toolwindow', True)


        btn1 = Button(spher_div, width=15, text ='Apply Correction', command=apply_sd)
        btn1.place(x=30, y=80)

        btn2 = Button(spher_div, width=8, text ='Save', command= save_sd)
        btn2.place(x=50, y=120)  

        select(spher_div, 40, 40)

# ===================================================================================================

    def resize_segy():

        def savesgy():

            save_segy(seg, 'resize')

        def image_view():
            global seg
            
            tri = e_tri.get()
            if tri == '' :
                tri = 0
            else:
                tri = int(tri)

            trf = e_trf.get()
            if trf == '' :
                trf = nt
            else:
                trf = int(trf)

            tmi= e_tmi.get()
            if tmi == '':
                tmi = 0
            else:
                tmi = int(tmi)

            tmf = e_tmf.get()
            if tmf == '':
                tmf = ns*sr*1000
                tmf = int(tmf)
            else:    
                tmf = int(tmf)
                
            spi = int(tmi /sr /1000 )  # initial sample
            spf = int(tmf /sr /1000 )  # final sample

            seg = data[tri:trf, spi:spf]   #set working file in selected window
            vm = np.percentile(seg, 99)   # get the percentile from the raw data to start the viewer
            cmin = -vm
            cmax = vm
            sigfilt = '' 
            tit = 'SEG-Y VIEWER'

            cmin, cmax, cmap = img_show(segyfile, tit, seg, tri, trf, spi, spf, sr, cmin, cmax, palet, sigfilt, trace_index)

        # ---------------------------------------------------------------------------------------------
        
        resize = Toplevel(corrections)
        resize.title("RESIZE SEG-Y")
        # resize.iconbitmap('logotet.ico')
        w = 310 # window width
        h = 250 # window height
        ws = corrections.winfo_screenwidth()
        hs = corrections.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        resize.geometry('%dx%d+%d+%d' % (w, h, x, y))
        resize.resizable(width=False, height=False)
        resize.attributes('-toolwindow', True)


        lbl1 = Label(resize, text="Start Time (ms): ")
        lbl1.place(x=70, y=30)
        e_tmi = Entry(resize, width=10)
        e_tmi.place(x=170, y=30)
        e_tmi.focus()

        lbl2 = Label(resize, text="End Time (ms): ")
        lbl2.place(x=70, y=50)
        e_tmf = Entry(resize, width=10)
        e_tmf.place(x=170, y=50)

        lbl3 = Label(resize, text="Start Trace: ")
        lbl3.place(x=70, y=70)
        e_tri = Entry(resize, width=10)
        e_tri.place(x=170, y=70)

        lbl4 = Label(resize, text="End Trace: ")
        lbl4.place(x=70, y=90)
        e_trf = Entry(resize, width=10)
        e_trf.place(x=170, y=90)

        btn1 = Button(resize, width=12, text ='Image View', command= image_view)
        btn1.place(x=50, y=140)

        # btn2 = Button(resize, width=12, text ='Save SEG-Y file', command= savesgy)
        # btn2.place(x=150, y=140)

    # -------------------------------------------------------------------------------------------------------------------

    corrections = Toplevel(root)
    corrections.title("CORRECTIONS")
    # corrections.iconbitmap('logotet.ico')
    w = 240 # window width
    h = 250 # window height
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    corrections.geometry('%dx%d+%d+%d' % (w, h, x, y))
    corrections.resizable(width=False, height=False)
    corrections.attributes('-toolwindow', True)

    btn1 = Button(corrections, width=20, text="Spherical Divergence", command=spher_diver)
    btn1.place(x=45, y=30)

    btn2 = Button(corrections, width=12, text="DC removal", command=DCremoved)
    btn2.place(x=70, y=70)
    if  datatype == 1: 
        btn2['state'] = DISABLED    

    btn3 = Button(corrections, width=12, text="Swell filter", command=swell_filter)
    btn3.place(x=70, y=110)
       
    btn4 = Button(corrections, width=12, text="Resize SEG-Y", command=resize_segy)
    btn4.place(x=70, y=150)     

# ===================================================================================================

def freq_filters():

    def spectrum():

        def amplit_spectrum():
            global sigdata
            tr1 = int(trf-tri)//3
            tr2 = int(trf-tri)//2
            tr3 = int((trf-tri)*2)//3
            plt.figure(figsize=(8, 4))
            stp = (trf-tri)//4
            for i in range(0, trf-tri, stp):
                Xf_mag = np.abs(np.fft.fft(sigdata[i]))
                freqs = np.fft.fftfreq(len(Xf_mag), d=sr)
                plt.plot(abs(freqs), Xf_mag, label='Trace '+str(i))
            plt.title('Line '+str(segyfile)+ ' - AMPLITUDE SPECTRUM '+sigfilt, fontsize=11)
            plt.xlabel('frequency [Hz]')
            plt.ylabel('|amplitude|')
            plt.legend()
            plt.grid()
            plt.show(block=False)      

        def spectrogram():
            global sigdata
            # mode = psd, magnitude, angle, phase        
            freq = 1/sr
            Trc = int(e_Trc.get())
            if V.get() == 1:
                sig1t = np.ravel(sigdata[Trc])
                tit = 'SPECTROGRAM - Trace '+str(Trc)
            if V.get() == 2:
                sig1t = np.ravel(sigdata)
                tit = 'SPECTROGRAM (magnitude) '+sigfilt+ '  -  All Traces'

            plt.figure(figsize=(6, 4))
            Pxx, freqs, bins, im = plt.specgram(sig1t, NFFT=256, Fs=freq, noverlap=128, 
                                                mode='magnitude', scale='dB', cmap='turbo') 
            plt.title(tit, fontsize=11)
            plt.xlabel('Time (sec)')
            plt.ylabel('Frequency')
            plt.colorbar(label='Intensity',orientation='vertical',)
            plt.grid()
            plt.tight_layout()
            plt.show()

        spectrum = Toplevel(freq_filters)
        spectrum.title("SPECTRUM")
        # spectrum.iconbitmap("logotet.ico")
        w = 250 # window width
        h = 290 # window height
        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        spectrum.geometry('%dx%d+%d+%d' % (w, h, x, y))
        spectrum.resizable(width=False, height=False)
        spectrum.attributes('-toolwindow', True)

        btn1 = Button(spectrum, width=15, text ='Amplitude', command=amplit_spectrum)
        btn1.place(x=60, y=70)

        btn2 = Button(spectrum, width=15, text='Spectrogram', command=spectrogram)
        btn2.place(x=60, y=125)

        V = IntVar()
        V.set(2)       

        Rbtn2 = Radiobutton(spectrum, text = "All Traces", variable= V, value = 2)
        Rbtn2.place(x=70, y=170)

        Rbtn1 = Radiobutton(spectrum, text = "One Trace", variable= V, value = 1)
        Rbtn1.place(x=70, y=200)            

        lbl1 = Label(spectrum, text='Trace number: ')
        lbl1.place(x=70, y=220)

        e_Trc = StringVar(spectrum, value=100)
        e_Trc = Entry(spectrum, width= 5, textvariable= e_Trc)
        e_Trc.place(x=160, y=220)

        select(spectrum, 70, 30)

# ===================================================================================================  
  
    def band_pass():

        # note: applying BandPass the enveloped signal becomes analytical
        def bp_filter():
            global BP, cminBP, cmaxBP, cmapBP, tit
            global HC, LC

            LC = e_LC.get()
            HC = e_HC.get()
            LC = float(LC)
            HC = float(HC)
            ord = 5 # filter order

            # try:
            BP = butter_bandpass_filter(sigdata, LC, HC, 1/sr, order=ord) # order => ramp angle

            vm = np.percentile(BP, 95)
            cmin = -vm  
            cmax = vm
            palet = sigpal
            tit = 'Band-Pass Filter ['+str(LC)+' - '+str(HC)+' Hz]'  

            cminBP, cmaxBP, cmapBP = img_show(segyfile, tit, BP, tri, trf, spi, spf, sr, cmin, cmax, palet, sigfilt, trace_index)

            # except Exception as e:
            #     messagebox.showerror("Error", "frequency limits")

            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\nBand-Pass filter applied: ')
            txt_edit.insert(END, '\nLow Frequency (Hz): '+ str(LC))
            txt_edit.insert(END, '\nHigh Frequency (Hz): '+ str(HC))
        
        # ------------------------------------------------------------------------------------------

        def save_bp():

            scene = [BP, cminBP, cmaxBP, tit, sigfilt, cmapBP]
            scenes.loc[len(scenes)] = scene

            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\nBand-Pass filter saved')

            logfile = open(filename+'.log', 'a')
            logfile.write('\nBand-Pass Filter ['+str(LC)+' - '+str(HC)+' Hz]'+sigfilt)


         # -----------------------------------------------------------------------------------------

        def save_bp_sgy():

            save_segy(BP, 'BP')

            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\nBand-Pass filter saved as SEG-Y')

        band_pass = Toplevel(freq_filters)
        band_pass.title("BAND-PASS")
        # band_pass.iconbitmap('logotet.ico')
        w = 275 # window width
        h = 290 # window height
        ws = band_pass.winfo_screenwidth()
        hs = band_pass.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        band_pass.geometry('%dx%d+%d+%d' % (w, h, x, y))
        band_pass.resizable(width=False, height=False)
        band_pass.attributes('-toolwindow', True)


        lbl1 = Label(band_pass, text="Low Cut (Hz): ")
        lbl1.place(x=50, y=70)
        e_LC = Entry(band_pass, width=10)
        e_LC.place(x=140, y=70)
        e_LC.focus()

        lbl2 = Label(band_pass, text="High Cut (Hz): ")
        lbl2.place(x=50, y=100)
        e_HC = Entry(band_pass, width=10)
        e_HC.place(x=140, y=100)

        btn1 = Button(band_pass, width=12, text ='Apply', command=bp_filter)
        btn1.place(x=80, y=140)

        btn2 = Button(band_pass, width=12, text ='Save', command=save_bp)
        btn2.place(x=80, y=180)

        # btn3 = Button(band_pass, width=12, text ='Save SEG-Y', command=save_bp_sgy)
        # btn3.place(x=80, y=220)

        select(band_pass, 80, 30)

    # ===================================================================================================

    def param_equal():

        '''
        font: wjchen in
        https://github.com/chenwj1989/pafx
        
        '''
        class Biquad():
            def __init__(
                self, 
                sample_rate, 
                filter_type=None,
                fc=1000,
                bandwidth=1.0,
                gain_db=1.0
            ):
                if sample_rate < 0.0:
                    raise ValueError("sample_rate cannot be given a negative value")
                self.sample_rate = sample_rate
            
                self.b = np.zeros(3)
                self.a = np.zeros(3)
                self.a [0] = 1.0
                self.b[0] = 1.0

                self.y = None
                self.x_buf = np.zeros(2)
                self.y_buf = np.zeros(2)
                    
                self.filter_type = filter_type

                if fc > self.sample_rate /2.0:
                    fc = self.sample_rate

                # if fc < 0.0 or fc > self.sample_rate / 2.0:
                #     raise ValueError(f"illegal value: fc={fc}")
                self._fc = fc

                self._gain_db = gain_db
                A = 10.0 ** (gain_db/40.0)
                A_add_1 = A + 1.0
                A_sub_1 = A - 1.0
                sqrt_A  = np.sqrt(A)

                w0 = 2.0 * np.pi * self._fc / self.sample_rate
                cos_w0 = np.cos(w0) 
                sin_w0 = np.sin(w0)
                alpha = 0.5 * sin_w0 * fc / bandwidth

                if filter_type == "LowPass":
                    self.b[0] = (1.0 - cos_w0) * 0.5
                    self.b[1] = (1.0 - cos_w0)
                    self.b[2] = (1.0 - cos_w0) * 0.5
                    self.a[0] =  1.0 + alpha
                    self.a[1] = -2.0 * cos_w0
                    self.a[2] =  1.0 - alpha

                elif filter_type == "HighPass":
                    self.b[0] =  (1.0 + cos_w0) * 0.5
                    self.b[1] = -(1.0 + cos_w0)
                    self.b[2] =  (1.0 + cos_w0) * 0.5
                    self.a[0] =   1.0 + alpha
                    self.a[1] =  -2.0 * cos_w0
                    self.a[2] =   1.0 - alpha

                elif filter_type == "BandPass":
                    self.b[0] =  alpha
                    self.b[1] =  0.0
                    self.b[2] = -alpha
                    self.a[0] =  1.0 + alpha
                    self.a[1] = -2.0 * cos_w0
                    self.a[2] =  1.0 - alpha

                elif filter_type == "AllPass":
                    self.b[0] =  1.0 - alpha
                    self.b[1] = -2.0 * cos_w0
                    self.b[2] =  1.0 + alpha
                    self.a[0] =  1.0 + alpha
                    self.a[1] = -2.0 * cos_w0
                    self.a[2] =  1.0 - alpha

                elif filter_type == "Notch":
                    self.b[0] =  1.0
                    self.b[1] = -2.0 * cos_w0
                    self.b[2] =  1.0
                    self.a[0] =  1.0 + alpha
                    self.a[1] = -2.0 * cos_w0
                    self.a[2] =  1.0 - alpha

                elif filter_type == "Peaking":
                    if A != 1.0:
                        self.b[0] =  1.0 + alpha * A
                        self.b[1] = -2.0 * cos_w0
                        self.b[2] =  1.0 - alpha * A
                        self.a[0] =  1.0 + alpha / A
                        self.a[1] = -2.0 * cos_w0
                        self.a[2] =  1.0 - alpha / A

                elif filter_type == "LowShelf":
                    if A != 1.0:
                        self.b[0] =     A * (A_add_1 - A_sub_1 * cos_w0 + 2 * sqrt_A * alpha)
                        self.b[1] = 2 * A * (A_sub_1 - A_add_1 * cos_w0)
                        self.b[2] =     A * (A_add_1 - A_sub_1 * cos_w0 - 2 * sqrt_A * alpha)
                        self.a[0] =          A_add_1 + A_sub_1 * cos_w0 + 2 * sqrt_A * alpha
                        self.a[1] =    -2 * (A_sub_1 + A_add_1 * cos_w0)
                        self.a[2] =          A_add_1 + A_sub_1 * cos_w0 - 2 * sqrt_A * alpha

                elif filter_type == "HighShelf":
                    if A != 1.0:
                        self.b[0] =      A * (A_add_1 + A_sub_1 * cos_w0 + 2 * sqrt_A * alpha)
                        self.b[1] = -2 * A * (A_sub_1 + A_add_1 * cos_w0)
                        self.b[2] =      A * (A_add_1 + A_sub_1 * cos_w0 - 2 * sqrt_A * alpha)
                        self.a[0] =           A_add_1 - A_sub_1 * cos_w0 + 2 * sqrt_A * alpha
                        self.a[1] =      2 * (A_sub_1 - A_add_1 * cos_w0)
                        self.a[2] =           A_add_1 - A_sub_1 * cos_w0 - 2 * sqrt_A * alpha
                else:
                    raise ValueError(f"invalid filter_type: {filter_type}")
                
                self.b /= self.a[0]
                self.a /= self.a[0]
                
            def process(self, x):
                y = self.b[0] * x\
                    + self.b[1] * self.x_buf[1]\
                    + self.b[2] * self.x_buf[0]\
                    - self.a[1] * self.y_buf[1]\
                    - self.a[2] * self.y_buf[0]
                self.x_buf[0] = self.x_buf[1]
                self.x_buf[1] = x
                self.y_buf[0] = self.y_buf[1]
                self.y_buf[1] = y
                return y

        # ----------------------------------------------------------------------------------------

        class Equalizer():
            def __init__(self, gains, sample_rate):

                self.num_bands = 8
                    
                # Parallel biquad filters
                self.filters = []

                # A low shelf filter for the lowest band
                self.filters.append(Biquad(sample_rate, 'LowShelf', center_freqs[0],
                                            band_widths[0], gains[0]))  

                # Peaking filters for the middle bands
                for i in range(1, self.num_bands - 1):
                    self.filters.append(Biquad(sample_rate, 'Peaking', center_freqs[i],
                                            band_widths[i], gains[i]))   

                # A high shelf filter for the highest band
                self.filters.append(Biquad(sample_rate, 'HighShelf',
                                            center_freqs[self.num_bands - 1],
                                            band_widths[self.num_bands - 1],
                                            gains[self.num_bands - 1])
                                    )  

            def process(self, x):
                out = x
                for filter in self.filters:
                    out += filter.process(out)
                return out #/ self.num_bands
                
        #-----------------------------------------------------------------------------------------

        def apply_equaliz():
            global EQ, cminEQ, cmaxEQ, cmapEQ, tit
            global center_freqs, band_widths
            global G1, G2, G3, G4, G5, G6, G7, G8

            F1 = varf1.get()
            G1 = varg1.get()
            F2 = varf2.get()
            G2 = varg2.get()
            F3 = varf3.get()
            G3 = varg3.get()
            F4 = varf4.get()
            G4 = varg4.get()
            F5 = varf5.get()
            G5 = varg5.get()
            F6 = varf6.get()
            G6 = varg6.get()
            F7 = varf7.get()
            G7 = varg7.get()
            F8 = varf8.get()
            G8 = varg8.get()
            center_freqs = [F1, F2, F3, F4, F5, F6, F7, F8]

            BW1 = F1/np.sqrt(2)
            BW2 = F2/np.sqrt(2)
            BW3 = F3/np.sqrt(2)
            BW4 = F4/np.sqrt(2)
            BW5 = F5/np.sqrt(2)
            BW6 = F6/np.sqrt(2)
            BW7 = F7/np.sqrt(2)
            BW8 = F8/np.sqrt(2)
            band_widths = [BW1, BW2, BW3, BW4, BW5, BW6, BW7, BW8]


            x = np.ravel(sigdata)
            y = np.zeros(len(x))

            eq_gains = [G1, G2, G3, G4, G5, G6, G7, G8]   # ganho de cada canal
            fs = 1/sr
            eq = Equalizer(eq_gains, fs)
            
            # Start Processing
            for i in range(len(x)):
                y[i] = eq.process(x[i])   # processa a equalizacao

            sp = int(len(y)/(trf-tri))
            EQ = np.reshape(y, (trf-tri, sp))  

            G1 = int(G1)
            G2 = int(G2)
            G3 = int(G3)
            G4 = int(G4)
            G5 = int(G5)
            G6 = int(G6)
            G7 = int(G7)
            G8 = int(G8)

            vm = np.percentile(EQ, 95)
            cmin = -vm  
            cmax = vm
            palet = sigpal
            tit = 'Parametric Equalizer - Gain: '+str(G1)+' '+str(G2)+' '+str(G3)+' '+str(G4)+' '+str(G5)+' '+str(G6)+' '+str(G7)+' '+str(G8)

            cminEQ, cmaxEQ, cmapEQ = img_show(segyfile, tit, EQ, tri, trf, spi, spf, sr, cmin, cmax, palet, sigfilt, trace_index)      

        # -----------------------------------------------------------------------------------------------------

        def save_equaliz():

            scene = [EQ, cminEQ, cmaxEQ, tit, sigfilt, cmapEQ]
            scenes.loc[len(scenes)] = scene


            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\nParametric Equalizer filter saved')

            logfile = open(filename+'.log', 'a')
            logfile.write('\nParametric Equalizer - Gain: ['+str(G1)+' '+str(G2)+' '+str(G3)+' '+str(G4)+' '+str(G5)+' '+str(G6)+' '+str(G7)+' '+str(G8)+']'+sigfilt)

            # equaliz.destroy()

        # -------------------------------------------------------------------------------------------------------------------

        def save_equaliz_sgy():

            save_segy(EQ, 'EQ')

            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\nParametric Equalizer filter saved as SEG-Y')

        # ----------------------------------------------------------------------------------------

        equaliz = Toplevel(freq_filters)
        equaliz.title('PARAMETRIC EQUALIZER')
        # equaliz.iconbitmap("logotet.ico")
        w = 530 # window width
        h = 370 # window height
        ws = equaliz.winfo_screenwidth()
        hs = equaliz.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        equaliz.geometry('%dx%d+%d+%d' % (w, h, x, y))
        equaliz.resizable(width=False, height=False)
        equaliz.attributes('-toolwindow', True) 

        lbl = Label(equaliz, text='Freq max= '+str(int(1/sr)))
        lbl.place(x=75, y=30)

        lbl1 = Label(equaliz, text='BAND\n1')
        lbl1.place(x=45, y=60)

        lbl2 = Label(equaliz, text='Freq')
        lbl2.place(x=50, y=100)

        varf1 = DoubleVar(equaliz, value=500)
        f1 = Entry(equaliz, width=5, textvariable=varf1)
        f1.place(x=50, y=120)

        varg1 = DoubleVar()
        scale1 = Scale(equaliz, variable=varg1, from_=50, to=-50)
        scale1.place(x=35, y=150)

        lbl2a = Label(equaliz, text='Gain')
        lbl2a.place(x=50, y=250)

        lbl1 = Label(equaliz, text='BAND\n2')
        lbl1.place(x=95, y=60)

        lbl2 = Label(equaliz, text='Freq')
        lbl2.place(x=100, y=100)

        varf2 = DoubleVar(equaliz, value=1000)
        f2 = Entry(equaliz, width=5, textvariable=varf2)
        f2.place(x=100, y=120)

        varg2 = DoubleVar()
        scale2 = Scale(equaliz, variable=varg2, from_=50, to=-50)
        scale2.place(x=85, y=150)

        lbl2a = Label(equaliz, text='Gain')
        lbl2a.place(x=100, y=250)

        lbl1 = Label(equaliz, text='BAND\n3')
        lbl1.place(x=145, y=60)

        lbl2 = Label(equaliz, text='Freq')
        lbl2.place(x=150, y=100)

        varf3 = DoubleVar(equaliz, value=1500)
        f3= Entry(equaliz, width=5, textvariable=varf3)
        f3.place(x=150, y=120)

        varg3 = DoubleVar()
        scale3 = Scale(equaliz, variable=varg3, from_=50, to=-50)
        scale3.place(x=135, y=150)

        lbl2a = Label(equaliz, text='Gain')
        lbl2a.place(x=150, y=250)

        lbl1 = Label(equaliz, text='BAND\n4')
        lbl1.place(x=195, y=60)

        lbl2 = Label(equaliz, text='Freq')
        lbl2.place(x=200, y=100)

        varf4 = DoubleVar(equaliz, value=2000)
        f4 = Entry(equaliz, width=5, textvariable=varf4)
        f4.place(x=200, y=120)

        varg4 = DoubleVar()
        scale4 = Scale(equaliz, variable=varg4, from_=50, to=-50)
        scale4.place(x=185, y=150)

        lbl2a = Label(equaliz, text='Gain')
        lbl2a.place(x=200, y=250)

        lbl1 = Label(equaliz, text='BAND\n5')
        lbl1.place(x=245, y=60)

        lbl2 = Label(equaliz, text='Freq')
        lbl2.place(x=250, y=100)

        varf5 = DoubleVar(equaliz, value=2500)
        f5 = Entry(equaliz, width=5, textvariable=varf5)
        f5.place(x=250, y=120)

        varg5 = DoubleVar()
        scale5 = Scale(equaliz, variable=varg5, from_=50, to=-50)
        scale5.place(x=235, y=150)

        lbl2a = Label(equaliz, text='Gain')
        lbl2a.place(x=250, y=250)

        lbl1 = Label(equaliz, text='BAND\n6')
        lbl1.place(x=295, y=60)

        lbl2 = Label(equaliz, text='Freq')
        lbl2.place(x=300, y=100)

        varf6 = DoubleVar(equaliz, value=3000)
        f6 = Entry(equaliz, width=5, textvariable=varf6)
        f6.place(x=300, y=120)

        varg6 = DoubleVar()
        scale6= Scale(equaliz, variable=varg6, from_=50, to=-50)
        scale6.place(x=285, y=150)

        lbl2a = Label(equaliz, text='Gain')
        lbl2a.place(x=300, y=250)

        lbl1 = Label(equaliz, text='BAND\n7')
        lbl1.place(x=345, y=60)

        lbl2 = Label(equaliz, text='Freq')
        lbl2.place(x=350, y=100)

        varf7 = DoubleVar(equaliz, value=4000)
        f7 = Entry(equaliz, width=5, textvariable=varf7)
        f7.place(x=350, y=120)

        varg7 = DoubleVar()
        scale7 = Scale(equaliz, variable=varg7, from_=50, to=-50)
        scale7.place(x=335, y=150)

        lbl2a = Label(equaliz, text='Gain')
        lbl2a.place(x=350, y=250)

        lbl1 = Label(equaliz, text='BAND\n8')
        lbl1.place(x=395, y=60)

        lbl2 = Label(equaliz, text='Freq')
        lbl2.place(x=400, y=100)

        varf8 = DoubleVar(equaliz, value=6000)
        f8 = Entry(equaliz, width=5, textvariable=varf8)
        f8.place(x=400, y=120)

        varg8 = DoubleVar()
        scale8 = Scale(equaliz, variable=varg8, from_=50, to=-50)
        scale8.place(x=385, y=150)

        lbl2a = Label(equaliz, text='Gain')
        lbl2a.place(x=400, y=250)

        lbl1 = Label(equaliz, text='BAND\n9')
        lbl1.place(x=445, y=60)

        lbl2 = Label(equaliz, text='Freq')
        lbl2.place(x=450, y=100)

        varf9 = DoubleVar(equaliz, value=8000)              # passei pra 8000
        f9 = Entry(equaliz, width=5, textvariable=varf9)
        f9.place(x=450, y=120)

        varg9 = DoubleVar()
        scale9 = Scale(equaliz, variable=varg9, from_=50, to=-50)
        scale9.place(x=435, y=150)

        lbl2a = Label(equaliz, text='Gain')
        lbl2a.place(x=450, y=250)

        btn1 = Button(equaliz, text='Apply', width=15, command=apply_equaliz)
        btn1.place(x=350, y=290)

        btn2 = Button(equaliz, text='Save', width=10, command=save_equaliz)
        btn2.place(x=200, y=290)

        # btn3 = Button(equaliz, text='Save SEG-Y', width=12, command=save_equaliz_sgy)
        # btn3.place(x=70, y=290)

        select(equaliz, 200, 30)

#===============================================================================

    def spectral_whitening():


        def sel():
            global meth

            m = var.get()
            if m == 1:
                meth = 'zca'
            if m == 2:
                meth = 'pca'
            if m == 3:
                meth = 'cholesky'

            
        def show_whitening():
            global WT, cminWT, cmaxWT, cmapWT, tit

            WT = whiten(sigdata, method=meth)

            vm = np.percentile(WT, 95)
            cmin = -vm  
            cmax = vm
            palet = sigpal
            tit = 'Spectrum Whitening - Method: '+meth 
            
            cminWT, cmaxWT, cmapWT = img_show(segyfile, tit, WT, tri, trf, spi, spf, sr, cmin, cmax, palet, sigfilt, trace_index)

            # btn2['state'] = 'normal'
            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\nSpectral Whitening applied: ')
            txt_edit.insert(END, '\nMethod: ' + meth+sigfilt)

        # -------------------------------------------------------------------------------------

        def save_whiten():

            scene = [WT, cminWT, cmaxWT, tit, sigfilt, cmapWT]
            scenes.loc[len(scenes)] = scene


            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\nSpectral Whitening saved')

            logfile = open(filename+'.log', 'a')
            logfile.write('\nSpectral Whitening - Method: '+meth+sigfilt)

            # whitening.destroy()

        # -------------------------------------------------------------------------------------
            
        def save_whiten_sgy():

            save_segy(WT, 'SW')

            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\nSpectrum Whitening filter saved as SEG-Y')

        # -------------------------------------------------------------------------------------

        whitening = Toplevel(freq_filters)        
        whitening.title("SPECTRAL WHITENING")
        # whitening.iconbitmap("logotet.ico")
        w = 240 # window width
        h = 300 # window height
        ws = whitening.winfo_screenwidth()
        hs = whitening.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        whitening.geometry('%dx%d+%d+%d' % (w, h, x, y))
        whitening.resizable(width=False, height=False)
        whitening.attributes('-toolwindow', True)

        btn1 = Button(whitening, width=10, text="Apply", command=show_whitening)
        btn1.place(x=65, y=150)

        btn2 = Button(whitening, width=10, text ='Save', command= save_whiten)
        btn2.place(x=65, y=190)

        # btn3 = Button(whitening, width=10, text ='Save SEG-Y', command= save_whiten_sgy)
        # btn3.place(x=65, y=230)

        var = IntVar()    

        but1 = Radiobutton(whitening, text='ZCA method', variable=var, value=1, command=sel)
        but1.place(x=60, y=70)
           
        but2 = Radiobutton(whitening, text='PCA method', variable=var, value=2, command=sel) 
        but2.place(x=60, y=90)
  
        but3 = Radiobutton(whitening, text='Cholesky method', variable=var, value=3, command=sel) 
        but3.place(x=60, y=110)

        select(whitening, 60, 30)

    #=================================================================================================================

    freq_filters = Toplevel(root)
    freq_filters.title("FREQUENCIES FILTERS")
    # freq_filters.iconbitmap('logotet.ico')
    w = 220 # window width
    h = 270 # window height
    ws = freq_filters.winfo_screenwidth()
    hs = freq_filters.winfo_screenheight()
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    freq_filters.geometry('%dx%d+%d+%d' % (w, h, x, y))
    freq_filters.resizable(width=False, height=False)
    freq_filters.attributes('-toolwindow', True)

    btn1 = Button(freq_filters, text='SPECTRUM', width=18, command=spectrum)
    btn1.place(x=40, y=40)
    btn2 = Button(freq_filters, text='BAND PASS', width=18, command=band_pass)
    btn2.place(x=40, y=90)
    btn3 = Button(freq_filters, text='SPECTRAL WHITEN', width=18, command=spectral_whitening)
    btn3.place(x=40, y=140)
    btn4 = Button(freq_filters, text='PARAM EQUALIZ', width=18, command=param_equal)
    btn4.place(x=40, y=190)

# ===================================================================================================

def ampl_filter():
        
    def agc_gain():
    
        def apply_agc():                
            global AGC, cminAGC, cmaxAGC, cmapAGC, tit
            global weigth

            def agc(signal, window_size):
                """
                Applies Automatic Gain Control (AGC) to the seismic signal.

                Parameters:
                    - signal: NumPy array representing the seismic signal.
                    - window_size: Sliding window size in samples.

                Return:
                    - Adjusted AGC signal.
                """
                # Calculates the square of the absolute value of the signal
                envelope = np.abs(signal) ** 2

                # Calculates moving average with sliding window
                smoothed = np.convolve(envelope, np.ones(window_size) / window_size, mode='same')

                # Calculates the square root of the smoothed moving average
                smoothed_sqrt = np.sqrt(smoothed)

                # Normalizes the original signal by the smoothed signal
                agc_signal = signal / smoothed_sqrt

                return agc_signal

            weigth = w.get()
            AGC = []
            for i in range(trf-tri):
                a = agc(sigdata[i], int(weigth))
                AGC = np.append(AGC, a)

            sp = int(len(AGC)/(trf-tri))
            AGC = np.reshape(AGC, (trf-tri, spf-spi))

            vm = np.percentile(AGC, 95)
            cmin = -vm  
            cmax = vm
            palet = sigpal
            tit = 'Automatic Gain Control (AGC) - Weigth: '+ str(weigth)

            cminAGC, cmaxAGC, cmapAGC = img_show(segyfile, tit, AGC, tri, trf, spi, spf, sr, cmin, cmax, palet, sigfilt, trace_index)

            btn2['state'] = 'normal'
        

        # --------------------------------------------------------------------------------------------

        def save_agc():

            scene = [AGC, cminAGC, cmaxAGC, tit, sigfilt, cmapAGC]
            scenes.loc[len(scenes)] = scene

            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\nAGC filter saved')

            logfile = open(filename+'.log', 'a')
            logfile.write('\nAutomatic Gain Control (AGC) - Weigths: '+ str(weigth)+sigfilt)


        # --------------------------------------------------------------------------------------------

        def save_agc_sgy():

            save_segy(AGC, 'AGC')

            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\nAGC filter saved as SEG-Y')

        # --------------------------------------------------------------------------------------------

        Agc = Toplevel(ampl_filters)
        Agc.title('AGC')
        # AGC.iconbitmap('logotet.ico')
        w = 250 # window width
        h = 240 # window height
        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        Agc.geometry('%dx%d+%d+%d' % (w, h, x, y))
        Agc.resizable(width=False, height=False)
        Agc.attributes('-toolwindow', True)

        lbl1 = Label(Agc, text="Weigth ")
        lbl1.place(x=40, y=70)
        var1 = StringVar(Agc, value=50)        
        w = Entry(Agc, width=8, textvariable=var1)
        w.place(x=100, y=70)

        btn1 = Button(Agc, width=10, text='Apply', command= apply_agc)
        btn1.place(x=80, y= 100)
        btn2 = Button(Agc, width=10, text='Save', command= save_agc)
        btn2.place(x=80, y= 140)
        # btn3 = Button(Agc, width=10, text='Save SEG-Y', command= save_agc_sgy)
        # btn3.place(x=80, y= 180)

        select(Agc, 70, 30)

    # ==============================================================================================================

    def manual_gain():

        def edit_mgain():
            global MG, cminMG, cmaxMG, cmapMG, tit

            if min(sigdata[50]) < 0:

                # mpl.rcParams['toolbar']='None'
                fig = plt.figure(figsize=(10,4))#, constrained_layout=True)

                ax1 = fig.add_subplot(211)
                plt.hlines(0, spi, spf, color='k')
                ax2 = fig.add_subplot(212)
                plt.hlines(0, spi, spf, color='k')
                trc = sigdata[int((trf-tri)/2)] # pega um traco no meio
                ax1.plot(trc, color='b')
                ax1.set_title('Click to draw Gain line    -    ESC to show applyied gain', fontsize=11)
                ax2.set_title('Applyied Gain over trace', fontsize=11)
                draw_line = ManualGain(fig, ax1, ax2, trc)
                fig.canvas.mpl_connect('button_press_event', draw_line.mouse_click)
                fig.canvas.mpl_connect('key_press_event', draw_line.on_key)
                plt.grid()
                plt.tight_layout()
                plt.show()

                nx, nt = sigdata.shape
                interv = nt
                xnew = np.linspace(draw_line.xs[0], draw_line.xs[-1], interv) # vector with line xy
                # fint = scipy.interpolate.interp1d(draw_line.xs, draw_line.ys)
                # y = fint(xnew)  # interpolacao y desse vetor
                fint = make_interp_spline(draw_line.xs, draw_line.ys)
                y = fint(xnew)  # y-interpolation of this vector

                for i in range(nx-1):
                    for n1, n2 in zip(y, sigdata[i, :]): 
                        draw_line.Y.append((n1/4) * n2)

                sp = int(len(draw_line.Y)/nx)
                MG = np.reshape(draw_line.Y, (nx, sp))

                vm = np.percentile(MG, 95)
                cmin = -vm  
                cmax = vm
                palet = sigpal
                tit = 'Gain aplyied with built curve'

                cminMG, cmaxMG, cmapMG = img_show(segyfile, tit, MG, tri, trf, spi, spf, sr, cmin, cmax, palet, sigfilt, trace_index)

                btn2['state'] = 'normal'
            else:
                messagebox.showerror("Error", "Enveloped data - apply Band-Pass first")
            

        # -----------------------------------------------------------------------------------------

        def save_mgain():

            scene = [MG, cminMG, cmaxMG, tit, sigfilt, cmapMG]
            scenes.loc[len(scenes)] = scene


            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\nManual Gain filter saved')

            logfile = open(filename+'.log', 'a')
            logfile.write('\nManual Gain filter'+sigfilt)

        # -----------------------------------------------------------------------------------------

        def save_mgain_sgy():

            save_segy(MG, 'MG')

            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\nManual Gain saved as SEG-Y')

        # -----------------------------------------------------------------------------------------

        mgain = Toplevel(ampl_filters)
        mgain.title('MANUAL GAIN')
        # mgain.iconbitmap('logotet.ico')
        w = 220 # window width
        h = 260 # window height
        ws = mgain.winfo_screenwidth()
        hs = mgain.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        mgain.geometry('%dx%d+%d+%d' % (w, h, x, y))
        mgain.resizable(width=False, height=False)
        mgain.attributes('-toolwindow', True)

        btn1 = Button(mgain, width=15, text='Manual Gain Editor', command= edit_mgain)
        btn1.place(x=50, y= 70)

        btn2 = Button(mgain, width=10, text='Save', command= save_mgain)
        btn2.place(x=60, y= 120)

        # btn3 = Button(mgain, width=12, text='Save SEG-Y', command= save_mgain_sgy)
        # btn3.place(x=60, y= 160)

        select(mgain, 60, 30)

    # ===================================================================================================

    def stalta():

        '''
        font: 
        https://docs.obspy.org/_modules/obspy/signal/trigger.html#classic_sta_lta
        
        '''

        def stalta_calc():
            global STA, cminSTA, cmaxSTA, cmapSTA, tit
            global nsta, nlta

            nsta = s_ini.get()
            if nsta == '':
                nsta = 10
            else:
                nsta = int(nsta)

            nlta = s_end.get()
            if nlta == '':
                nlta = 20
            else:
                nlta = int(nlta)

            sigdata1d = np.ravel(sigdata)  

            st_lt = classic_sta_lta_py(sigdata1d, nsta, nlta)

            sp = int(len(st_lt)/(trf-tri))
            STA = np.reshape(st_lt, (trf-tri, sp))

            vm = np.percentile(STA, 95)
            cmin = -vm  
            cmax = vm
            palet = sigpal           
            tit = 'STA/LTA Filter - STA window= '+str(nsta)+' - LTA window= '+ str(nlta)

            cminSTA, cmaxSTA, cmapSTA = img_show(segyfile, tit, STA, tri, trf, spi, spf, sr, cmin, cmax, palet, sigfilt, trace_index)

        # -----------------------------------------------------------------------------------------

        def save_stalta():
            
            scene = [STA, cminSTA, cmaxSTA, tit, sigfilt, cmapSTA]
            scenes.loc[len(scenes)] = scene

            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\n STA/LTA filter saved')

            logfile = open(filename+'.log', 'a')
            logfile.write('\nSTA/LTA filter - Relation: '+str(nsta)+'/'+ str(nlta)+' samples'+sigfilt)

        # -----------------------------------------------------------------------------------------

        def save_stalta_sgy():

            save_segy(STA, 'ST')

            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\nSTA/LTA filter saved as SEG-Y')

        # -----------------------------------------------------------------------------------------

        st_lt = Toplevel(ampl_filters)
        st_lt.title('STA/LTA')
        # st_lt.iconbitmap("logotet.ico")
        w = 240 # window width
        h = 310 # window height
        ws = st_lt.winfo_screenwidth()
        hs = st_lt.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        st_lt.geometry('%dx%d+%d+%d' % (w, h, x, y))
        st_lt.resizable(width=False, height=False)
        st_lt.attributes('-toolwindow', True)

        lbl1 = Label(st_lt, text="STA window duration: ")
        lbl1.place(x=25, y=80)
        s_ini = Entry(st_lt, width=5)
        s_ini.place(x=150, y=80)

        lbl2 = Label(st_lt, text="LTA window duration: ")
        lbl2.place(x=25, y=110)
        s_end = Entry(st_lt, width=5)
        s_end.place(x=150, y=110)

        btn1 = Button(st_lt, width=15, text='Apply', command=stalta_calc)
        btn1.place(x=50, y= 150)
        
        btn4 = Button(st_lt, width=15, text='Save', command=save_stalta)
        btn4.place(x=50, y= 190)

        # btn5 = Button(st_lt, width=15, text='Save SEG-Y', command=save_stalta_sgy)
        # btn5.place(x=50, y= 230)

        select(st_lt, 70, 30)

    # ===================================================================================================
       
    def hilbert():

        def calc_hilb():  
            global IA, IF, IP

            # process Hilbert Transform
            IA, IF, IP = hilb(sigdata, sr) #   obtem InstAmplit, InstFreq, InstFase (IA, IF, IP)


            btn2['state'] = 'normal'
            btn5['state'] = 'normal'
            btn8['state'] = 'normal'

        # ----------------------------------------------------------------------------

        def show_ia():
            global cminIA, cmaxIA, cmapIA, tit

            vm = np.percentile(IA, 95)
            cmin = -vm  
            cmax = vm
            palet = sigpal
            tit = 'Instant Amplitude (Envelope)'

            cminIA, cmaxIA, cmapIA = img_show(segyfile, tit, IA, tri, trf, spi, spf, sr, cmin, cmax, palet, sigfilt, trace_index)

            btn3['state'] = 'normal'

        # ----------------------------------------------------------------------------

        def show_if():
            global cminIF, cmaxIF, cmapIF, tit

            vm = np.percentile(IF, 95)
            cmin = -vm  
            cmax = vm
            palet = sigpal
            tit = 'Instant Frequency'
            cminIF, cmaxIF, cmapIF = img_show(segyfile, tit, IF, tri, trf, spi, spf, sr, cmin, cmax, palet, sigfilt, trace_index)

            btn6['state'] = 'normal'

        # ----------------------------------------------------------------------------

        def show_pcos():
            global PC, cminPC, cmaxPC, cmapPC, tit

            PC = np.cos(IP)  # extract phase cosine

            vm = np.percentile(PC, 95)
            cmin = -vm  
            cmax = vm
            palet = sigpal
            tit = 'Phase Cosine'

            cminPC, cmaxPC, cmapPC = img_show(segyfile, tit, PC, tri, trf, spi, spf, sr, cmin, cmax, palet, sigfilt, trace_index)

            btn9['state'] = 'normal'

        # ----------------------------------------------------------------------------
        def show_accel():
            global AC, cminAC, cmaxAC, cmapAC, tit

            x = np.linspace(spi, spf, spf-spi) # array of samples

            for tr in range(trf-tri):    # calculates the derivatives of each trace

                y = data[tr, spi:spf] 
                spl = UnivariateSpline(x, y)    # creates spline over trace
                drv_spl = spl.derivative()    # derived from the spline
                
            AC = drv_spl(IF)  # derives the phase acceleration of all traces

            vm = np.percentile(AC, 95)
            cmin = -vm  
            cmax = vm
            palet = sigpal
            tit = 'Phase Accel'

            cminAC, cmaxAC, cmapAC = img_show(segyfile, tit, AC, tri, trf, spi, spf, sr, cmin, cmax, palet, sigfilt, trace_index)

            btn9['state'] = 'normal'

        # ----------------------------------------------------------------------------

        def save_ia():

            scene = [IA, cminIA, cmaxIA, tit, sigfilt, cmapIA]
            scenes.loc[len(scenes)] = scene

            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\n Inst Amplitude saved')

            logfile = open(filename+'.log', 'a')
            logfile.write('\nInstant Amplitude (Envelope) '+sigfilt)

        # ----------------------------------------------------------------------------

        def save_ia_segy():

            save_segy(IA, 'IA')

            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\nInstant Amplitude Saved as SEG-Y')    

        # ----------------------------------------------------------------------------

        def save_if():

            scene = [IF, cminIF, cmaxIF, tit, sigfilt, cmapIF]
            scenes.loc[len(scenes)] = scene


            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\nInst Frequency saved')

            logfile = open(filename+'.log', 'a')
            logfile.write('\nInstant Frequency '+sigfilt)

        # ----------------------------------------------------------------------------

        def save_if_segy():

            save_segy(IF, 'IF')

            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\nInstant FrequencY Saved as SEG-Y')    

        # ----------------------------------------------------------------------------

        def save_pcos():

            scene = [PC, cminPC, cmaxPC, tit, sigfilt, cmapPC]
            scenes.loc[len(scenes)] = scene

            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\nPhase cosine saved')

            logfile = open(filename+'.log', 'a')
            logfile.write('\nPhase cosine '+sigfilt)

        # ----------------------------------------------------------------------------

        def save_pcos_segy():

            save_segy(PC, 'PC')

            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\nPhase Cosine Saved as SEG-Y')   

        # -----------------------------------------------------------------------------------------

        def save_accel():

            scene = [AC, cminAC, cmaxAC, tit, sigfilt, cmapAC]
            scenes.loc[len(scenes)] = scene

            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\nPhase acceleration saved')

            logfile = open(filename+'.log', 'a')
            logfile.write('\nPhase acceleration '+sigfilt)

        # ----------------------------------------------------------------------------

        def save_accel_segy():

            save_segy(AC, 'AC')

            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\nPhase Acceleration Saved as SEG-Y')   

        # -----------------------------------------------------------------------------------------

        transf = Toplevel(ampl_filters)
        transf.title('HILBERT TRANSFORM')
        # transf.iconbitmap("logotet.ico")
        w = 370 # window width
        h = 400 # window height
        ws = transf.winfo_screenwidth()
        hs = transf.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        transf.geometry('%dx%d+%d+%d' % (w, h, x, y))
        transf.resizable(width=False, height=False)
        transf.attributes('-toolwindow', True) 

        txt_edit.insert(END, '\n')
        txt_edit.insert(END, '\nHilbert transform applied: ')

        btn1 = Button(transf, width=20, text ='Calc Hilbert Transform', command= calc_hilb)
        btn1.place(x=100, y=80)

        btn2 = Button(transf, width=15, text='Inst Amplitude', command= show_ia )
        btn2.place(x=50, y=130)
        btn3 = Button(transf, width=15, text ='Save', command= save_ia )
        btn3.place(x=50, y=165)
        # btn4 = Button(transf, width=15, text ='Save SEG-Y', command= save_ia_segy )
        # btn4.place(x=50, y=200)

        btn5 = Button(transf, width=15, text='Inst Frequency', command= show_if )
        btn5.place(x=200, y=130)
        btn6 = Button(transf, width=15, text ='Save', command= save_if )
        btn6.place(x=200, y=165)    
        # btn7 = Button(transf, width=15, text ='Save SEG-Y', command= save_if_segy )
        # btn7.place(x=200, y=200)

        btn8 = Button(transf, width=15, text='Phase Cosine', command= show_pcos )
        btn8.place(x=50, y=250)        
        btn9 = Button(transf, width=15, text ='Save', command= save_pcos )
        btn9.place(x=50, y=285)
        # btn10 = Button(transf, width=15, text ='Save SEG-Y', command= save_pcos_segy )
        # btn10.place(x=50, y=320)

        btn8 = Button(transf, width=15, text='Phase Accel', command= show_accel )
        btn8.place(x=200, y=250)        
        btn9 = Button(transf, width=15, text ='Save', command= save_accel )
        btn9.place(x=200, y=285)
        # btn10 = Button(transf, width=15, text ='Save SEG-Y', command= save_accel_segy )
        # btn10.place(x=200, y=320)

        select(transf, 130, 30)

    # -----------------------------------------------------------------------------------------

    ampl_filters = Toplevel(root)
    ampl_filters.title('AMPLITUDE FILTERS')
    # ampl_filters.iconbitmap("logotet.ico")
    w = 200 # window width
    h = 270 # window height
    ws = ampl_filters.winfo_screenwidth()
    hs = ampl_filters.winfo_screenheight()
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    ampl_filters.geometry('%dx%d+%d+%d' % (w, h, x, y))
    ampl_filters.resizable(width=False, height=False)
    ampl_filters.attributes('-toolwindow', True) 


    btn_agc = Button(ampl_filters, width=12, text='AGC', command=agc_gain)
    btn_agc.place(x=50, y= 40)

    btn_mgain = Button(ampl_filters, width=12, text='MANUAL GAIN', command=manual_gain)
    btn_mgain.place(x=50, y= 90)

    btn_stalta = Button(ampl_filters, width=12, text='STA/LTA', command=stalta)
    btn_stalta.place(x=50, y= 140)

    if datatype == 1:          
        btn_hilb = Button(ampl_filters, width=12, text='HILBERT', state=DISABLED, command=hilbert)
        btn_hilb.place(x=50, y= 190)
    else:        
        btn_hilb = Button(ampl_filters, width=12, text='HILBERT', command=hilbert)
        btn_hilb.place(x=50, y= 190)

# ===================================================================================================

def autocorrelation(x):
    """
    Compute the autocorrelation of the signal, based on the properties of the
    power spectral density of the signal.

    https://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation
    """
    xp = x-np.mean(x)
    f = np.fft.fft(xp)
    p = np.array([np.real(v)**2+np.imag(v)**2 for v in f])
    pi = np.fft.ifft(p)
    return np.real(pi)[:int(x.size/2)]/np.sum(xp**2)

# ----------------------------------------------------------------------------

def ricker(s):
    # waveket Ricker
    points = s  # number of samples per pulse
    a = 4.0  # constant ???
    return signal.ricker(points, a)

# ----------------------------------------------------------------------------

def conv_deconv():

    def conv_chirp():

        def conv_chirp_ricker():
            global RIK, cminRIK, cmaxRIK, cmapRIK, tit, siz

            # convolve using the Ricker wavelet

            pw = e_PW.get()
            duration = float(pw)

            fs = 1 / sr  # sampling frequency
            samples = fs*duration
            t = np.arange(int(samples)) / fs
            siz = t.size  # number of Pulse Width samples

            rick = []
            rik = signal.ricker(siz, 4.0)

            for tr in range(trf-tri):
                r = np.convolve(sigdata[tr], rik, mode='same')  
                rick = np.append(rick, r)

            n_sp = int(len(rick)/(trf - tri))
            RIK= np.reshape(rick, (trf - tri, n_sp))

            vm = np.percentile(RIK, 95)
            cmin = -vm
            cmax = vm
            palet = sigpal
            tit = f' Convolution of Signal with Ricker (pulse width={pw} s)'

            cminRIK, cmaxRIK, cmapRIK = img_show(segyfile, tit, RIK, tri, trf, spi, spf, sr, cmin, cmax, palet, sigfilt, trace_index)
       
        # ----------------------------------------------------------------------------

        def save_rick_conv():

            scene = [RIK, cminRIK, cmaxRIK, tit , sigfilt, cmapRIK]
            scenes.loc[len(scenes)] = scene            

            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\nRicker convolution saved - Wavelet Ricker size= '+str(siz))

            logfile = open(filename+'.log', 'a')
            logfile.write('\nRicker Convolution'+sigfilt)

        # ----------------------------------------------------------------------------

        def save_rick_conv_sgy():

            save_segy(RIK, 'RIK')

            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\nRicker Convolution Saved as SEG-Y')    

        # -----------------------------------------------------------------------------------------

        conv_chirp = Toplevel(conv_deconv)
        conv_chirp.title('RICKER CONVOLUTION')
        # chirp.iconbitmap("logotet.ico")
        w = 300 # window width
        h = 270 # window height
        ws = conv_deconv.winfo_screenwidth()
        hs = conv_deconv.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        conv_chirp.geometry('%dx%d+%d+%d' % (w, h, x, y))
        conv_chirp.resizable(width=False, height=False)
        conv_chirp.attributes('-toolwindow', True) 

        lbl4 = Label(conv_chirp, text='Pulse Width (s): ')
        lbl4.place(x=75, y=70)
        e_PW = Entry(conv_chirp, width=5)
        e_PW.place(x=165, y=70)

        btn1= Button(conv_chirp, width=20, text='Ricker Convol', command=conv_chirp_ricker)
        btn1.place(x=70, y= 110)

        btn2 = Button(conv_chirp, width=20, text=' Save Ricker Convol ', command= save_rick_conv)
        btn2.place(x=70, y= 150)

        # btn3 = Button(conv_chirp, width=20, text=' Save SEG-Y ', command= save_rick_conv_sgy)
        # btn3.place(x=70, y= 190)

        select(conv_chirp, 100, 30)

    # =============================================================================================================

    def deconv_boom():    

        def moving_average(x, w):
            return np.convolve(x, np.ones(w), 'same') / w
        
        # -----------------------------------------------------------------------------------

        def view_section():

            global x1, y1, x2, y2, soma, t1, t2, s1, s2



            fig, ax = plt.subplots(figsize=(14,8))
            vm = np.percentile(sigdata, 95)
            
            ax.set_title('Press ''t'' to toggle between ZOOM IMAGE WINDOW and CAPTURE BOTTOM WINDOW')
            ax.imshow(sigdata.T, vmin= -vm, vmax= vm, cmap='gray')

            def line_select_callback(eclick, erelease):
                global x1, y1, x2, y2
                x1, y1 = eclick.xdata, eclick.ydata
                x2, y2 = erelease.xdata, erelease.ydata

                rect = plt.Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2) )
                ax.add_patch(rect)

            rs = RectangleSelector(ax, line_select_callback,
                                drawtype='box', useblit=True, button=[1], 
                                minspanx=3, minspany=3, spancoords='pixels', 
                                interactive=True)

            plt.show()

        # -----------------------------------------------------------------------------------

        def extract_pulse():
            global soma, t1, t2, s1, s2

            # With the limits of the background area we add the pulses within that window

            t1 = int(x1)
            t2 = int(x2)
            s1 = int(y1)
            s2 = int(y2)

            bot_win = sigdata[t1:t2, s1:s2]

            fig, ax = plt.subplots(figsize=(6,4))    # bottom window display
            vm = np.percentile(bot_win, 95)
            plt.title('Window of the bottom reflection area')
            plt.imshow(bot_win.T, vmin = -vm, vmax=vm, cmap='gray')
            plt.tight_layout()
            plt.show()

            
            def extractSmax(trace, sta, lta):   # extracts Smax (sample nuymber) from the trace

                I = classic_sta_lta_py(trace, sta, lta)
                Ymax = max(I)
                I1 = I.tolist()
                Smax = I1.index(Ymax)
                return Smax
                
            Smax = []
            n_sta = 10
            n_lta = 20  # using the ones that worked best

            sigtrace = np.copy(bot_win)

            for tr in range(t2-t1):  

                smax = extractSmax(sigtrace[tr], n_sta, n_lta) 
                Smax = np.append(Smax, smax)  # sample number that contains the max intensity of each trace
            

            fator = 1.5                                # 1.5 is the multiplication factor
            q75, q25 = np.percentile(Smax, [75, 25])   # returns the third and first quartile
            iqr = q75 - q25                            # calculates the iqr(interquartile range)

            lowpass = q25 - (iqr * fator)              # calculates the minimum value to apply to the filter
            highpass = q75 + (iqr * fator)             # calculates the maximum value to apply to the filter

            for i in range(t2-t1):          # interpolates the range of the outlier
                if Smax[i] > highpass:
                    Smax[i] = Smax[i-1] # use the previous

                if Smax[i] < lowpass:
                    Smax[i] = Smax[i-1] # use the previous

            
            win = 5    # applying the moving average in a window of win samples
            sma = moving_average(Smax, win)


            w = win//2
            last = len(sma)-1
            for i in range(w):
                sma[i] = sma[w]
            for i in range(w):
                sma[last-i] = sma[last-w]

            # Creating the cubic spline over the moving average
            nt = len(sma)
            x = np.arange(nt)
            cs = interpolate.CubicSpline(x, sma, axis=0)
            cs_x = np.arange(0, nt)
            cs_y = cs(np.arange(0, nt))

            # difference between the spline and theshold at each point
            # corresponds to the number of samples between the two
            dif = (cs_y - Smax) 
            dif = dif.astype(int)

            # shifts the strokes according to the difference (option using shift function)
            shft = []
            for i in range(nt):
                shft.append(shift(data[i,:], dif[i], cval=0))

            correct = np.nan_to_num(shft).astype(int)

            # the pulses in this window will be summed to produce a single mean pulse
            plt.figure(figsize=(3, 8))

            x = np.arange(s2 - s1)
            for t in range(t1, t2-1):
                tr1 = sigdata[t, s1:s2] # first trace
                tr2 = sigdata[t+1, s1:s2] # second trace
                soma = [tr1[i] + tr2[i] for i in range(len(tr1))]
                plt.plot(tr1, -x)
                plt.plot(tr2, -x)
                
            plt.plot(soma, -x, '-', color='r', linewidth=3)
            plt.axvline(x=0.0, color='k', linestyle='-')
            plt.title('Sum of Traces')
            plt.savefig('sum_of_traces')
            plt.show()

            # soma = np.array(soma)
            ap = autocorrelation(np.array(soma))
            plt.figure(figsize=(4,4))
            plt.plot(ap)
            plt.title('Wavetet of Autocorrelation')
            plt.savefig('wavelet_autocorr_sum')
            plt.show()

        # -----------------------------------------------------------------------------------
        
        def decon_bottom():   
            global bDEC, cminbDEC, cmaxbDEC, cmapbDEC, tit

            # deconvolve with the autocorrelation wavelet

            bDEC = []
            for i in range(trf-tri):
                ac1 = autocorrelation(np.array(soma))
                dec, remainder = signal.deconvolve(sigdata[i], ac1)  
                bDEC = np.append(bDEC, dec)

            sp = int(len(bDEC)/(trf-tri))
            bDEC = np.reshape(bDEC, (trf-tri, sp))

            vm = np.percentile(bDEC, 95)
            cmin = -vm
            cmax = vm 
            palet = sigpal 
            tit = 'Deconvolution using Bottom Signal Capture'

            cminbDEC, cmaxbDEC, cmapbDEC = img_show(filename, tit, bDEC, tri, trf, spi, spf, sr, cmin, cmax, palet, sigfilt, trace_index)

        # -----------------------------------------------------------------------------------

        def predictive_deconvolution(trace, pred_order):
            """
            Aplica deconvolução preditiva a um traço sísmico.
            
            Parameters:
                trace (numpy array): Traço sísmico de entrada.
                pred_order (int): Ordem do filtro preditivo.
            
            Returns:
                numpy array: Traço sísmico após deconvolução preditiva.
            """
            
            # Calcular autocorrelação do traço
            autocorr = np.correlate(trace, trace, mode='full')
            mid = len(autocorr) // 2
            R = autocorr[mid:mid + pred_order]  # Apenas parte positiva

            # Construir matriz Toeplitz
            R_matrix = toeplitz(R[:-1])
            b = -R[1:]

            # Resolver equação normal para obter coeficientes do filtro preditivo
            filter_coeffs = np.linalg.solve(R_matrix, b)
            filter_coeffs = np.concatenate(([1], filter_coeffs))  # Adicionar coeficiente inicial

            # Aplicar o filtro ao traço (convolução)
            deconvolved_trace = np.convolve(trace, filter_coeffs, mode='same')
            
            return deconvolved_trace


        # -----------------------------------------------------------------------------------

        def pred_decon():

            global prDEC, cminprDEC, cmaxprDEC, cmapprDEC, tit

            # deconvolve with toeplitz matrix
            prDEC = []
            for i in range(trf-tri):
                dec = predictive_deconvolution(sigdata[i], 2) 
                prDEC = np.append(prDEC, dec)

            sp = int(len(prDEC)/(trf-tri))
            prDEC = np.reshape(prDEC, (trf-tri, sp))

            vm = np.percentile(prDEC, 95)
            cmin = -vm
            cmax = vm 
            # palet = sigpal 
            tit = 'Predicted Deconvolution using Toeplitz matrix'

            cminprDEC, cmaxprDEC, cmapprDEC = img_show(filename, tit, prDEC, tri, trf, spi, spf, sr, cmin, cmax, palet, sigfilt, trace_index)

        # -----------------------------------------------------------------------------------

        def decon_pulse(signal_segment, pulse):
            """
            Realiza a deconvolução de um segmento do sinal sísmico com um pulso conhecido.

            Parameters:
            signal_segment (np.array): Segmento do sinal sísmico
            pulse (np.array): Pulso Chirp conhecido

            Returns:
            np.array: Segmento deconvoluído
            """
            # Transformada de Fourier do segmento e do pulso
            signal_fft = fft(signal_segment)
            pulse_fft = fft(pulse, n=len(signal_segment))
            
            # Deconvolução no domínio da frequência
            deconvolved_fft = signal_fft / pulse_fft
            
            # Transformada inversa para voltar ao domínio do tempo
            deconvolved_segment = np.real(ifft(deconvolved_fft))
            
            return deconvolved_segment

        # -----------------------------------------------------------------------------------

        def decon_with_pulse():

            global pDEC, cminpDEC, cmaxpDEC, cmappDEC, tit

            # deconvolve with the autocorrelation wavelet

            pDEC = []
            for i in range(trf-tri):
                ac = autocorrelation(np.array(sigdata[i]))
                dec = decon_pulse(sigdata[i], ac)
                pDEC = np.append(pDEC, dec)

            sp = int(len(pDEC)/(trf-tri))
            pDEC = np.reshape(pDEC, (trf-tri, sp))

            vm = np.percentile(pDEC, 95)
            cmin = -vm
            cmax = vm 
            # palet = sigpal 
            tit = 'Deconvolution with Signal Autocorrelation'

            cminpDEC, cmaxpDEC, cmappDEC = img_show(filename, tit, pDEC, tri, trf, spi, spf, sr, cmin, cmax, palet, sigfilt, trace_index)


        # -----------------------------------------------------------------------------------
        def save_decon_bottom():

            scene = [bDEC, cminbDEC, cmaxbDEC, tit, sigfilt, cmapbDEC]
            scenes.loc[len(scenes)] = scene

            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\nDeconvolution saved')
            txt_edit.insert(END, '\nBottom Window between traces '+str(t1)+' and '+str(t2))
            txt_edit.insert(END, '\nBottom Window between samples '+str(s1)+' and '+str(s2))

            logfile = open(filename+'.log', 'a')
            logfile.write('\nDeconvolution with bottom'+sigfilt)

            logfile.write('\nBottom Window between traces '+str(t1)+' and '+str(t2))
            logfile.write('\nBottom Window between samples '+str(s1)+' and '+str(s2))

            # deconv_boom.destroy()

        # -----------------------------------------------------------------------------------

        def save_decon_pulse():

            scene = [pDEC, cminpDEC, cmaxpDEC, tit, sigfilt, cmappDEC]
            scenes.loc[len(scenes)] = scene

            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\nDeconvolution saved')
            txt_edit.insert(END, '\nBottom Window between traces '+str(t1)+' and '+str(t2))
            txt_edit.insert(END, '\nBottom Window between samples '+str(s1)+' and '+str(s2))

            logfile = open(filename+'.log', 'a')
            logfile.write('\nDeconvolution with pulse'+sigfilt)

            logfile.write('\nBottom Window between traces '+str(t1)+' and '+str(t2))
            logfile.write('\nBottom Window between samples '+str(s1)+' and '+str(s2))

            # deconv_boom.destroy()

        # -----------------------------------------------------------------------------------

        def save_pred_decon():

            scene = [prDEC, cminprDEC, cmaxprDEC, tit, sigfilt, cmapprDEC]
            scenes.loc[len(scenes)] = scene

            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\nPreditive Deconvolution saved')
            txt_edit.insert(END, '\nBottom Window between traces '+str(t1)+' and '+str(t2))
            txt_edit.insert(END, '\nBottom Window between samples '+str(s1)+' and '+str(s2))

            logfile = open(filename+'.log', 'a')
            logfile.write('\nPreditive Deconvolution'+sigfilt)

            logfile.write('\nBottom Window between traces '+str(t1)+' and '+str(t2))
            logfile.write('\nBottom Window between samples '+str(s1)+' and '+str(s2))

            # deconv_boom.destroy()

        # -----------------------------------------------------------------------------------

        def save_deconv_sgy():

            save_segy(bDEC, 'DC')

            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\nBoomer deconvolution saved as SEG-Y')

        # -----------------------------------------------------------------------------------

        deconv_boom = Toplevel(conv_deconv)
        deconv_boom.title('DECONVOLUTION')
        # deconvolution.iconbitmap("logotet.ico")
        w = 320 # window width
        h = 480 # window height
        ws = deconv_boom.winfo_screenwidth()
        hs = deconv_boom.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        deconv_boom.geometry('%dx%d+%d+%d' % (w, h, x, y))
        deconv_boom.resizable(width=False, height=False)
        deconv_boom.attributes('-toolwindow', True)   

        lbl1 = Label(deconv_boom, text='Select Seabottom Window Pulse')
        lbl1.place(x=75, y=65)

        btn1 = Button(deconv_boom, width=15, text='Show Image', command= view_section)
        btn1.place(x=100, y=90)

        btn2 = Button(deconv_boom, width=15, text='Show Pulse', command= extract_pulse)
        btn2.place(x=100, y= 130)

        btn3 = Button(deconv_boom, width=20, text='Deconvol with Bottom', command= decon_bottom)
        btn3.place(x=80, y= 170)

        btn6 = Button(deconv_boom, width=20, text='Save', command= save_decon_bottom)
        btn6.place(x=80, y= 210)  

        btn4 = Button(deconv_boom, width=20, text='Deconvol with Pulse', command= decon_with_pulse)
        btn4.place(x=80, y= 250)

        btn6 = Button(deconv_boom, width=20, text='Save', command= save_decon_pulse)
        btn6.place(x=80, y= 290)  

        btn5 = Button(deconv_boom, width=20, text='Preditive Deconvol', command= pred_decon)
        btn5.place(x=80, y= 330)

        btn6 = Button(deconv_boom, width=20, text='Save', command= save_pred_decon)
        btn6.place(x=80, y= 370)        

        select(deconv_boom, 110, 30)

    # -----------------------------------------------------------------------------------

    conv_deconv = Toplevel(root)
    conv_deconv.title('CONV/DECONV')
    # conv_deconv.iconbitmap("logotet.ico")
    w = 200 # window width
    h = 200 # window height
    ws = conv_deconv.winfo_screenwidth()
    hs = conv_deconv.winfo_screenheight()
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    conv_deconv.geometry('%dx%d+%d+%d' % (w, h, x, y))
    conv_deconv.resizable(width=False, height=False)
    conv_deconv.attributes('-toolwindow', True)        

    btn1 = Button(conv_deconv, width=15, text='CONVOLUTION', command= conv_chirp)
    btn1.place(x=40, y=40)
    btn2 = Button(conv_deconv, width=15, text='DECONVOLUTION', command= deconv_boom)
    btn2.place(x=40, y= 100)

# ===================================================================================================   

def imag_filters():

    def laplace():
        global LAP, cminLAP, cmaxLAP, cmapLAP, tit

        LAP = ndimage.laplace(sigdata) # apply the filter

        vm = np.percentile(LAP, 95)
        cmin = -vm  
        cmax = vm
        palet = sigpal
        tit = 'Laplace Filter'

        cminLAP, cmaxLAP, cmapLAP = img_show(segyfile, tit, LAP, tri, trf, spi, spf, sr, cmin, cmax, palet, sigfilt, trace_index)

    # ----------------------------------------------------------------------------        
        
    def x_deriv():
        global DX, cminDX, cmaxDX, cmapDX, tit
        
        DX = np.diff(sigdata, axis = 0)  # compute x-derivative

        vm = np.percentile(DX, 95)
        cmin = -vm  
        cmax = vm
        palet = sigpal
        tit = 'derivative in x-direction'

        cminDX, cmaxDX, cmapDX = img_show(segyfile, tit, DX, tri, trf, spi, spf, sr, cmin, cmax, palet, sigfilt, trace_index)

    # ----------------------------------------------------------------------------

    def y_deriv():
        global DY, cminDY, cmaxDY, cmapDY, tit


        DY = np.diff(sigdata, axis = 1)  # compute y-derivative

        vm = np.percentile(DY, 95)
        cmin = -vm  
        cmax = vm
        palet = sigpal
        tit = 'derivative in y-direction'

        cminDY, cmaxDY, cmapDY = img_show(segyfile, tit, DY, tri, trf, spi, spf, sr, cmin, cmax, palet, sigfilt, trace_index)

    # ----------------------------------------------------------------------------

    def wiener():
        global WN, cminWN, cmaxWN, cmapWN, tit
        
        WN = signal.wiener(sigdata) # Reduces noise but blurs the image a little     

        vm = np.percentile(WN, 95)
        cmin = -vm  
        cmax = vm
        palet = sigpal
        tit = 'Wiener Filter'

        cminWN, cmaxWN, cmapWN = img_show(segyfile, tit, WN, tri, trf, spi, spf, sr, cmin, cmax, palet, sigfilt, trace_index)

    # ----------------------------------------------------------------------------

    def save_lap():

        scene = [LAP, cminLAP, cmaxLAP, tit, sigfilt, cmapLAP]
        scenes.loc[len(scenes)] = scene


        txt_edit.insert(END, '\n')      
        txt_edit.insert(END, '\nLaplace Filter saved')

        logfile = open(filename+'.log', 'a')
        logfile.write('\nLaplace Filter'+sigfilt)

    # ----------------------------------------------------------------------------

    def save_lap_segy():

        save_segy(LAP, 'LP')

        txt_edit.insert(END, '\n')
        txt_edit.insert(END, '\nLaplace filter Saved as SEG-Y')    

    # ---------------------------------------------------------------------------- 
       
    def save_x_deriv():

        scene = [DX, cminDX, cmaxDX, tit, sigfilt, cmapDX]
        scenes.loc[len(scenes)] = scene

        txt_edit.insert(END, '\n')      
        txt_edit.insert(END, '\nx-Derivative Filter saved')

        logfile = open(filename+'.log', 'a')
        logfile.write('\nx_Derivative Filter'+sigfilt)

    # ----------------------------------------------------------------------------

    def save_x_deriv_segy():

        save_segy(DX, 'Dx')

        txt_edit.insert(END, '\n')
        txt_edit.insert(END, '\n x-Derivative filter Saved as SEG-Y')    

    # ----------------------------------------------------------------------------
     
    def save_y_deriv():

        scene = [DY, cminDY, cmaxDY, tit, sigfilt, cmapDY]
        scenes.loc[len(scenes)] = scene

        txt_edit.insert(END, '\n')      
        txt_edit.insert(END, '\ny-Derivative Filter saved')

        logfile = open(filename+'.log', 'a')
        logfile.write('\ny-Derivative Filter'+sigfilt)

    # ----------------------------------------------------------------------------

    def save_y_deriv_segy():

        save_segy(DY, 'Dy')

        txt_edit.insert(END, '\n')
        txt_edit.insert(END, '\ny-Derivative filter Saved as SEG-Y')    

    # ----------------------------------------------------------------------------
     
    def save_wiener():

        scene = [WN, cminWN, cmaxWN, tit, sigfilt, cmapWN]
        scenes.loc[len(scenes)] = scene

        txt_edit.insert(END, '\n')      
        txt_edit.insert(END, '\nWiener Filter saved')

        logfile = open(filename+'.log', 'a')
        logfile.write('\nWiener Filter'+sigfilt)

    # ----------------------------------------------------------------------------

    def save_wiener_segy():

        save_segy(WN, 'WN')

        txt_edit.insert(END, '\n')
        txt_edit.insert(END, '\nWiener filter Saved as SEG-Y')    

    # ---------------------------------------------------------------------------- 

    enhance = Toplevel(root)
    enhance.title('ENHANCES IMAGE')
    # enhance.iconbitmap("logotet.ico")
    w = 360 # window width
    h = 370 # window height
    ws = enhance.winfo_screenwidth()
    hs = enhance.winfo_screenheight()
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    enhance.geometry('%dx%d+%d+%d' % (w, h, x, y))
    enhance.resizable(width=False, height=False)
    enhance.attributes('-toolwindow', True) 

    btn1 = Button(enhance, text='Laplace', width=15, command=laplace)
    btn1.place(x=50, y=90)    
    btn2 = Button(enhance, text='Save', width=15, command=save_lap)
    btn2.place(x=50, y=125)
    # btn3 = Button(enhance, text='Save SEG-Y', width=15, command=save_lap_segy)
    # btn3.place(x=50, y=160)


    btn4 = Button(enhance, text='x-derivate', width=15, command=x_deriv)
    btn4.place(x=200, y=90)
    btn5 = Button(enhance, text='Save', width=15, command=save_x_deriv)
    btn5.place(x=200, y=125)
    # btn6 = Button(enhance, text='Save SEG-Y', width=15, command=save_x_deriv_segy)
    # btn6.place(x=200, y=160)

    btn7 = Button(enhance, text='y-derivate', width=15, command=y_deriv)
    btn7.place(x=50, y=210)
    btn8 = Button(enhance, text='Save', width=15, command=save_y_deriv)
    btn8.place(x=50, y=245)
    # btn9 = Button(enhance, text='Save SEG-Y', width=15, command=save_y_deriv_segy)
    # btn9.place(x=50, y=280)

    btn10 = Button(enhance, text='Wiener', width=15, command= wiener)
    btn10.place(x=200, y=210)
    btn11 = Button(enhance, text='Save', width=15, command=save_wiener)
    btn11.place(x=200, y=245)
    # btn12 = Button(enhance, text='Save SEG-Y', width=15, command=save_wiener_segy)
    # btn12.place(x=200, y=280)

    select(enhance, 120, 40)  

# ===================================================================================================

def interpret():

    def intpret():
        global hrz, ref

        class PolygonDrawer:
            
            def __init__(self, digit):
                self.n = -1
                self.rf = 0       
                self.cresc = True
                self.refs = []
                self.hrz = []
                self.digit = digit
                self.digit.title("Draw polygons")
                self.image_path = ""
                self.fig, self.ax = plt.subplots(facecolor='lightblue')
                self.canvas = FigureCanvasTkAgg(self.fig, master=self.digit)
                self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
                # self.toolbar = NavigationToolbar2Tk(self.canvas, self.digit)
                # self.toolbar.update()
                self.canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)
                self.polygons = []
                self.current_polygon = []
                self.colors = ['red', 'green', 'blue', 'orange', 'purple']  # Lista de cores para os polígonos
                self.current_color_index = 0  # Índice da cor atual
                self.canvas.mpl_connect('button_press_event', self.on_click)
                self.canvas.mpl_connect('button_release_event', self.on_release)

                self.button_frame = Frame(self.digit)
                self.button_frame.pack(side=RIGHT)

                self.print_button = Button(self.button_frame, text="   Exit   ", command=self.exit)
                self.print_button.pack(side=RIGHT)
                self.save_button = Button(self.button_frame, text=" Save Image ", command=self.save_image)
                self.save_button.pack(side=RIGHT)
                self.load_button = Button(self.button_frame, text="  Change Image  ", command=self.load_image)
                self.load_button.pack(side=RIGHT)
                self.load_image()
                self.finalize_button = Button(self.button_frame, text="  Finish Reflector  ", command=self.finalize_polygon)
                self.finalize_button.pack(side=RIGHT)


            def load_image(self):

                if self.n < len(scenes)-1 and self.cresc == True: 
  
                    param = scenes.iloc[self.n]    # param = field values in scenes[x]
                    imgsig = param[0]
                    imgcmin = param[1]
                    imgcmax = param[2]
                    imgtit = param[3]
                    imgfilt = param[4]
                    imgcmap = param[5]

                    # self.ax.clear()

                    self.ax.set_title(segyfile + '  -  Filter: '+ imgtit + imgfilt)
                    self.ax.imshow(imgsig.T, vmin=imgcmin, vmax=imgcmax, cmap=imgcmap, aspect='auto', interpolation='nearest', extent=ext)
                    
                    self.n += 1  

                else:
                    self.cresc = False
                    if self.n > 0:
                        param = scenes.iloc[self.n]    # param = field values in scenes[x]
                        imgsig = param[0]
                        imgcmin = param[1]
                        imgcmax = param[2]
                        imgtit = param[3]
                        imgfilt = param[4]
                        imgcmap = param[5]

                        # self.ax.clear()

                        self.ax.set_title(segyfile + '  -  Filtro: '+ imgtit + imgfilt)
                        self.ax.imshow(imgsig.T, vmin=imgcmin, vmax=imgcmax, cmap=imgcmap, aspect='auto', interpolation='nearest', extent=ext)

                        self.n -= 1

                        if self.n == 0:
                            self.cresc = True

                # self.ax.imshow(image)
                self.draw_polygons()
                self.ax.set_aspect('auto')
                self.fig.tight_layout()  
                self.canvas.draw()
                self.current_polygon = []   

            def save_image(self):
                file_path = asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG files", "*.png"),
                                                        ("JPEG files", "*.jpg"),
                                                        ("All Files", "*.*")])
                if file_path:
                    self.fig.savefig(file_path, dpi=300)
                    txt_edit.insert(END, '\n')
                    txt_edit.insert(END, '\n Horizonts exported to: '+str(file_path))
                    txt_edit.insert(END, f)      

            def on_click(self, event):
                if event.button == 1:
                    if event.xdata is not None and event.ydata is not None:
                        self.current_polygon.append((event.xdata, event.ydata))
                        self.ax.plot(event.xdata, event.ydata, 'ro', color=self.colors[self.current_color_index])
                        self.canvas.draw()

                        if len(self.current_polygon) > 1:
                            x = [coord[0] for coord in self.current_polygon[-2:]]
                            y = [coord[1] for coord in self.current_polygon[-2:]]
                            self.ax.plot(x, y, '-', color=self.colors[self.current_color_index])

            def on_release(self, event):
                if event.button == 3:
                    if len(self.current_polygon) > 0:
                        self.current_polygon.pop()
                        self.ax.lines[-1].remove()
                        self.canvas.draw()

            def clear_polygons(self):
                self.ax.clear()
                self.ax.imshow(sigdata.T, vmin=cmin, vmax=cmax, cmap=palet, aspect='auto', interpolation='nearest', extent=ext)
                self.ax.set_aspect('auto')
                self.fig.tight_layout()  
                self.canvas.draw()
                self.coordinates = []
                self.current_polygon = []
                self.polygon_coordinates = []

            def draw_polygons(self):
                for i, polygon in enumerate(self.polygons):
                    color = self.colors[i % len(self.colors)]  # Select color from color list
                    x = [coord[0] for coord in polygon]
                    y = [coord[1] for coord in polygon]
                    self.ax.plot(x, y, '-', color=color)

            def print_coordinates(self):                

                if self.polygons:
                    for ref, polygon in enumerate(self.polygons):
                        # print(f"Polígono {ref+1}:")
                        for coordinate in polygon:
                            # print(i, coordinate)
                            self.refs.append(self.rf)
                            self.hrz.append(coordinate)

                        # print(self.refs)
                        # print(self.hrz)
                        self.rf += 1

                    self.polygons = []
                    self.current_polygon = []
                else:
                    print("No polygons drawn")                

            def finalize_polygon(self):
                if self.current_polygon:
                    self.polygons.append(self.current_polygon)
                    self.current_polygon = []
                    self.current_color_index += 1  # Increments the current color index
                    if self.current_color_index > 4:
                        self.current_color_index = 0
                    self.canvas.draw()
                    self.draw_polygons()
                    self.print_coordinates()

            def exit(self):
                self.ax.clear()
                digit.destroy()

        digit = Toplevel(interp)
        digit.title('DIGITALIZE')
        # interp.iconbitmap("logotet.ico")
        w = 1600 # window width
        h = 900 # window height
        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        digit.geometry('%dx%d+%d+%d' % (w, h, x, y))
        # digit.resizable(width=False, height=False)
        digit.attributes('-toolwindow', True) 
        digit.geometry("1600x800")     

        dig = PolygonDrawer(digit)

        hrz = dig.hrz
        ref = dig.refs

        btn2['state'] = 'normal'

    # -----------------------------------------------------------------------------------------

    def save_horiz():   # save digitalized horizonts
        global hrz, ref
        
        if no_coords == True:
            messagebox.showinfo(title='Warning', message='Coordinates are Missing')
        else:
            tr = []  # array of traces
            sp = []  # array of samples
            rf = []  # array of reflectors
            i = 0

            hz = np.transpose(hrz) # change row x columns
            hrz = hz.tolist()  # ndarray to list

            for item in hrz: # fills the tr, sp and rf vectors of each scanned line
                if hrz.index(item) % 2 == 0:
                    for item2 in item:
                        tr.append(int(item2)) 
                else:
                    for item2 in item:
                        sp.append(item2)
                        rf.append(i)

                    i=i+1  
                        
            D = {'Ref': ref, 'Trace': tr, 'Sample': sp} 

            # merge with coordenates
            refs = pd.DataFrame(D, columns= ['Ref', 'Trace', 'Sample'])
            refs_coords = pd.merge(refs, coords) 

            # sort by Ref, Trace
            horizonts = refs_coords.sort_values(by=['Ref', 'Trace']) # tabela sem as interpolacoes

            vlc = int(veloc.get())
            sound_veloc = vlc/2/1000 # sound velocity

            # adjust coordinates and depth
            horizonts['Xcoord'] = abs(horizonts['SourceX'].divide(horizonts['ElevationScalar']))
            horizonts['Ycoord'] = abs(horizonts['SourceY'].divide(horizonts['ElevationScalar']))
            horizonts['Depth'] = abs(horizonts['Sample'].multiply(sound_veloc))

            size = horizonts.shape[0] # number of scanned vertices

            i = 0

            R = horizonts['Ref']
            T = horizonts['Trace']
            X = horizonts['Xcoord']
            Y = horizonts['Ycoord']
            D = horizonts['Depth']

            Ri = []
            Ti = []
            Xi = []
            Yi = []
            Di = []
            Ai = []

            r1 = R[i]  # first row data
            t1 = T[i]
            x1 = X[i]
            y1 = Y[i]
            d1 = D[i]

            while i < size-1:  # for all vertices

                if R[i] == R[i+1]:  # if the same reflector

                    t2 = T[i+1]  # take the next vertex
                    x2 = X[i+1]
                    y2 = Y[i+1]
                    d2 = D[i+1]

                    interv = t2 - t1 + 1 # calculates interval between traces

                    r1 = R[i]
                    r = [r1] * interv # fills the vector r with the reflector number
                    t = [t1, t2]  # trace numbers in range
                    x = [x1, x2]
                    y = [y1, y2]
                    d = [d1, d2]   # depths in range

                    t_new = np.linspace(t1, t2, interv)     # array of tr in range 

                    function = interp1d(t, x)
                    x = function(t_new) # calculates x in the range

                    function = interp1d(t, y)
                    y = function(t_new)  # calculates y in the range

                    function = interp1d(t, d)
                    d = function(t_new)  # calculates d in the range

                    t_int = t_new.astype(int)   # trace to integer                
                    d_int = d.astype(int)   # depth to integer
                    
                    A = sigdata[t_int, d_int]  # calculates Amplit (Intensity) in sigdata

                    # next vertex will now be the current one
                    t1 = t2  
                    x1 = x2
                    y1 = y2
                    d1 = d2
                    
                    i += 1

                    # save
                    Ri = Ri + r
                    Ti = np.append(Ti, t_new)
                    Xi = np.append(Xi, x)
                    Yi = np.append(Yi, y)
                    Di = np.append(Di, -d)
                    Ai = np.append(Ai, A)


                else:

                    i += 1

                    t1 = T[i]
                    x1 = X[i]
                    y1 = Y[i]
                    d1 = D[i]

                    t2 = T[i+1]
                    x2 = X[i+1]
                    y2 = Y[i+1]
                    d2 = D[i+1]
                    
                    interv = t2 - t1 + 1

                    r1 = R[i]
                    r = [r1] * interv
                    t = [t1, t2]
                    x = [x1, x2]
                    y = [y1, y2]
                    d = [d1, d2]

                    t_new = np.linspace(t1, t2, interv)  

                    function = interp1d(t, x)
                    x = function(t_new)

                    function = interp1d(t, y)
                    y = function(t_new)

                    function = interp1d(t, d)
                    d = function(t_new)

                    t_int = t_new.astype(int)   # convert t_new to integer              
                    d_int = d.astype(int)   # convert d to integer
                    
                    A = sig[t_int, d_int]  # calculates Amplit (Intensity) in sigdata

                    # next vertex will now be the current one
                    t1 = t2
                    x1 = x2
                    y1 = y2
                    d1 = d2

                    i += 1
                    
                    # save
                    Ri = Ri + r
                    Ti = np.append(Ti, t_new)
                    Xi = np.append(Xi, np.round(x, 2)) 
                    Yi = np.append(Yi, np.round(y, 2))
                    Di = np.append(Di, np.round(-d, -1))
                    Ai = np.append(Ai, A)

            # -----------------------------------------------------------------------------------------

            H = {'Ref': Ri, 'Trace': Ti, 'Xcoord': Xi, 'Ycoord': Yi, 'Depth': Di, 'Intens': Ai}
            df = pd.DataFrame(H, columns= ['Ref','Trace','Xcoord','Ycoord','Depth','Intens'])

            df_sort = df.sort_values(by=['Ref', 'Trace'])

            # save
            f = filepath[:-4]+'_horizonts.csv'
            export_csv = df_sort.to_csv (f, index = None, header=True) 
            print('finish')

            txt_edit.insert(END, '\n')
            txt_edit.insert(END, '\n Horizonts exported to: ')
            txt_edit.insert(END, f)

    # -----------------------------------------------------------------------------------------

    interp = Toplevel(root)
    interp.title('INTERPRET')
    # interp.iconbitmap("logotet.ico")
    w = 230 # window width
    h = 180 # window height
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    interp.geometry('%dx%d+%d+%d' % (w, h, x, y))
    interp.resizable(width=False, height=False)
    interp.attributes('-toolwindow', True) 

    btn1 = Button(interp, width=15, text='Open Data', command= intpret)
    btn1.place(x=50, y= 30)

    lbl1 = Label(interp, text='Sound velocity= ')
    lbl1.place(x=40, y=80)

    veloc = StringVar(interp, value=1470)
    v = Entry(interp, width=5, textvariable=veloc)
    v.place(x=140, y=80)

    btn2= Button(interp, width=15, text='Save Horizonts', command= save_horiz)
    btn2.place(x=50, y= 115)

# =================================================================================================== 

def naveg():
    
    def exportDXF():
        filenames = askopenfilename(filetypes=[("SEG-Y", "*.sgy"), ("SEG files", "*.seg"), 
                    ("All Files", "*.*")], multiple=True)
        if not filenames:
            return
        filename = nav.tk.splitlist(filenames)            
        for filename in filenames:
            try:
                with segyio.open(filename, ignore_geometry=True) as f:
                    # Get basic attributes
                    nt = f.tracecount
                    sample_rate = segyio.tools.dt(f) / 1000000
                    n_samples = f.samples.size
                    twt = f.samples
                    data = f.trace.raw[:]  # Get all data into memory (could cause on big files)
                    sr = float(sample_rate)
                    
                    fr = segyio.TraceField.FieldRecord
                    rec_num = f.attributes(fr)[:]  # array of field record number
                    x_coord = f.attributes(segyio.TraceField.SourceX)[:]  # array of x coord
                    y_coord = f.attributes(segyio.TraceField.SourceY)[:]  # array of y coord
                    Scalar = f.attributes(segyio.TraceField.SourceGroupScalar)[:]
                    units = f.attributes(segyio.TraceField.CoordinateUnits)[:]

                    if x_coord[10] == 0:
                        messagebox.showinfo(title='Warning', message='Coordinates are Missing')
                        
                    # -----------------------------------------------------------------------------------------

                    if units[10] == 1: # meters (UTM) coordinates
                        sc = Scalar[10]
                        if sc < 0:
                            scaler = -1/sc
                        else:
                            scaler = 1/sc
                        
                        x_coord = x_coord * scaler
                        y_coord = y_coord * scaler 

                    # -----------------------------------------------------------------------------------------

                    if units[10] == 2:   # lat/long coordinates
                        sc = Scalar[10] 
                        if sc < 0:
                            scaler = -1/sc
                        else:
                            scaler = 1/sc                            
                        long = x_coord * scaler/ 3600 
                        lat = y_coord * scaler/ 3600

                        # if convUTM == 1: # lat long to UTM 
                        UTM = utm.from_latlon(lat,long)  # (easting, northing, zone_number, zone_letter)
                        x_coord = UTM[0]
                        y_coord = UTM[1]

                    # -----------------------------------------------------------------------------------------

                    f = filename.split("/")
                    segyfile = f[-1]
                    name = segyfile[:-4]

                    # -----------------------------------------------------------------------------------------

                    if chkCSV.get() == 1:
                        f = filename[:-4] +'_coord.csv'
                        D = {'TRACE_INDEX': rec_num, 'EAST': x_coord, 'NORTH': y_coord}    
                        df = pd.DataFrame(D, columns= ['TRACE_INDEX', 'EAST', 'NORTH'])
                        export_csv = df.to_csv (f, index = None, header=True) 
                        txt_edit.insert(END, '\n')
                        txt_edit.insert(END, 'Save CSV at: '+filename[:-4] +'_coord.csv')

                   # -----------------------------------------------------------------------------------------

                    drawing = dxf.drawing(filename[:-4] +'.dxf')
                    drawing.add_layer('Name', color=2)
                    drawing.add_layer('Trace_index', color=3)
                    drawing.add_layer('Line', color=4)

                    j = 0
                    for i in range(rec_num.size - 1):

                        if i == 0:
                            drawing.add(dxf.text(name, insert=(x_coord[0], y_coord[0]), height=2.0, layer='NAME'))

                        drawing.add(dxf.line((x_coord[i], y_coord[i]), (x_coord[i+1], y_coord[i+1]), color=1, layer='LINE'))

                        if j <= 3 :   # put the Field Rec Num from 5 to 5
                            j += 1
                        else:
                            trc_num = str(rec_num[i])
                            text = dxf.text(trc_num, (x_coord[i], y_coord[i]), height=0.5, rotation=0.0, layer='TRACE INDEX')
                            drawing.add(text)    
                            j = 0

                    txt_edit.insert(END, '\n')
                    txt_edit.insert(END, 'DXF save in: '+filename[:-4] +'_nav.dxf')

                # -----------------------------------------------------------------------------------------

                drawing.save() 
                #export_csv = df.to_csv (f, index = None, header=True) 
            except:            
                print("Oops!",sys.exc_info()[0]," in "+name)
                print()
            txt_edit.insert(END, '\n')

    # -----------------------------------------------------------------------------------------

    nav = Toplevel(root)
    nav.title('EXPORT DXF OF NAVIGATION')
    # nav.iconbitmap("logotet.ico")
    w = 250 # window width
    h = 170 # window height
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    nav.geometry('%dx%d+%d+%d' % (w, h, x, y))
    nav.resizable(width=False, height=False)
    nav.attributes('-toolwindow', True) 

    chkCSV = IntVar()     
    ckb1 = Checkbutton(nav, text='Include CSV', var=chkCSV, onvalue=1, offvalue=0) 
    ckb1.place(x=80, y=20)

    btn1 = Button(nav, width=18, text='Select SEG-Y files', command= exportDXF)
    btn1.place(x=50, y=60)
  
# ===================================================================================================
#    ROOT
# ===================================================================================================

root = Tk()

# s = ttk.Style()
# s.theme_use('clam')

root.title("  TETHYS   Hi-Res Monochannel Seismic Filters and Interpretation")
# root.iconbitmap("logotet.ico")
root.rowconfigure(0, minsize=450, weight=1)
root.columnconfigure(1, minsize=900, weight=1)
w = 940 # window width
h = 550 # window height
x = 300 #(ws/10) - (w/10)
y = 50 #(hs/10) - (h/100)
root.geometry('%dx%d+%d+%d' % (w, h, x, y))
root.resizable(width=False, height=False)
root.attributes('-toolwindow', True) 

txt_edit = Text(root)
fr_buttons = Frame(root, relief=RAISED, bd=2, bg='#ADEAEA')   
fr_buttons.grid(row=0, column=1)

palets = ['Greys','seismic','bwr','coolwarm','Spectral','PuOr', 
          'BrBG','PRGn','RdBu']

combox1 = ttk.Combobox(fr_buttons, textvariable = palets, state="readonly", width=10)
combox1['values'] = palets
combox1.set('Palettes')
combox1.bind("<<ComboboxSelected>>", pal_choice)

palet = 'binary'
# sty = 'ggplot'
# sty = 'seaborn-darkgrid'
# sty = 'seaborn'
# sty = 'classic'
# sty = 'default'
# plt.style.use(sty)

btn_open = Button(fr_buttons, text="OPEN", width=8, command=open_segy_file, borderwidth=3, font=('sans 9 bold'))
btn_corrections = Button(fr_buttons, text="CORRECTIONS", width=12, command=correction, borderwidth=3, state=DISABLED, font=('sans 9 bold'))
btn_freq_filters = Button(fr_buttons, text="FREQ FILTERS", width=12, command=freq_filters, borderwidth=3, state=DISABLED, font=('sans 9 bold'))
btn_ampl_filters = Button(fr_buttons, text="AMPL FILTERS", width=12, command=ampl_filter, borderwidth=3, state=DISABLED, font=('sans 9 bold'))
btn_conv_deconv = Button(fr_buttons, text="CONV/DECONV", width=15, command=conv_deconv, borderwidth=3, state=DISABLED, font=('sans 9 bold'))
btn_imag_filters = Button(fr_buttons, text="IMAGE FILTERS", width=12, command=imag_filters, borderwidth=3, state=DISABLED, font=('sans 9 bold'))
btn_interp = Button(fr_buttons, text="INTERPRET", width=12, command=interpret, borderwidth=3, state=DISABLED, font=('sans 9 bold'))
btn_naveg = Button(fr_buttons, text="NAVEGATION", width=12, command=naveg, borderwidth=3, font=('sans 9 bold'))

btn_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
btn_corrections.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
btn_freq_filters.grid(row=0, column=2, sticky="ew", padx=5, pady=5)
btn_ampl_filters.grid(row=0, column=3, sticky="ew", padx=5)
btn_conv_deconv.grid(row=0, column=4, sticky="ew", padx=5)
btn_imag_filters.grid(row=0, column=5, sticky="ew", padx=5, pady=5)
btn_interp.grid(row=0, column=6, sticky='ew', padx=5)
btn_naveg.grid(row=0, column=7, sticky="ew", padx=5)

combox1.grid(row=0, column=8, sticky="ew", padx=5) # palette

fr_buttons.grid(row=10, column=1, sticky="ew")
txt_edit.grid(row=0, column=1, sticky="nsew")

root.mainloop()