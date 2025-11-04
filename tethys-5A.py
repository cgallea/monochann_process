import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button as Btn, Slider, RectangleSelector
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import *
from tkinter import ttk, messagebox, filedialog
import segyio
import pandas as pd
from scipy import signal
from scipy.ndimage import shift
from scipy import interpolate
from scipy.linalg import toeplitz
from scipy.signal import lfilter, chirp, fftconvolve

import csv
import ezdxf
from dxfwrite import DXFEngine as dxf
import utm

mpl.rcParams['figure.dpi'] = 92
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'


class SeismicApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TETHYS - Hi-Res Monochannel Seismic Filters and Interpretation")
        self.root.geometry("850x540+300+50")
        self.root.resizable(False, False)

        self.data = None
        self.segy_file_path = None
        self.filename = None
        self.nt = self.ns = self.sr = self.twt = None
        self.sig = self.sigdata = self.sigfilt = None
        self.cmin = self.cmax = None
        self.headers = None
        self.trace_index = None
        self.coords = None
        self.text_header = None
        self.datatype = None        
        self.palet = 'binary'
        self.cmap = 'binary'  # Adicionado para corrigir referência
        self.scenes = pd.DataFrame(columns=['img', 'cmin', 'cmax', 'tit', 'sigfilt', 'cmap'])
        
        self.tri = self.trf = self.spi = self.spf = 0

        self.bp_data = self.ormsby_data = self.wt_data = None
        self.agc_data = self.stalta_data = None
        self.conv_rick_data = self.conv_ormsby_data = None
        self.deconv_bot_data = self.deconv_pred_data = None
        self.deconv_wav_data = None

        self.bp_title = self.wt_title = None
        self.agc_title = self.stalta_title = None
        self.conv_rick_title = self.conv_ormsby_title = None
        self.deconv_bot_title = self.deconv_pred_title = None
        self.deconv_wav_title = None
     
        self.soma = None  # Para deconvolução

        # Variáveis para Interpretation
        self.reflectors = {}
        self.current_reflector = None
        self.reflector_colors = {"A": "red", "B": "blue", "C": "green", "D": "orange", "E": "purple"}
        self.interpret_lines = {}
        self.modo_interpretacao = None
        self._dragging_point = None
        self._drag_reflector = None

        self._setup_ui()

    def _setup_ui(self):

        """Configura a interface principal da aplicação."""

        self.txt_edit = Text(self.root, bg="lightyellow", fg="blue", font=("Arial", 9, "bold"),
                selectbackground="orange", selectforeground="white", width=40, height=10)

        self.txt_edit.grid(row=0, column=1, sticky="nsew")

        self.fr_buttons = Frame(self.root, relief=RAISED, bd=2, bg='#ADEAEA') # painel de botões
        self.fr_buttons.grid(row=1, column=1, sticky="ew")

        self.btn_open = Button(self.fr_buttons, text="OPEN", width=8, 
                       command=self.open_segy_file, borderwidth=3, 
                       font=('sans 9 bold'), bg="#71B9F4", fg='grey20')
        self.btn_open.grid(row=0, column=0, padx=5, pady=5)

        self.btn_freq_filters = Button(self.fr_buttons, text="FREQ FILTERS", 
                               width=12, command=self.open_freq_filters_window, 
                               borderwidth=3, font=('sans 9 bold'), 
                               bg="#56AF59", fg='grey20')
        self.btn_freq_filters.grid(row=0, column=1, padx=5, pady=5)

        self.btn_ampl_filters = Button(self.fr_buttons, text="AMPLIT FILTERS", 
                                command=self.open_amplitude_filters_window, font=('sans 9 bold'),
                                bg='#56AF59', fg='grey20')
        self.btn_ampl_filters.grid(row=0, column=2, padx=5, pady=5)

        self.btn_conv = Button(self.fr_buttons, text="CONVOLUTION", 
                               command=self.open_conv_window, font=('sans 9 bold'),
                            bg="#F8C374", fg='grey20')
        self.btn_conv.grid(row=0, column=3, padx=5, pady=5)

        self.btn_deconv = Button(self.fr_buttons, text="DECONVOLUTION", 
                                 command=self.open_deconv_window, font=('sans 9 bold'),
                                 bg='#F8C374', fg='grey20')
        self.btn_deconv.grid(row=0, column=4, padx=5, pady=5)

        self.btn_interpretation = Button(self.fr_buttons, text="INTERPRETATION", 
                                         command=self.open_interpretation_window, font=('sans 9 bold'),
                                         bg="#D69494", fg='grey20')
        self.btn_interpretation.grid(row=0, column=7, padx=5, pady=5)

        btn_navigation = Button(self.fr_buttons, text="NAVIGATION", 
                                command=self.open_navigation_window, font=('sans 9 bold'),
                                bg="#6C838F", fg='grey20')
        btn_navigation.grid(row=0, column=8, padx=5, pady=5)        

        self.btn_freq_filters.config(state='disabled')
        self.btn_ampl_filters.config(state='disabled')
        self.btn_conv.config(state='disabled')
        self.btn_deconv.config(state='disabled')
        self.btn_interpretation.config(state='disabled')

        palettes = ['Greys', 'seismic', 'seismic_r', 'binary', 'Spectral', 'PuOr', 'BrBG', 'PRGn', 'RdBu']
        self.pal_combobox = ttk.Combobox(self.fr_buttons, textvariable=StringVar(value='Palettes'), 
                                         state="readonly", width=10)
        self.pal_combobox['values'] = palettes
        self.pal_combobox.set('Greys') 
        self.pal_combobox.bind("<<ComboboxSelected>>", self.pal_choice)
        self.pal_combobox.grid(row=0, column=9, padx=5)
        
        self.root.rowconfigure(0, minsize=450, weight=1)
        self.root.columnconfigure(1, minsize=900, weight=1)

    def pal_choice(self, event):
        """Define a paleta de cores para plotagem."""
        self.palet = self.pal_combobox.get()
        self.cmap = self.palet

    def open_segy_file(self):
        """Abre e lê um File SEG-Y."""
        self.segy_file_path = filedialog.askopenfilename(
            filetypes=[("SEG-Y", "*.sgy"), ("SEG files", "*.seg"), ("All Files", "*.*")]
        )
        if not self.segy_file_path:
            return

        self.filename = os.path.basename(self.segy_file_path)[:-4]

        try:
            with segyio.open(self.segy_file_path, ignore_geometry=True) as f:
                self.nt = f.tracecount
                self.sr = segyio.tools.dt(f) / 1000000
                self.ns = f.samples.size
                self.twt = f.samples
                self.data = f.trace.raw[:]
                self.sig = np.copy(self.data)

                self.nyquist = 0.5 / self.sr
                self.fs = 1 / self.sr

                # Headers
                headers = segyio.tracefield.keys
                trace_headers = pd.DataFrame(index=range(1, self.nt + 1), columns=headers.keys())
                for k, v in headers.items():
                    trace_headers[k] = f.attributes(v)[:]
                
                self.headers = trace_headers
                self.trace_index = trace_headers.FieldRecord.values
                
                # Coordinates
                coord_data = trace_headers[['TRACE_SEQUENCE_LINE','SourceX','SourceY', 'LagTimeA', 'ElevationScalar']]
                coords = coord_data.rename(columns={'TRACE_SEQUENCE_LINE': 'Trace'})
                c0 = coords['Trace'][1]
                coords['Trace'] = coords['Trace'] - c0
                self.coords = coords
                
                # Text header
                self.text_header = segyio.tools.wrap(f.text[0])

                # Bin header
                bin_header = f.bin
                self.txt_edit.insert(END, f'Bin Header:\n {bin_header}\n')
                                
                # Data type detection
                self.datatype = 0 if min(self.data[10]) < 0 else 1

                self.txt_edit.insert(END, f'\nFile:  {self.filename}\n')                
                self.txt_edit.insert(END, f'Data Type: {"Not Enveloped" if self.datatype == 0 else "Enveloped"}\n')

                self.txt_edit.insert(END, f'Traces: {self.nt}, Samples: {self.ns}, Sample Rate: {self.sr*1000:.4f} ms\n')

                self.open_work_window() # select data window

        except Exception as e:
            messagebox.showerror("Error", f"Failed to open SEG-Y file: {e}")

    def open_work_window(self):
        """Abre a janela para definir a área de trabalho."""
        self.work_window = Toplevel(self.root)
        self.work_window.title("WORKING WINDOW")
        w = 275
        h = 290
        ws = self.work_window.winfo_screenwidth()
        hs = self.work_window.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        self.work_window.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.work_window.resizable(width=False, height=False)
        self.work_window.attributes('-toolwindow', True)

        Label(self.work_window, text="Start Time (ms): ").place(x=55, y=30)
        self.e_tmi = Entry(self.work_window, width=10)
        self.e_tmi.place(x=155, y=30)
        
        Label(self.work_window, text="End Time (ms): ").place(x=55, y=50)
        self.e_tmf = Entry(self.work_window, width=10)
        self.e_tmf.place(x=155, y=50)
        
        Label(self.work_window, text="Start Trace: ").place(x=55, y=70)
        self.e_tri = Entry(self.work_window, width=10)
        self.e_tri.place(x=155, y=70)
        
        Label(self.work_window, text="End Trace: ").place(x=55, y=90)
        self.e_trf = Entry(self.work_window, width=10)
        self.e_trf.place(x=155, y=90)
        
        Button(self.work_window, width=12, text='Image View', command=self.img_view,
               font=('sans 9 bold'), bg="#307FA7", fg='white').place(x=90, y=140)
        Button(self.work_window, width=10, text='Save', command=self.save_limits,
               font=('sans 9 bold'), bg="#43AB1E", fg='white').place(x=97, y=190)

    def img_view(self):
        """Exibe a Image sísmica na janela de plotagem."""
        try:
            tmi = int(self.e_tmi.get() or 0)
            tmf = int(self.e_tmf.get() or self.ns * self.sr * 1000)
            self.tri = int(self.e_tri.get() or 0)
            self.trf = int(self.e_trf.get() or self.nt)
            self.spi = int(tmi / self.sr / 1000)
            self.spf = int(tmf / self.sr / 1000)

            self.sigdata = self.data[self.tri:self.trf, self.spi:self.spf]
            
            vm = np.percentile(self.sigdata, 95)
            self.cmin = -vm
            self.cmax = vm
            self.cmap = self.palet

            title = 'Working Window over Raw Data'
            self.sigfilt = ''
            self.cmin, self.cmax, self.palet = self.img_show(
                self.filename, title, self.sigdata, self.tri, self.trf,
                self.spi, self.spf, self.sr, self.cmin, self.cmax, self.palet, self.sigfilt
            )
            self.limits = True
            self.btn_freq_filters.config(state='normal')
            self.btn_ampl_filters.config(state='normal')
            self.btn_conv.config(state='normal')
            self.btn_deconv.config(state='normal')
            self.btn_interpretation.config(state='normal')

        except Exception as e:
            messagebox.showerror("Errorr", f"Failed to display image: {e}")

    def save_limits(self):
        """saved a visualização atual na lista de cenas."""
        if self.limits:  # area com limites definidos 
            scene = [self.sigdata, self.cmin, self.cmax, 'Working Window', self.sigfilt, self.palet]
            self.scenes.loc[len(self.scenes)] = scene
            self.txt_edit.insert(END, f'\nWorking Window saved:\n Start Trace: {self.tri}, End Trace: {self.trf}\n')
            self.txt_edit.insert(END, f' Start Time: {self.spi * self.sr * 1000:.0f} ms, End Time: {self.spf * self.sr * 1000:.0f} ms\n')
        else:  # area total selecionada
            scene = [self.sigdata, self.cmin, self.cmax, 'Working Window', self.sigfilt, self.palet]
            self.scenes.loc[len(self.scenes)] = scene
            self.txt_edit.insert(END, f'\nWorking Window saved:\n Start Trace: {self.tri}, End Trace: {self.trf}\n')
            self.txt_edit.insert(END, f' Start Time: {self.spi * self.sr * 1000:.0f} ms, End Time: {self.spf * self.sr * 1000:.0f} ms\n')
            self.limits == False
        self.work_window.destroy()

    def img_show(self, filename, title, image_data, tri, trf, spi, spf, sr, v_min, v_max, colors, filt):
        """Exibe a Image sísmica com opções de zoom e sliders."""
        fig, ax1 = plt.subplots(figsize=(15, 7))
        tmi = spi * sr * 1000
        tmf = spf * sr * 1000
        twt = np.linspace(tmi, tmf, image_data.shape[1])  
        ext = [tri, trf, twt[-1], twt[0]]
        
        ax1.set_xlabel('Trace number')
        ax1.set_ylabel('TWT [ms]')
        ax1.set_title(f'{filename} - {title}{filt}')
        
        ax1_smp = ax1.twinx()
        ax1_smp.set_ylabel('Samples')
        ax1_smp.set_ylim(spf, spi)
        ax1_smp.set_xlim(tri, trf)
        
        vm = np.percentile(image_data, 95)
        img = ax1.imshow(image_data.T, vmin=-vm, vmax=vm, cmap=colors, aspect='auto', extent=ext)

        ax_clr_min = plt.axes([0.96, 0.1, 0.01, 0.8])
        ax_clr_max = plt.axes([0.98, 0.1, 0.01, 0.8])
        s_clr_min = Slider(ax_clr_min, '', -vm, 0, orientation='vertical')
        s_clr_max = Slider(ax_clr_max, '', 0, vm, orientation='vertical')
        
        def update(val):
            img.set_clim([s_clr_min.val, s_clr_max.val])
            fig.canvas.draw_idle()

        s_clr_min.on_changed(update)
        s_clr_max.on_changed(update)
        
        plt.show()
        return s_clr_min.val, s_clr_max.val, colors  # return cmin, cmax, palette
    
    def select_data_combobox(self, window, px, py):
        """Cria um combobox para Select o tipo de dado para filtragem."""
        options = ['Raw Data', 'Band Pass', 'Ormsby','Spectral Whitening', 'AGC', 'STALTA', 'Conv Ricker', 'Conv Ormsby',
                    'Deconv Bottom', 'Deconv Predictive','Deconv Wavelet'] 
        cbox = ttk.Combobox(window, state="readonly", width=13)
        cbox['values'] = options
        cbox.place(x=px, y=py)
        cbox.bind('<<ComboboxSelected>>', lambda event: self._update_sigdata(cbox.get()))
        Label(window, text="Select Data").place(x=px+15, y=py-22)
        return cbox  # Retorna o combobox para referência
    
    def _update_sigdata(self, selection):
        """Atualiza a variável de Data com base na seleção do combobox."""
        try:
            if selection == 'Raw Data':
                self.sigdata = self.data[self.tri:self.trf, self.spi:self.spf]
                self.sigfilt = ' over Raw Data'
            elif selection == 'Band Pass':
                bp_scene = self.scenes[self.scenes['tit'].str.contains('Band Pass')]
                if not bp_scene.empty:
                    self.sigdata = bp_scene.iloc[-1]['img']
                    self.cmin = bp_scene.iloc[-1]['cmin']
                    self.cmax = bp_scene.iloc[-1]['cmax']   
                    self.sigfilt = ' over Band Pass'
                    self.cmap = self.palet
            elif selection == 'Ormsby':
                orm_scene = self.scenes[self.scenes['tit'].str.contains('Ormsby')]
                if not orm_scene.empty:
                    self.sigdata = orm_scene.iloc[-1]['img']
                    self.cmin = orm_scene.iloc[-1]['cmin']
                    self.cmax = orm_scene.iloc[-1]['cmax']
                    self.sigfilt = ' over Ormsby'
                    self.cmap = self.palet  
            elif selection == 'Spectral Whitening':
                wt_scene = self.scenes[self.scenes['tit'].str.contains('Spectral Whitening')]
                if not wt_scene.empty:
                    self.sigdata = wt_scene.iloc[-1]['img']
                    self.cmin = wt_scene.iloc[-1]['cmin']
                    self.cmax = wt_scene.iloc[-1]['cmax']   
                    self.sigfilt = ' over Spectral Whitening'
                    self.cmap = self.palet
            elif selection == 'AGC':
                agc_scene = self.scenes[self.scenes['tit'].str.contains('AGC')]
                if not agc_scene.empty:
                    self.sigdata = agc_scene.iloc[-1]['img']
                    self.cmin = agc_scene.iloc[-1]['cmin']
                    self.cmax = agc_scene.iloc[-1]['cmax']
                    self.sigfilt = ' over AGC'
                    self.cmap = self.palet
            elif selection == 'STALTA':
                stalta_scene = self.scenes[self.scenes['tit'].str.contains('STALTA')]
                if not stalta_scene.empty:
                    self.sigdata = stalta_scene.iloc[-1]['img']
                    self.cmin = stalta_scene.iloc[-1]['cmin']
                    self.cmax = stalta_scene.iloc[-1]['cmax']
                    self.sigfilt = ' over STALTA'
                    self.cmap = self.palet
            elif selection == 'Conv Ricker':
                conv_rik_scene = self.scenes[self.scenes['tit'].str.contains('Conv Ricker')]
                if not conv_rik_scene.empty:
                    self.sigdata = conv_rik_scene.iloc[-1]['img']
                    self.cmin = conv_rik_scene.iloc[-1]['cmin']
                    self.cmax = conv_rik_scene.iloc[-1]['cmax']
                    self.sigfilt = ' over Ricker Convolution'
                    self.cmap = self.palet
            elif selection == 'Conv Ormsby':
                conv_orm_scene = self.scenes[self.scenes['tit'].str.contains('Conv Ormsby')]
                if not conv_orm_scene.empty:
                    self.sigdata = conv_orm_scene.iloc[-1]['img']
                    self.cmin = conv_orm_scene.iloc[-1]['cmin']
                    self.cmax = conv_orm_scene.iloc[-1]['cmax']
                    self.sigfilt = ' over Ormsby Convolution'
                    self.cmap = self.palet
            elif selection == 'Deconv Bottom':
                deconv_bot_scene = self.scenes[self.scenes['tit'].str.contains('Deconv Bottom')]
                if not deconv_bot_scene.empty:
                    self.sigdata = deconv_bot_scene.iloc[-1]['img']
                    self.cmin = deconv_bot_scene.iloc[-1]['cmin']
                    self.cmax = deconv_bot_scene.iloc[-1]['cmax']
                    self.sigfilt = ' over Bottom Deconvolution '
                    self.cmap = self.palet
            elif selection == 'Deconv Predictive':
                deconv_pred_scene = self.scenes[self.scenes['tit'].str.contains('Deconv Predictive')]
                if not deconv_pred_scene.empty:
                    self.sigdata = deconv_pred_scene.iloc[-1]['img']
                    self.cmin = deconv_pred_scene.iloc[-1]['cmin']
                    self.cmax = deconv_pred_scene.iloc[-1]['cmax']
                    self.sigfilt = ' over Predict Deconvolution'
                    self.cmap = self.palet
            elif selection == 'Deconv Wavelet':
                deconv_wav_scene = self.scenes[self.scenes['tit'].str.contains('Deconv Wavelet')]
                if not deconv_wav_scene.empty:
                    self.sigdata = deconv_wav_scene.iloc[-1]['img']
                    self.cmin = deconv_wav_scene.iloc[-1]['cmin']
                    self.cmax = deconv_wav_scene.iloc[-1]['cmax']
                    self.sigfilt = ' over Wavelet Deconvolution'
                    self.cmap = self.palet
        except Exception:
            messagebox.showerror("Error", "Filter not defined or data not available.")

    def select_scene(self, window, px, py):
        """Cria um combobox para Select cenas saveds."""

        options = ['Raw Data', 'Band Pass', 'Ormsby', 'Spectral Whitening', 'AGC', 'STALTA', 'Conv Ricker', 'Conv Ormsby',
                    'Deconv Bottom', 'Deconv Predictive','Deconv Wavelet']
        cbox = ttk.Combobox(window, state="readonly", width=13)
        cbox['values'] = options
        cbox.place(x=px, y=py)
        cbox.bind('<<ComboboxSelected>>', lambda event: self._update_scene(cbox.get()))
        Label(window, text="Select Scene").place(x=px+15, y=py-22)
        return cbox  # Retorna o combobox para referência
    
    def _update_scene(self, selection):
        """Atualiza a Image exibida com base na seleção do combobox."""
        # try:
        if selection == 'Raw Data':
            self.sigdata = self.data[self.tri:self.trf, self.spi:self.spf]
            self.cmin = -np.percentile(self.sigdata, 95)
            self.cmax = np.percentile(self.sigdata, 95)
            self.sigfilt = ' over Raw Data'
            self.cmap = self.palet
        elif selection == 'Band Pass':
            bp_scene = self.scenes[self.scenes['tit'].str.contains('Band Pass')]
            if not bp_scene.empty:
                self.sigdata = bp_scene.iloc[-1]['img']
                self.cmin = bp_scene.iloc[-1]['cmin']
                self.cmax = bp_scene.iloc[-1]['cmax']
                self.sigfilt = ' over Band Pass'
                self.cmap = self.palet
        elif selection == 'Ormsby':
            orm_scene = self.scenes[self.scenes['tit'].str.contains('Ormsby')]
            if not orm_scene.empty:
                self.sigdata = orm_scene.iloc[-1]['img']
                self.cmin = orm_scene.iloc[-1]['cmin']
                self.cmax = orm_scene.iloc[-1]['cmax']
                self.sigfilt = ' over Ormsby filter'
                self.cmap = self.palet  
        elif selection == 'Spectral Whitening':
            wt_scene = self.scenes[self.scenes['tit'].str.contains('Spectral Whitening')]
            if not wt_scene.empty:
                self.sigdata = wt_scene.iloc[-1]['img']
                self.cmin = wt_scene.iloc[-1]['cmin']
                self.cmax = wt_scene.iloc[-1]['cmax']
                self.sigfilt = ' over Spectral Whitening'
                self.cmap = self.palet
        elif selection == 'AGC':
            agc_scene = self.scenes[self.scenes['tit'].str.contains('AGC')]
            if not agc_scene.empty:
                self.sigdata = agc_scene.iloc[-1]['img']
                self.cmin = agc_scene.iloc[-1]['cmin']
                self.cmax = agc_scene.iloc[-1]['cmax']
                self.sigfilt = ' over AGC'
                self.cmap = self.palet
        elif selection == 'STALTA':
            stalta_scene = self.scenes[self.scenes['tit'].str.contains('STALTA')]
            if not stalta_scene.empty:
                self.sigdata = stalta_scene.iloc[-1]['img']
                self.cmin = stalta_scene.iloc[-1]['cmin']
                self.cmax = stalta_scene.iloc[-1]['cmax']
                self.sigfilt = ' over STALTA'
                self.cmap = self.palet
        elif selection == 'Conv Ricker':
            conv_ricker_scene = self.scenes[self.scenes['tit'].str.contains('Ricker')]
            if not conv_ricker_scene.empty:
                self.sigdata = conv_ricker_scene.iloc[-1]['img']
                self.cmin = conv_ricker_scene.iloc[-1]['cmin']
                self.cmax = conv_ricker_scene.iloc[-1]['cmax']
                self.sigfilt = ' over Ricker Convolution'
                self.cmap = self.palet
        elif selection == 'Conv Ormsby':
            conv_ormsby_scene = self.scenes[self.scenes['tit'].str.contains('Conv Ormsby')]
            if not conv_ormsby_scene.empty:
                self.sigdata = conv_ormsby_scene.iloc[-1]['img']
                self.cmin = conv_ormsby_scene.iloc[-1]['cmin']
                self.cmax = conv_ormsby_scene.iloc[-1]['cmax']
                self.sigfilt = ' over Ormsby Convolution'
                self.cmap = self.palet
        elif selection == 'Deconv Bottom':
            deconv_bottom_scene = self.scenes[self.scenes['tit'].str.contains('Bottom')]
            if not deconv_bottom_scene.empty:
                self.sigdata = deconv_bottom_scene.iloc[-1]['img']
                self.cmin = deconv_bottom_scene.iloc[-1]['cmin']
                self.cmax = deconv_bottom_scene.iloc[-1]['cmax']
                self.sigfilt = ' over Bottom Pulse Deconvolution'
                self.cmap = self.palet
        elif selection == 'Deconv Predictive':
            deconv_predictive_scene = self.scenes[self.scenes['tit'].str.contains('Predictive')]
            if not deconv_predictive_scene.empty:
                self.sigdata = deconv_predictive_scene.iloc[-1]['img']
                self.cmin = deconv_predictive_scene.iloc[-1]['cmin']
                self.cmax = deconv_predictive_scene.iloc[-1]['cmax']
                self.sigfilt = ' over Predicitive Deconvolution'
                self.cmap = self.palet
        elif selection == 'Deconv Wavelet':
            deconv_wavelet_scene = self.scenes[self.scenes['tit'].str.contains('Wavelet Deconv')]
            if not deconv_wavelet_scene.empty:
                self.sigdata = deconv_wavelet_scene.iloc[-1]['img']
                self.cmin = deconv_wavelet_scene.iloc[-1]['cmin']
                self.cmax = deconv_wavelet_scene.iloc[-1]['cmax']
                self.sigfilt = ' over Wavelet Deconvolution'
                self.cmap = self.palet
        
    def _save_filter_result(self, filter_type):
        """salve o resultado de um Filter na lista de cenas."""

        if filter_type == 'Band Pass' and self.bp_data is not None:
            scene = [self.bp_data, self.cmin, self.cmax, self.bp_title, self.sigfilt, self.cmap]
            self.scenes.loc[len(self.scenes)] = scene
            self.band_pass_window.destroy()

        elif filter_type == 'Ormsby' and self.ormsby_data is not None:
            scene = [self.ormsby_data, self.cmin, self.cmax, self.ormsby_title, self.sigfilt, self.cmap]
            self.scenes.loc[len(self.scenes)] = scene
            self.ormsby_window.destroy()
            
        elif filter_type == 'Spectral Whitening' and self.wt_data is not None:
            scene = [self.wt_data, self.cmin, self.cmax, self.wt_title, self.sigfilt, self.cmap]
            self.scenes.loc[len(self.scenes)] = scene
            self.whitening_window.destroy()

        elif filter_type == 'AGC' and self.agc_data is not None:
            scene = [self.agc_data, self.cmin, self.cmax, self.agc_title, self.sigfilt, self.cmap]
            self.scenes.loc[len(self.scenes)] = scene
            self.agc_window.destroy()

        elif filter_type == 'STALTA' and self.stalta_data is not None:  
            scene = [self.stalta_data, self.cmin, self.cmax, self.stalta_title, self.sigfilt, self.cmap]
            self.scenes.loc[len(self.scenes)] = scene
            self.stalta_window.destroy()

        elif filter_type == 'Conv Ricker' and self.conv_ricker_data is not None:
            scene = [self.conv_ricker_data, self.cmin, self.cmax, self.conv_ricker_title, self.sigfilt, self.cmap]
            self.scenes.loc[len(self.scenes)] = scene
            self.ricker_window.destroy()

        elif filter_type == 'Conv Ormsby' and self.conv_ormsby_data is not None:
            scene = [self.conv_ormsby_data, self.cmin, self.cmax, self.conv_ormsby_title, self.sigfilt, self.cmap]
            self.scenes.loc[len(self.scenes)] = scene
            self.ormsby_window.destroy()

        elif filter_type == 'Deconv Bottom' and self.deconv_bot_data is not None:
            scene = [self.deconv_bot_data, self.cmin, self.cmax, self.deconv_bot_title, self.sigfilt, self.cmap]
            self.scenes.loc[len(self.scenes)] = scene
            self.deconv_bot_window.destroy()

        elif filter_type == 'Deconv Predictive' and self.deconv_pred_data is not None:
            scene = [self.deconv_pred_data, self.cmin, self.cmax, self.deconv_pred_title, self.sigfilt, self.cmap]
            self.scenes.loc[len(self.scenes)] = scene
            self.deconv_pred_window.destroy()

        elif filter_type == 'Deconv Wavelet' and self.deconv_wav_data is not None:
            scene = [self.deconv_wav_data, self.cmin, self.cmax, self.deconv_wav_title, self.sigfilt, self.cmap]
            self.scenes.loc[len(self.scenes)] = scene
            self.deconv_wav_window.destroy()

        else:
            messagebox.showerror("Error", f"No {filter_type} data to save.")
        
        # print(self.scenes)
            
    def open_freq_filters_window(self):
        """Abre a janela de Filters de Frequency."""
        freq_filters_window = Toplevel(self.root)
        freq_filters_window.title("FREQUENCIES FILTERS")
        w = 250
        h = 360
        ws = self.root.winfo_screenwidth()  
        hs = self.root.winfo_screenheight()  
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        freq_filters_window.geometry('%dx%d+%d+%d' % (w, h, x, y))
        freq_filters_window.resizable(width=False, height=False)
        freq_filters_window.attributes('-toolwindow', True)
       
        Button(freq_filters_window, text='Amplitude SPECTRUM', width=22, command=self.open_amplitude_spectrum_window,
               bg="#4CAF50", fg='white').place(x=40, y=50)
        Button(freq_filters_window, text='SPECTROGRAM', width=22, command=self.open_spectrogram_window,
               bg="#3F8F41", fg='white').place(x=40, y=100)
        Button(freq_filters_window, text='BAND-PASS FILTER', width=22, command=self.open_band_pass_window,
               bg="#295E2B", fg='white').place(x=40, y=150)
        Button(freq_filters_window, text='ORMSBY FILTER', width=22, command=self.open_ormsby_filter_window,
               bg="#1D421E", fg='white').place(x=40, y=200)
        Button(freq_filters_window, text='SPECTRAL WHITENING', width=22, command=self.open_spectral_whitening_window,
               bg="#1D421E", fg='white').place(x=40, y=250)

    def open_amplitude_spectrum_window(self):
        self.amplit_spectrum = Toplevel(self.root)
        self.amplit_spectrum.title("Amplitude SPECTRUM")
        w = 250
        h = 200
        ws = self.root.winfo_screenwidth()  
        hs = self.root.winfo_screenheight()  
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        self.amplit_spectrum.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.amplit_spectrum.resizable(width=False, height=False)
        self.amplit_spectrum.attributes('-toolwindow', True)

        self.select_data_combobox(self.amplit_spectrum, 70, 30)
        Button(self.amplit_spectrum, text='Apply', width=10, command=self.plot_amplitude_spectrum,
               bg="#2196F3", fg='white').place(x=85, y=80)

    def plot_amplitude_spectrum(self):
        try:
            plt.figure(figsize=(10, 6))
            stp = max(1, (self.trf-self.tri)//4)  # Evita divisão por zero
            for i in range(0, self.trf-self.tri, stp):
                if i < self.sigdata.shape[0]:  # Verifica limites
                    Xf_mag = np.abs(np.fft.fft(self.sigdata[i]))
                    freqs = np.fft.fftfreq(len(Xf_mag), d=self.sr)
                    plt.plot(abs(freqs), Xf_mag, label='Trace '+str(i))
            plt.title('Line '+str(self.filename)+ ' - Amplitude SPECTRUM '+self.sigfilt, fontsize=11)
            plt.xlabel('frequency [Hz]')
            plt.ylabel('|Amplitude|')
            plt.legend()
            plt.grid()
            plt.show(block=False)
            self.amplit_spectrum.destroy()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot spectrum: {e}")

    def open_spectrogram_window(self):
        """Abre a janela para o Spectrumgrama."""
        self.spectrogram_window = Toplevel(self.root)
        self.spectrogram_window.title("SPECTROGRAM")
        w = 250
        h = 200
        ws = self.root.winfo_screenwidth()  
        hs = self.root.winfo_screenheight()  
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        self.spectrogram_window.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.spectrogram_window.resizable(width=False, height=False)
        self.spectrogram_window.attributes('-toolwindow', True)

        self.select_data_combobox(self.spectrogram_window, 70, 30)
        Button(self.spectrogram_window, text='Apply', width=10, command=self.plot_spectrogram,
               bg="#2196F3", fg='white').place(x=85, y=80)

    def plot_spectrogram(self):
        try:
            plt.figure(figsize=(10, 6))
            sig1t = np.ravel(self.sigdata)
            tit = 'SPECTROGRAM (magnitude) '+self.sigfilt+ '  -  All Traces'
            Pxx, freqs, bins, im = plt.specgram(sig1t, NFFT=256, Fs=1/self.sr, noverlap=128, 
                                                mode='magnitude', scale='dB', cmap='turbo') 
            plt.title(tit, fontsize=11)
            plt.xlabel('Time (sec)')
            plt.ylabel('Frequency')
            plt.colorbar(orientation='vertical')
            plt.grid()
            plt.show(block=False)
            self.spectrogram_window.destroy()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot spectrogram: {e}")

    def open_band_pass_window(self):
        """Abre a janela para o Filter Band Pass."""
        self.band_pass_window = Toplevel(self.root)
        self.band_pass_window.title("BAND-PASS")
        w = 275
        h = 290
        ws = self.band_pass_window.winfo_screenwidth()
        hs = self.band_pass_window.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        self.band_pass_window.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.band_pass_window.resizable(width=False, height=False)
        self.band_pass_window.attributes('-toolwindow', True)
        
        Label(self.band_pass_window, text="Low Cut (Hz): ").place(x=50, y=70)
        self.e_LC = Entry(self.band_pass_window, width=10)
        self.e_LC.place(x=140, y=70)
        
        Label(self.band_pass_window, text="High Cut (Hz): ").place(x=50, y=100)
        self.e_HC = Entry(self.band_pass_window, width=10)
        self.e_HC.place(x=140, y=100)
        
        Button(self.band_pass_window, width=12, text='Apply', command=self.apply_band_pass,
               bg="#307FA7", fg='white').place(x=80, y=140)
        Button(self.band_pass_window, width=12, text='Save', command=lambda: self._save_filter_result('Band Pass'),
               bg="#43AB1E", fg='white').place(x=80, y=180)
        
        self.select_data_combobox(self.band_pass_window, 80, 30)

    def apply_band_pass(self):
        """Aplica o Filter Band Pass."""
        try:
            lowcut = float(self.e_LC.get())
            highcut = float(self.e_HC.get())
            
            low = lowcut / self.nyquist
            high = highcut / self.nyquist
            
            b, a = signal.butter(5, [low, high], btype='band')
            self.bp_data = signal.lfilter(b, a, self.sigdata)
            
            vm = np.percentile(self.bp_data, 95)
            self.cmin, self.cmax = -vm, vm
            self.bp_title = f'Band Pass Filter [{lowcut} - {highcut} Hz]'
            self.cmap = self.palet
            
            self.img_show(
                self.filename, self.bp_title, self.bp_data, self.tri, self.trf,
                self.spi, self.spf, self.sr, self.cmin, self.cmax, self.cmap, self.sigfilt
            )
            self.txt_edit.insert(END, f'\nBand Pass filter applied: Low: {lowcut} Hz, High: {highcut} Hz')
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid frequency limits.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply filter: {e}")

    def open_ormsby_filter_window(self):    
        """Abre a janela para o Filter Ormsby."""
        self.ormsby_window = Toplevel(self.root)
        self.ormsby_window.title("ORMSBY FILTER")
        w = 300
        h = 350
        ws = self.ormsby_window.winfo_screenwidth()
        hs = self.ormsby_window.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        self.ormsby_window.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.ormsby_window.resizable(width=False, height=False)
        self.ormsby_window.attributes('-toolwindow', True)
        
        Label(self.ormsby_window, text="f1 (Hz): ").place(x=50, y=70)
        self.e_f1 = Entry(self.ormsby_window, width=10)
        self.e_f1.place(x=150, y=70)
        
        Label(self.ormsby_window, text="f2 (Hz): ").place(x=50, y=100)
        self.e_f2 = Entry(self.ormsby_window, width=10)
        self.e_f2.place(x=150, y=100)
        
        Label(self.ormsby_window, text="f3 (Hz): ").place(x=50, y=130)
        self.e_f3 = Entry(self.ormsby_window, width=10)
        self.e_f3.place(x=150, y=130)
        
        Label(self.ormsby_window, text="f4 (Hz): ").place(x=50, y=160)
        self.e_f4 = Entry(self.ormsby_window, width=10)
        self.e_f4.place(x=150, y=160)
        
        Button(self.ormsby_window, width=12, text='Apply', command=self.apply_ormsby_filter,
               bg="#307FA7", fg='white').place(x=90, y=200)
        Button(self.ormsby_window, width=12, text='Save', command=lambda: self._save_filter_result('Ormsby'),
               bg="#43AB1E", fg='white').place(x=90, y=240)

        self.select_data_combobox(self.ormsby_window, 80, 30)

    def apply_ormsby_filter(self):
        """Aplica o Filter Ormsby."""
        try:
            f1 = float(self.e_f1.get())
            f2 = float(self.e_f2.get())
            f3 = float(self.e_f3.get())
            f4 = float(self.e_f4.get())
            
            def ormsby_filter(f, f1, f2, f3, f4):
                """Cria o filtro Ormsby."""
                H = np.zeros_like(f)
                for i in range(len(f)):
                    fi = abs(f[i])
                    if fi < f1:
                        H[i] = 0
                    elif f1 <= fi < f2:
                        H[i] = ((fi - f1)**2) / ((f2 - f1)**2)
                    elif f2 <= fi < f3:
                        H[i] = 1
                    elif f3 <= fi < f4:
                        H[i] = ((f4 - fi)**2) / ((f4 - f3)**2)
                    else:
                        H[i] = 0
                return H

            n_samples = self.sigdata.shape[1]
            freqs = np.fft.fftfreq(n_samples, d=self.sr)
            H = ormsby_filter(freqs, f1, f2, f3, f4)

            self.ormsby_data = np.zeros_like(self.sigdata)
            for i in range(self.sigdata.shape[0]):
                Xf = np.fft.fft(self.sigdata[i])
                Xf_filtered = Xf * H
                self.ormsby_data[i] = np.fft.ifft(Xf_filtered).real

            vm = np.percentile(self.ormsby_data, 95)
            self.cmin, self.cmax = -vm, vm
            self.ormsby_title = f'Ormsby Filter [{f1}, {f2}, {f3}, {f4} Hz]'
            self.cmap = self.palet
            
            self.img_show(
                self.filename, self.ormsby_title, self.ormsby_data, self.tri, self.trf,
                self.spi, self.spf, self.sr, self.cmin, self.cmax, self.cmap, self.sigfilt
            )
            self.txt_edit.insert(END, f'\nOrmsby filter applied: f1: {f1} Hz, f2: {f2} Hz, f3: {f3} Hz, f4: {f4} Hz')
        except ValueError:
            messagebox.showerror("Error", "Please enter valid frequency limits.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply Ormsby filter: {e}")

    def open_spectral_whitening_window(self):
        """Abre a janela para o Filter Spectral Whitening."""
        self.whitening_window = Toplevel(self.root)
        self.whitening_window.title("SPECTRAL WHITENING")
        w = 240
        h = 220
        ws = self.whitening_window.winfo_screenwidth()
        hs = self.whitening_window.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        self.whitening_window.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.whitening_window.resizable(width=False, height=False)
        self.whitening_window.attributes('-toolwindow', True)

        self.meth = 'zca' # Default method
        self.var = IntVar(value=1)

        Label(self.whitening_window, text="Using Method ZCA").place(x=70, y=70)
                
        Button(self.whitening_window, width=10, text="Apply", command=self.apply_spectral_whitening,
               bg="#307FA7", fg='white').place(x=75, y=100)
        Button(self.whitening_window, width=10, text='Save', command=lambda: self._save_filter_result('Spectral Whitening'),
               bg="#43AB1E", fg='white').place(x=75, y=140)

        self.select_data_combobox(self.whitening_window, 70, 30)

    def _set_method(self, method):
        """Define o método para o clareamento espectral."""
        self.meth = method

    def _whiten(self, X, method='zca'):
        """Função para clareamento espectral."""
        original_shape = X.shape
        X = X.reshape((-1, np.prod(X.shape[1:])))
        X_centered = X - np.mean(X, axis=0)
        Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
        W = None
        
        U, Lambda, _ = np.linalg.svd(Sigma)
        if method == 'zca':
            W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))
        elif method == 'pca':
            W = np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T)
        elif method == 'cholesky':
            W = np.linalg.cholesky(np.dot(U, np.dot(np.diag(1.0 / (Lambda + 1e-5)), U.T))).T
        else:
            raise ValueError('Whitening method not found.')

        result = np.dot(X_centered, W.T)
        return result.reshape(original_shape)

    def apply_spectral_whitening(self):
        """Aplica o clareamento espectral."""
        try:
            self.wt_data = self._whiten(self.sigdata, method=self.meth)
            vm = np.percentile(self.wt_data, 95)
            self.cmin, self.cmax = -vm, vm
            self.wt_title = f'Spectral Whitening - Method: {self.meth}'
            self.cmap = self.palet
            
            self.img_show(
                self.filename, self.wt_title, self.wt_data, self.tri, self.trf,
                self.spi, self.spf, self.sr, self.cmin, self.cmax, self.cmap, self.sigfilt
            )
            self.txt_edit.insert(END, f'\nSpectral Whitening applied: Method: {self.meth}')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply whitening: {e}")

    def open_amplitude_filters_window(self):
        """Abre a janela de Filters de Amplitude."""
        if self.sigdata is None:
            messagebox.showerror("Error", "Please load a SEG-Y file and define a working window first.")
            return

        amp_filters_window = Toplevel(self.root)
        amp_filters_window.title("Amplitude FILTERS")
        w = 200
        h = 190
        ws = self.root.winfo_screenwidth()
        hs = self.root.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        amp_filters_window.geometry('%dx%d+%d+%d' % (w, h, x, y))
        amp_filters_window.resizable(width=False, height=False)
        amp_filters_window.attributes('-toolwindow', True)

        Button(amp_filters_window, width=10, text='AGC', command=self.open_AGC_window,
               bg="#4CAF50", fg='white').place(x=60, y=30)
        Button(amp_filters_window, width=10, text='STA/LTA', command=self.open_STALTA_window,
               bg="#3A853D", fg='white').place(x=60, y=80)

    def open_AGC_window(self):
        """Abre a janela para o Filter AGC."""
        self.agc_window = Toplevel(self.root)
        self.agc_window.title('AGC')
        w = 250
        h = 280
        ws = self.root.winfo_screenwidth()
        hs = self.root.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        self.agc_window.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.agc_window.resizable(width=False, height=False)
        self.agc_window.attributes('-toolwindow', True)

        Label(self.agc_window, text="Window size\n(samples): ").place(x=55, y=70)
        self.e_agc_window = Entry(self.agc_window, width=7)  # Nome único
        self.e_agc_window.place(x=130, y=75)

        Button(self.agc_window, width=12, text='Apply', command=self.apply_agc,
               bg="#307FA7", fg='white').place(x=75, y=130)
        Button(self.agc_window, width=12, text='Save', command=lambda: self._save_filter_result('AGC'),
               bg="#43AB1E", fg='white').place(x=75, y=180)

        self.select_data_combobox(self.agc_window, 70, 30)

    def apply_agc(self):
        try:
            agc_data = np.ravel(self.sigdata)
            win_size = int(self.e_agc_window.get())

            # Calculates the square of the absolute value of the signal
            envelope = np.abs(agc_data) ** 2
            # Calculates moving average with sliding window
            smoothed = np.convolve(envelope, np.ones(win_size) / win_size, mode='same')
            # Calculates the square root of the smoothed moving average
            smoothed_sqrt = np.sqrt(smoothed)
            # Avoid division by zero
            smoothed_sqrt[smoothed_sqrt < 1e-10] = 1e-10
            # Normalizes the original signal by the smoothed signal
            agc_signal = agc_data / smoothed_sqrt
            self.agc_data = np.reshape(agc_signal, self.sigdata.shape)

            vm = np.percentile(self.agc_data, 95)
            self.cmin, self.cmax = -vm, vm
            self.agc_title = f'Automatic Gain Control (AGC) - Window: {win_size} samples'
            self.cmap = self.palet

            self.img_show(
                self.filename, self.agc_title, self.agc_data, self.tri, self.trf,
                self.spi, self.spf, self.sr, self.cmin, self.cmax, self.cmap, self.sigfilt
            )
            self.txt_edit.insert(END, f'\nAutomatic Gain Control (AGC) - Window= {win_size} samples')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply AGC: {e}")

    def open_STALTA_window(self):
        """Abre a janela para o Filter Stalta."""
        self.stalta_window = Toplevel(self.root)
        self.stalta_window.title('STA/LTA filter')
        w = 280
        h = 310
        ws = self.stalta_window.winfo_screenwidth()
        hs = self.stalta_window.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        self.stalta_window.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.stalta_window.resizable(width=False, height=False)
        self.stalta_window.attributes('-toolwindow', True)

        if self.sigdata is None:
            raise ValueError("None data loaded or selected.")
        
        Label(self.stalta_window, text="Short-Time Average\n  window samples= ").place(x=50, y=70)

        self.e_sta = Entry(self.stalta_window, width=5)  # Nome único
        self.e_sta.place(x=170, y=80)
        self.e_sta.insert(0, '5')

        Label(self.stalta_window, text="Long-Time Average\n  window samples= ").place(x=50, y=120)
        self.e_lta = Entry(self.stalta_window, width=5)  # Nome único
        self.e_lta.place(x=170, y=130)
        self.e_lta.insert(0, '20')
        
        Button(self.stalta_window, width=12, text='Apply', command=self.apply_stalta,
               bg="#307FA7", fg='white').place(x=90, y=170)
        Button(self.stalta_window, width=12, text='Save', command=lambda: self._save_filter_result('STALTA'),
               bg="#43AB1E", fg='white').place(x=90, y=210)

        self.select_data_combobox(self.stalta_window, 70, 30)

    @staticmethod
    def classic_sta_lta_py(a, nsta, nlta):
        """Implementação do algoritmo STA/LTA."""
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

    def apply_stalta(self):
        try:
            sta = int(self.e_sta.get())
            lta = int(self.e_lta.get())

            stalta_data = np.ravel(self.sigdata)
            stalta = self.classic_sta_lta_py(stalta_data, sta, lta)
            self.stalta_data = np.reshape(stalta, self.sigdata.shape)

            vm = np.percentile(self.stalta_data, 95)
            self.cmin, self.cmax = -vm, vm
            self.stalta_title = f'STALTA Filter - STA={sta} / LTA={lta}'
            self.cmap = self.palet

            self.img_show(
                self.filename, self.stalta_title, self.stalta_data, self.tri, self.trf,
                self.spi, self.spf, self.sr, self.cmin, self.cmax, self.cmap, self.sigfilt
            )
            self.txt_edit.insert(END, f'\nShort-Time Average / Long-Time Average (STA/LTA) - STA: {sta}, LTA: {lta}')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply STA/LTA: {e}")

# ================= CONVOLUTION ====================================================

    def open_conv_window(self):
        """Abre a janela de Convolution."""
        self.conv_window = Toplevel(self.root)
        self.conv_window.title('CONVOLUTION')
        w = 280
        h = 150
        ws = self.conv_window.winfo_screenwidth()
        hs = self.conv_window.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        self.conv_window.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.conv_window.resizable(width=False, height=False)
        self.conv_window.attributes('-toolwindow', True)

        Button(self.conv_window, width=25, text='with Ricker Wavelet', command=self.open_ricker_wavelet_window,
               bg="#E58A02", fg='white').place(x=50, y=30)
        Button(self.conv_window, width=25, text='with Ormsby Wavelet', command=self.open_ormsby_wavelet_window,
               bg="#9D5F02", fg='white').place(x=50, y=70)

        # self.select_data_combobox(self.conv_window, 100, 30)

    def open_ricker_wavelet_window(self):
        """Abre a janela para a Wavelet Ricker."""
        self.ricker_window = Toplevel(self.root)
        self.ricker_window.title('RICKER WAVELET CONVOLUTION')
        w = 250
        h = 230
        ws = self.ricker_window.winfo_screenwidth()
        hs = self.ricker_window.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        self.ricker_window.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.ricker_window.resizable(width=False, height=False)
        self.ricker_window.attributes('-toolwindow', True)

        Label(self.ricker_window, text="Pulse Width: ").place(x=50, y=70)
        self.e_PW = Entry(self.ricker_window, width=7)
        self.e_PW.place(x=140, y=70)

        Button(self.ricker_window, width=12, text='Apply', command=self.apply_ricker_wavelet,
               bg="#307FA7", fg='white').place(x=75, y=120)
        Button(self.ricker_window, width=12, text='Save', command=lambda: self._save_filter_result('Conv Ricker'),
               bg="#43AB1E", fg='white').place(x=75, y=160)

        self.select_data_combobox(self.ricker_window, 70, 30)   

    def apply_ricker_wavelet(self):

        self.pw = float(self.e_PW.get())   # Pulse width in seconds    
        if self.pw is None:
            return  
        
        try:
            samples = self.fs * self.pw
            t = np.arange(int(samples)) / self.fs
            siz = t.size  # numero de amostras de pulse width
            rick = signal.ricker(siz, 4.0)

            self.ricker_conv = []
            for tr in range(self.trf - self.tri):
                conv = np.convolve(self.sigdata[tr], rick, mode='same')  
                self.ricker_conv = np.append(self.ricker_conv, conv)

            n_samples = int(len(self.ricker_conv)/(self.trf - self.tri))
            self.conv_ricker_data = np.reshape(self.ricker_conv, (self.trf - self.tri, n_samples) )

            vm = np.percentile(self.conv_ricker_data, 95)
            self.cmin, self.cmax = -vm, vm
            # self.cmap = self.palet
            self.conv_ricker_title = f'Convol with Ricker Wavelet - Pulse Width: {self.pw:.3f} s'
            self.cmap = self.palet

            self.img_show(
                self.filename, self.conv_ricker_title, self.conv_ricker_data, self.tri, self.trf,
                self.spi, self.spf, self.sr, self.cmin, self.cmax, self.cmap, self.sigfilt
            )
            self.txt_edit.insert(END, f'\nConvol with Ricker Wavelet - Pulse Width: {self.pw:.3f} s')

        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply Ricker wavelet: {e}")

    def open_ormsby_wavelet_window(self):
            """Abre a janela para a Wavelet Ormsby."""
            self.ormsby_window = Toplevel(self.root)
            self.ormsby_window.title('ORMSBY WAVELET CONVOLUTION')
            w = 260
            h = 360
            ws = self.ormsby_window.winfo_screenwidth()
            hs = self.ormsby_window.winfo_screenheight()
            x = (ws/2) - (w/2)
            y = (hs/2) - (h/2)
            self.ormsby_window.geometry('%dx%d+%d+%d' % (w, h, x, y))
            self.ormsby_window.resizable(width=False, height=False)
            self.ormsby_window.attributes('-toolwindow', True)

            # Parâmetros do Ormsby
            Label(self.ormsby_window, text="Ormsby Parameters", 
                font=('Arial', 10, 'bold')).place(x=55, y=70)

            Label(self.ormsby_window, text="f1 (Hz):").place(x=70, y=110)
            self.e_ormsby_f1 = Entry(self.ormsby_window, width=6)
            self.e_ormsby_f1.insert(0, "200")
            self.e_ormsby_f1.place(x=120, y=110)

            Label(self.ormsby_window, text="f2 (Hz):").place(x=70, y=140)
            self.e_ormsby_f2 = Entry(self.ormsby_window, width=6)
            self.e_ormsby_f2.insert(0, "1000")
            self.e_ormsby_f2.place(x=120, y=140)

            Label(self.ormsby_window, text="f3 (Hz):").place(x=70, y=170)
            self.e_ormsby_f3 = Entry(self.ormsby_window, width=6)
            self.e_ormsby_f3.insert(0, "3000")
            self.e_ormsby_f3.place(x=120, y=170)

            Label(self.ormsby_window, text="f4 (Hz):").place(x=70, y=200)
            self.e_ormsby_f4 = Entry(self.ormsby_window, width=6)
            self.e_ormsby_f4.insert(0, "4000")
            self.e_ormsby_f4.place(x=120, y=200)

            self.f1_ormsby = float(self.e_ormsby_f1.get())
            self.f2_ormsby = float(self.e_ormsby_f2.get())
            self.f3_ormsby = float(self.e_ormsby_f3.get())
            self.f4_ormsby = float(self.e_ormsby_f4.get())

            # Validação
            if not (self.f1_ormsby < self.f2_ormsby < self.f3_ormsby < self.f4_ormsby):
                messagebox.showerror("Errorr", "Ormsby frequencies must be: f1 < f2 < f3 < f4")
                return

            if None in (self.f1_ormsby, self.f2_ormsby, self.f3_ormsby, self.f4_ormsby):
                return  # Usuário cancelou

            Button(self.ormsby_window, width=12, text='Apply', command=self.apply_ormsby_wavelet,
                   bg="#307FA7", fg='white').place(x=75, y=240)
            Button(self.ormsby_window, width=12, text='Save', command=lambda: self._save_filter_result('Conv Ormsby'),
                   bg="#43AB1E", fg='white').place(x=75, y=280)

            self.select_data_combobox(self.ormsby_window, 70, 30)   

    def apply_ormsby_wavelet(self):

        try:

            t_max = self.ns * self.sr  # Tempo máximo em segundos
            dt = self.sr  # Intervalo de amostragem
            t = np.arange(-t_max/2, t_max/2, dt)

            def ormsby_wavelet(t, f1_ormsby, f2_ormsby, f3_ormsby, f4_ormsby):
                pi = np.pi
                term1 = (pi * f4_ormsby) ** 2 * np.sinc(f4_ormsby * t) - (pi * f3_ormsby) ** 2 * np.sinc(f3_ormsby * t)
                term2 = (pi * f2_ormsby) ** 2 * np.sinc(f2_ormsby * t) - (pi * f1_ormsby) ** 2 * np.sinc(f1_ormsby * t)
                return (term1 - term2) / (f4_ormsby - f3_ormsby - f2_ormsby + f1_ormsby)

            ormsby_wavelet_data = ormsby_wavelet(t, self.f1_ormsby, self.f2_ormsby, self.f3_ormsby, self.f4_ormsby)

            conv_data = np.array([np.convolve(trace, ormsby_wavelet_data, mode='same') for trace in self.sigdata])
            self.conv_ormsby_data = conv_data

            vm = np.percentile(self.conv_ormsby_data, 95)
            self.cmin, self.cmax = -vm, vm
            self.conv_ormsby_title = f'Convolution with Ormsby Wavelet - {self.f1_ormsby}-{self.f2_ormsby}-{self.f3_ormsby}-{self.f4_ormsby} Hz'
            self.cmap = self.palet

            self.img_show(
                self.filename, self.conv_ormsby_title, self.conv_ormsby_data, self.tri, self.trf,
                self.spi, self.spf, self.sr, self.cmin, self.cmax, self.cmap, self.sigfilt
            )

            self.txt_edit.insert(END, f'\nConvolution with Ormsby Wavelet - F1={self.f1_ormsby} F2={self.f2_ormsby} F3={self.f3_ormsby} F4={self.f4_ormsby} Hz')

        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply Ormsby wavelet: {e}")   

# ================ DECONVOLUTION ==================================================================

    def open_deconv_window(self):
        """Abre a janela de deConvolution."""
        self.deconv_window = Toplevel(self.root)
        self.deconv_window.title('DECONVOLUTION')
        w = 340
        h = 260
        ws = self.deconv_window.winfo_screenwidth()
        hs = self.deconv_window.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        self.deconv_window.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.deconv_window.resizable(width=False, height=False)
        self.deconv_window.attributes('-toolwindow', True)

        Button(self.deconv_window, width=25, text='Bottom Pulse Deconvolution', command=self.apply_bottom_pulse_deconvolution,
               bg="#B48CD2", fg='white').place(x=65, y=80)
        Button(self.deconv_window, width=25, text='Predictive Deconvolution', command=self.apply_predictive_deconvolution,
               bg="#9071A8", fg='white').place(x=65, y=130)
        Button(self.deconv_window, width=25, text='Wavelet Deconvolution', command=self.apply_wavelet_deconvolution,
               bg="#6A537B", fg='white').place(x=65, y=180)

        self.select_data_combobox(self.deconv_window, 100, 30)

# ============== PREDICTIVE DECONVOLUTION ===========================================

    def apply_predictive_deconvolution(self):
        """Placeholder para deConvolution preditiva."""
        self.deconv_pred_window = Toplevel(self.root)
        self.deconv_pred_window.title('PREDICTIVE DECONVOLUTION')
        w = 220
        h = 220
        ws = self.deconv_pred_window.winfo_screenwidth()
        hs = self.deconv_pred_window.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        self.deconv_pred_window.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.deconv_pred_window.resizable(width=False, height=False)
        self.deconv_pred_window.attributes('-toolwindow', True)

        Label(self.deconv_pred_window, text="Filter Order: ").place(x=20, y=30)
        # 2. Definir ordem do Filter preditivo (ajuste conforme necessidade)
        self.filter_order = Entry(self.deconv_pred_window, width=5)
        self.filter_order.insert(0, '10')
        self.filter_order.place(x=100, y=30)

        Button(self.deconv_pred_window, width=12, text='Apply', command=self.predictive_deconv,
               bg="#307FA7", fg='white').place(x=65, y=70)
        Button(self.deconv_pred_window, width=12, text='Save', command=lambda: self._save_filter_result('Deconv Predictive'),
               bg="#43AB1E", fg='white').place(x=65, y=110)

    def predictive_deconv(self):

        ordem = int(self.filter_order.get())
        if ordem <= 0 or ordem >= self.sigdata.shape[1]:
            messagebox.showerror("Error", "Invalid filter order.")
            return
        
        # Função de deConvolution preditiva (copiada do exemplo anterior)
        def pred_deconv(signal, ordem):
        
            N = len(signal)
            X = toeplitz(signal[:-ordem], signal[ordem-1::-1])
            y = signal[ordem:]
            
            coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            
            predicted = lfilter(coef, [1.0], signal)
            residual = signal - predicted 
            return residual, coef


        # 3. Apply a deConvolution preditiva
        self.deconv_pred_data = []

        for i in range(self.nt):
            deconv, _ = pred_deconv(self.sigdata[i], ordem)
            self.deconv_pred_data = np.append(self.deconv_pred_data, deconv)
        
        comprimento_padrao = max(len(traco) for traco in self.sigdata)
    
        matriz = []
        for traco in self.sigdata:
            if len(traco) < comprimento_padrao:
                # preenche zeros até comprimento_padrao
                tr_padded = np.pad(traco, (0, comprimento_padrao - len(traco)), 'constant')
            else:
                # trunca se maior
                tr_padded = traco[:comprimento_padrao]
            matriz.append(tr_padded)
        
        # Reshape
        self.deconv_pred_data =  np.array(matriz)
        
        vm = np.percentile(self.deconv_pred_data, 95)
        self.cmin, self.cmax = -vm, vm
        self.deconv_pred_title = 'Predictive Deconvolution - Order={ordem}'
        self.cmap = self.palet
        
        self.img_show(
            self.filename, self.deconv_pred_title, self.deconv_pred_data, self.tri, self.trf,
            self.spi, self.spf, self.sr, self.cmin, self.cmax, self.cmap, self.sigfilt)
        
        self.txt_edit.insert(END, f'\nPredictive Deconvolution - Order={ordem}')


# ============= Wavelet CONVOLUTION==========================================================

    def apply_wavelet_deconvolution(self):
        """Placeholder para deConvolution Wavelet."""
        self.deconv_wav_window = Toplevel(self.root)
        self.deconv_wav_window.title('WAVELET DECONVOLUTION')
        w = 220
        h = 280
        ws = self.deconv_wav_window.winfo_screenwidth()
        hs = self.deconv_wav_window.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        self.deconv_wav_window.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.deconv_wav_window.resizable(width=False, height=False)
        self.deconv_wav_window.attributes('-toolwindow', True)

        Label(self.deconv_wav_window, text="Chirp Pulse Width: ").place(x=37, y=70)
        self.e_PW = Entry(self.deconv_wav_window, width=7)
        self.e_PW.place(x=140, y=70)

        Label(self.deconv_wav_window, text="Chirp Freq Init: ").place(x=50, y=100)
        self.e_f0 = Entry(self.deconv_wav_window, width=7)
        self.e_f0.place(x=140, y=100)

        Label(self.deconv_wav_window, text="Chirp Freq End: ").place(x=50, y=130)
        self.e_f1 = Entry(self.deconv_wav_window, width=7)
        self.e_f1.place(x=140, y=130)

        Button(self.deconv_wav_window, width=12, text='Apply', command=self.apply_wav_deconv,
               bg="#307FA7", fg='white').place(x=65, y=170)
        Button(self.deconv_wav_window, width=12, text='Save', command=lambda: self._save_filter_result('Deconv Wavelet'),
               bg="#43AB1E", fg='white').place(x=65, y=210)

    def apply_wav_deconv(self):
        """Aplica a deConvolution Wavelet."""

        self.pw = float(self.e_PW.get())   # Pulse width in seconds
        t = np.linspace(0, self.pw, int(self.pw*self.fs))

        self.txt_edit.insert(END, f'\nWavelet Deconvolution - Pulse Width: {self.pw:.3f} s')

        f0 = float(self.e_f0.get())   # Frequency start
        f1 = float(self.e_f1.get())   # Frequency end

        chirp_signal = chirp(t, f0=f0, f1=f1, t1=self.pw, method='linear')

        # DeConvolution por correlação cruzada (match filtering)
        # correlaciona o sinal sísmico com o Pulse chirp

        deconv_wav = []
        for i in range(self.trf - self.tri):
            dec = fftconvolve(self.sigdata[i], chirp_signal[::-1], mode='same')
            deconv_wav = np.append(deconv_wav, dec)
        
        self.deconv_wav_data = np.reshape(deconv_wav, self.sigdata.shape)
        vm = np.percentile(self.deconv_wav_data, 95)
        self.cmin, self.cmax = -vm, vm
        self.deconv_wav_title = f'Wavelet Deconvolution Params: Period={self.pw}s, F0={int(f0)},  F1={int(f1)} '
        self.cmap = self.palet

        self.img_show(self.filename, self.deconv_wav_title, self.deconv_wav_data, self.tri, self.trf,
                        self.spi, self.spf, self.sr, self.cmin, self.cmax, self.cmap, self.sigfilt)

        self.txt_edit.insert(END, f'\nWavelet Deconvolution - Pulse Width: {self.pw:.3f} s, F0={int(f0)},  F1={int(f1)} ')

# ============= BOTTOM DECONVOLUTION ======================================================= 

    def apply_bottom_pulse_deconvolution(self):
        """Abre janela para deConvolution por Pulse de fundo."""
        self.deconv_bot_window = Toplevel(self.root)
        self.deconv_bot_window.title('BOTTOM PULSE DECONVOLUTION')
        w = 320
        h = 250
        ws = self.deconv_bot_window.winfo_screenwidth()
        hs = self.deconv_bot_window.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        self.deconv_bot_window.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.deconv_bot_window.resizable(width=False, height=False)
        self.deconv_bot_window.attributes('-toolwindow', True)

        # self.select_data_combobox(self.deconv_bot_window, 100, 30)

        Button(self.deconv_bot_window, width=25,text='Select Seabottom Pulse', command=self.select_window_bottom_pulse, 
               bg="#307FA7", fg='white').place(x=60, y=40)
        Button(self.deconv_bot_window, width=15, text='Show Pulse', command=self.extract_pulse,
               bg="#307FA7", fg='white').place(x=90, y=80)
        Button(self.deconv_bot_window, width=20, text='Show Deconvol', command=self.deconv_bottom_pulse,
               bg="#307FA7", fg='white').place(x=70, y=120)
        Button(self.deconv_bot_window, width=12, text='Save', command=lambda: self._save_filter_result('Deconv Bottom'),
               bg="#43AB1E", fg='white').place(x=95, y=160)

    def select_window_bottom_pulse(self):
        """Seleciona janela para capturar Pulse do fundo."""
        fig, ax = plt.subplots(figsize=(8, 6))
        vm = np.percentile(self.sigdata, 95)
        ax.set_title('with mouse, select a window to capture the bottom signal')
        ax.imshow(self.sigdata.T, vmin=-vm, vmax=vm, cmap='binary', aspect='auto')


        def line_select_callback(eclick, erelease):
            self.x1, self.y1 = eclick.xdata, eclick.ydata
            self.x2, self.y2 = erelease.xdata, erelease.ydata
            rect = plt.Rectangle((min(self.x1, self.x2), min(self.y1, self.y2)), 
                               np.abs(self.x1 - self.x2), np.abs(self.y1 - self.y2), 
                               fill=False, edgecolor='red')
            ax.add_patch(rect)
            fig.canvas.draw_idle()

        rs = RectangleSelector(ax, line_select_callback, useblit=True, button=[1],
                            minspanx=3, minspany=3, spancoords='pixels', interactive=True)
        plt.show()

    def moving_average(self, x, w):
        """Calcula média móvel."""
        return np.convolve(x, np.ones(w), 'same') / w

    def extract_pulse(self):
        """Extrai Pulse do fundo marinho."""
        try:
            t1, t2 = int(min(self.x1, self.x2)), int(max(self.x1, self.x2))
            s1, s2 = int(min(self.y1, self.y2)), int(max(self.y1, self.y2))
        except AttributeError:
            messagebox.showerror("Error", "Please select a region first.")
            return

        bot_win = self.sigdata[t1:t2, s1:s2]
        if bot_win.size == 0:
            messagebox.showerror("Error", "Selected region is empty.")
            return
        
        plt.figure(figsize=(10, 5))
        plt.suptitle('Bottom Pulse Extraction - Check for good Klauder wavelet (Autocorrelation)', fontsize=12)
        plt.subplot(1, 3, 1)
        plt.title('Bottom window')
        plt.imshow(bot_win.T, vmin=-np.percentile(bot_win, 95), 
                  vmax=np.percentile(bot_win, 95), cmap='gray')

        # plt.show()
        
        def extract_smax(trace, sta, lta):
            I = self.classic_sta_lta_py(trace, sta, lta)
            if np.max(I) > 0:
                return np.argmax(I) + s1
            return -1

        sta = 5
        lta = 20
        Smax = [extract_smax(bot_win[tr], sta, lta) for tr in range(t2 - t1)]
        
        Smax = np.array([s for s in Smax if s != -1])
        if Smax.size == 0:
            messagebox.showerror("Error", "No bottom pulses found in the selected region.")
            return

        q75, q25 = np.percentile(Smax, [75, 25])
        iqr = q75 - q25
        lowpass = q25 - (iqr * 1.5)
        highpass = q75 + (iqr * 1.5)

        Smax_filtered = np.array([s if lowpass <= s <= highpass else np.nan for s in Smax])
        
        non_nan_indices = np.where(~np.isnan(Smax_filtered))[0]
        if non_nan_indices.size > 1:
            f = interpolate.interp1d(non_nan_indices, Smax_filtered[non_nan_indices], 
                                   kind='linear', fill_value='extrapolate')
            Smax_filtered[np.isnan(Smax_filtered)] = f(np.where(np.isnan(Smax_filtered))[0])
        else:
            if non_nan_indices.size > 0:
                Smax_filtered[np.isnan(Smax_filtered)] = np.mean(Smax_filtered[non_nan_indices])

        sma = self.moving_average(Smax_filtered, 5) 

        self.soma = np.zeros(s2 - s1)
        for t in range(t1, t2): 
            trace = self.sigdata[t, s1:s2]
            shift_amount = int(round(sma[t - t1] - Smax_filtered[t - t1]))
            shifted_trace = shift(trace, shift_amount, cval=0)
            self.soma += shifted_trace
        
        plt.subplot(1, 3, 2)
        x_axis = np.arange(s2 - s1)
        for t in range(t1, t2):
            plt.plot(self.sigdata[t, s1:s2], -x_axis, color='gray', alpha=0.5)
        plt.plot(self.soma, -x_axis, '-', color='r', linewidth=3)
        plt.axvline(x=0.0, color='k', linestyle='-')
        plt.title('Sum of Traces')
        # plt.show()

        plt.subplot(1, 3, 3)
        ap = self.autocorrelation(self.soma)
        # plt.figure(figsize=(4, 4))
        plt.plot(ap)
        plt.title('Autocorrelation')
        plt.show()

    def autocorrelation(self, x):
        """Calcula autocorrelação."""
        result = np.correlate(x, x, mode='full')
        return result[result.size // 2:]

    def deconv_bottom_pulse(self):
        """Aplica deconvolution usando o pulso do fundo."""
        if self.soma is None:
            messagebox.showerror("Error", "Please extract the pulse first by clicking 'Show Pulse'.")
            return

        autoc = self.autocorrelation(self.soma)
        
        if autoc.size == 0 or np.all(autoc == 0):
            messagebox.showerror("Error", "Autocorrelation of the pulse is zero.")
            return

        try:
            self.deconv_bot_data = []
            for i in range(self.trf - self.tri):
                trace = self.sigdata[i]
                dec, _ = signal.deconvolve(trace, autoc)
                self.deconv_bot_data.append(dec)

            # Convert to numpy array with consistent shape
            max_len = max(len(t) for t in self.deconv_bot_data) if self.deconv_bot_data else 0
            self.deconv_bot_data = np.array([np.pad(t, (0, max_len - len(t))) 
                                    if len(t) < max_len else t[:max_len] 
                                    for t in self.deconv_bot_data])

            vm = np.percentile(self.deconv_bot_data, 95)
            self.cmin, self.cmax = -vm, vm
            self.deconv_bot_title = f'Bottom Signal Deconvolution'
            self.cmap = self.palet

            self.img_show(self.filename, self.deconv_bot_title, self.deconv_bot_data, self.tri, self.trf,
                         self.spi, self.spf, self.sr, self.cmin, self.cmax, self.cmap, self.sigfilt)

            self.txt_edit.insert(END, f'\nDeconvolution using Bottom Signal applied.')
            self.txt_edit.see(END)

        except Exception as e:
            messagebox.showerror("Error", f"Deconvolution failed: {e}")

# ========== INTERPRETATION =========================================================

    def open_interpretation_window(self):
        self.win = Toplevel(self.root)
        self.win.title("Seismic Interpretation")
        self.win.geometry("1500x700")

        self.reflectors = {}
        self.current_reflector = StringVar(value="A")
        self.reflector_colors = {"A": "red", "B": "blue", "C": "green", "D": "orange", "E": "purple"}
        self.interpret_lines = {}

        # ===== TOPO: Seletor de Reflector =====
        top_frame = Frame(self.win)
        top_frame.pack(fill=X, padx=10, pady=5)

        Label(top_frame, text="Reflector:").pack(side=LEFT, padx=5)
        cmb_refletor = ttk.Combobox(top_frame, values=list(self.reflector_colors.keys()), 
                                    state="readonly", textvariable=self.current_reflector)
        cmb_refletor.pack(side=LEFT, padx=5)

        Label(top_frame, text="Speed of sound (m/s):").pack(side=LEFT, padx=10)
        self.entry_velocity = Entry(top_frame, width=8)
        self.entry_velocity.insert(0, "1500")
        self.entry_velocity.pack(side=LEFT)

        # ===== Image =====
        Label(top_frame, text="Choose the filter to interpret:").pack(side=LEFT, padx=15)
        self.cmb_image = ttk.Combobox(top_frame, state="readonly", width=40)
        
        # Preencher combobox com as opções disponíveis
        options = ['Raw Data', 'Band Pass', 'Ormsby','Spectral Whitening', 'AGC', 'STALTA', 'Conv Ricker', 'Conv Ormsby',
                    'Deconv Bottom', 'Deconv Predictive','Deconv Wavelet']
        self.cmb_image['values'] = options
        if options:
            self.cmb_image.set(options[0])
        self.cmb_image.pack(pady=5)

        # ===== MODO DE Interpretation =====
        self.modo_interpretacao = StringVar(value="desenho")

        # ===== BARRA INFERIOR =====
        btn_frame = Frame(self.win)
        btn_frame.pack(fill=X, pady=5)

        btn_show = Button(btn_frame, text="Show Image", command=self.show_image_interp)
        btn_show.pack(side=LEFT, padx=10)

        Label(btn_frame, text="Modo:").pack(side=LEFT)
        Radiobutton(btn_frame, text="Draw", variable=self.modo_interpretacao, 
                    value="desenho").pack(side=LEFT)
        Radiobutton(btn_frame, text="Edit", variable=self.modo_interpretacao, 
                    value="edicao").pack(side=LEFT)

        btn_close = Button(btn_frame, text="Close", command=self.win.destroy)
        btn_close.pack(side=RIGHT)
        
        btn_export = Button(btn_frame, text="Export CSV", command=self.export_reflectors_csv)
        btn_export.pack(side=RIGHT, padx=10)

        btn_export_dxf = Button(btn_frame, text="Export DXF", command=self.export_reflectors_dxf)
        btn_export_dxf.pack(side=RIGHT, padx=10)

        # btn_close = Button(btn_frame, text="Close", command=self.win.destroy)
        # btn_close.pack(side=RIGHT)

        # ===== CANVAS =====
        self.plot_frame = Frame(self.win)
        self.plot_frame.pack(fill=BOTH, expand=True)

    def show_image_interp(self):
        """Mostra a Image selecionada para Interpretation"""
        # Atualiza os Data baseado na seleção
        selection = self.cmb_image.get()
        self._update_scene(selection)        
        # Plota a Image
        self.plot_interpret_image()
        
        # Bind para atualizar quando mudar a seleção
        self.cmb_image.bind("<<ComboboxSelected>>", self.on_image_change)

    def on_image_change(self, event):
        """Callback quando a Image é alterada no combobox"""
        selection = self.cmb_image.get()
        self._update_scene(selection)
        self.plot_interpret_image()

    def plot_interpret_image(self):
        """Plota a Image sísmica com opções de Interpretation"""
        # Limpar widgets anteriores
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Verificar se há Data para plotar
        if self.sigdata is None:
            messagebox.showerror("Errorrr", "No data available for plotting")
            return

        # Criar figura
        fig, ax = plt.subplots(figsize=(15, 7))
        im = ax.imshow(self.sigdata.T, aspect='auto', origin='upper', 
                    cmap=self.palet, vmin=self.cmin, vmax=self.cmax)
        ax.set_xlabel("Traces")
        ax.set_ylabel("Samples")
        ax.set_title(f"Interpretation{self.sigfilt}")

        # Criar canvas
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        
        # Toolbar
        toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
        toolbar.update()
        toolbar.pack(fill=X)

        # ReDraw linhas existentes
        for letra, pontos in self.reflectors.items():
            if pontos:
                xs = [pt["x"] for pt in pontos]
                ys = [pt["y"] for pt in pontos]
                line, = ax.plot(xs, ys, marker='o', 
                            color=self.reflector_colors.get(letra, "black"), 
                            label=f"Refletor {letra}")
                self.interpret_lines[letra] = line
        
        if self.interpret_lines:
            ax.legend()

        # Eventos de mouse
        def on_click(event):
            if event.inaxes != ax:
                return

            # Modo desenho
            if self.modo_interpretacao.get() == "desenho":
                x, y = int(event.xdata), int(event.ydata)
                r = self.current_reflector.get()

                if r not in self.reflectors:
                    self.reflectors[r] = []

                self.reflectors[r].append({"x": x, "y": y})

                xs = [pt["x"] for pt in self.reflectors[r]]
                ys = [pt["y"] for pt in self.reflectors[r]]

                if r in self.interpret_lines:
                    self.interpret_lines[r].set_data(xs, ys)
                else:
                    line, = ax.plot(xs, ys, marker='o', 
                                color=self.reflector_colors.get(r, "black"), 
                                label=f"Refletor {r}")
                    self.interpret_lines[r] = line

                ax.legend()
                canvas.draw()

            # Modo edição - Select ponto
            elif self.modo_interpretacao.get() == "edicao":
                x_click, y_click = int(event.xdata), int(event.ydata)
                tol = 5

                for letra, pontos in self.reflectors.items():
                    for idx, pt in enumerate(pontos):
                        if abs(pt["x"] - x_click) <= tol and abs(pt["y"] - y_click) <= tol:
                            self._dragging_point = idx
                            self._drag_reflector = letra
                            return

        def on_drag(event):
            if self.modo_interpretacao.get() != "edicao":
                return
            if self._dragging_point is None or self._drag_reflector is None:
                return
            if event.inaxes != ax:
                return

            x_new, y_new = int(event.xdata), int(event.ydata)
            idx = self._dragging_point
            letra = self._drag_reflector

            self.reflectors[letra][idx]["x"] = x_new
            self.reflectors[letra][idx]["y"] = y_new

            xs = [pt["x"] for pt in self.reflectors[letra]]
            ys = [pt["y"] for pt in self.reflectors[letra]]

            self.interpret_lines[letra].set_data(xs, ys)
            canvas.draw()

        def on_release(event):
            self._dragging_point = None
            self._drag_reflector = None

        # Conectar eventos
        canvas.mpl_connect("button_press_event", on_click)
        canvas.mpl_connect("motion_notify_event", on_drag)
        canvas.mpl_connect("button_release_event", on_release)

        # Save referências
        self._current_canvas = canvas
        self._current_ax = ax

    def export_reflectors_csv(self):
        """Exporta Reflectores para CSV."""
        if not self.reflectors:
            messagebox.showwarning("Nothing to Export", "No Reflectors were designed.")
            return

        try:
            velocity = float(self.entry_velocity.get())
        except ValueError:
            messagebox.showerror("Errorr", "Invalid velocity.")
            return

        filepath = filedialog.asksaveasfilename(defaultextension=".csv", 
                                              filetypes=[("CSV", "*.csv")])
        if not filepath:
            return

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Reflector", "X_trace", "Y_sample", "Time_ms", "Depth_m"])

            for letra, pontos in self.reflectors.items():
                for pt in pontos:
                    x_trace = pt["x"]
                    y_sample = pt["y"] 
                    tempo_ms = y_sample * self.sr * 1000
                    prof_m = tempo_ms * velocity / 2000  # Conversão para metros
                    writer.writerow([letra, x_trace, y_sample, round(tempo_ms, 2), round(prof_m, 2)])

        messagebox.showinfo("Exportation completed", f"Arquivo salvo em:\n{filepath}")

    def export_reflectors_dxf(self):
        """Exporta Reflectores para DXF."""
        if not self.reflectors:
            messagebox.showwarning("Nothing to Export", "No Reflectors were designed.")
            return

        filepath = filedialog.asksaveasfilename(defaultextension=".dxf", 
                                              filetypes=[("DXF", "*.dxf")])
        if not filepath:
            return

        try:
            doc = ezdxf.new()
            msp = doc.modelspace()

            for letra, pontos in self.reflectors.items():
                color = self.get_dxf_color(letra)
                points_3d = []
                
                for pt in pontos:
                    x = pt["x"]
                    y = pt["y"]
                    z = y * self.sr * 1000  # tempo em ms como Z
                    points_3d.append((x, y, z))

                if len(points_3d) >= 2:
                    layer_name = f"Refletor_{letra}"
                    if not doc.layers.has_entry(layer_name):
                        doc.layers.add(name=layer_name, color=color)
                    msp.add_polyline3d(points_3d, dxfattribs={"layer": layer_name, "color": color})

            doc.saveas(filepath)
            messagebox.showinfo("Exportação DXF", f"Arquivo salvo com sucesso:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Errorrr", f"Falha ao exportar DXF: {e}")

    def get_dxf_color(self, letra):
        """Retorna cor DXF para Reflector."""
        cores = {"A": 1, "B": 5, "C": 3, "D": 6, "E": 2}
        return cores.get(letra.upper(), 7)
    
    def exportDXF(self):
        filenames = filedialog.askopenfilename(filetypes=[("SEG-Y", "*.sgy"), ("SEG files", "*.seg"), 
            ("All Files", "*.*")], multiple=True)
        if not filenames:
            return
        
        # Solicitar onde Save o File DXF combinado
        output_file = filedialog.asksaveasfilename(
            defaultextension=".dxf",
            filetypes=[("DXF files", "*.dxf"), ("All files", "*.*")],
            title="Save File DXF combinado como..."
        )
        if not output_file:
            return
        
        # Criar um único drawing para todos os Files
        drawing = dxf.drawing(output_file)
        drawing.add_layer('Name', color=2)
        drawing.add_layer('Trace_index', color=3)
        drawing.add_layer('Line', color=4)
        
        # Cores diferentes para cada File (opcional)
        colors = [1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        color_index = 0
        
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
                        messagebox.showinfo(title='Warning', message=f'Coordinates are Missing in {filename}')
                        continue  # Pula este arquivo se não tiver coordenadas
                        
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

                    f_parts = filename.split("/")
                    segyfile = f_parts[-1]
                    name = segyfile[:-4]

                    # Usar cor diferente para cada File
                    current_color = colors[color_index % len(colors)]
                    color_index += 1

                    # -----------------------------------------------------------------------------------------

                    j = 0
                    for i in range(rec_num.size - 1):

                        if i == 0:
                            # Adicionar nome do File na primeira coordenada
                            drawing.add(dxf.text(name, insert=(x_coord[0], y_coord[0]), height=2.0, layer='NAME'))

                        # Usar a cor específica do File atual
                        drawing.add(dxf.line((x_coord[i], y_coord[i]), (x_coord[i+1], y_coord[i+1]), 
                                    color=current_color, layer='LINE'))

                        if j <= 3 :   # coloca o Field Rec Num de 5 em 5
                            j += 1
                        else:
                            trc_num = str(rec_num[i])
                            text = dxf.text(trc_num, (x_coord[i], y_coord[i]), height=0.5, rotation=0.0, layer='TRACE_INDEX')
                            drawing.add(text)    
                            j = 0

                print(f"Processado: {name}")
                
            except Exception as e:            
                print(f"Erro ao processar {filename}: {str(e)}")
                continue

        # Save o File DXF combinado uma única vez
        try:
            drawing.save()
            messagebox.showinfo(title='Success', message=f'Arquivo DXF combinado salvo como: {output_file}')
            print(f"Arquivo DXF combinado salvo: {output_file}")
        except Exception as e:
            messagebox.showerror(title='Errorrr', message=f'Erro ao salvar arquivo DXF: {str(e)}')
            print(f"Erro ao salvar: {str(e)}") 

    def exportCSV(self):
        filenames = filedialog.askopenfilename(filetypes=[("SEG-Y", "*.sgy"), ("SEG files", "*.seg"), 
                    ("All Files", "*.*")], multiple=True)
        if not filenames:
            return
        
        # Solicitar onde Save o File CSV combinado
        output_file = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save File CSV combinado como..."
        )
        if not output_file:
            return
        
        all_data = []
        
        for filename in filenames:
            try:
                with segyio.open(filename, ignore_geometry=True) as f:
                    # Get basic attributes
                    fr = segyio.TraceField.FieldRecord
                    rec_num = f.attributes(fr)[:]  # array of field record number
                    x_coord = f.attributes(segyio.TraceField.SourceX)[:]  # array of x coord
                    y_coord = f.attributes(segyio.TraceField.SourceY)[:]  # array of y coord
                    Scalar = f.attributes(segyio.TraceField.SourceGroupScalar)[:]
                    units = f.attributes(segyio.TraceField.CoordinateUnits)[:]

                    if x_coord[10] == 0:
                        messagebox.showinfo(title='Warning', message=f'Coordinates are Missing in {filename}')
                        continue  # Pula este arquivo se não tiver coordenadas
                        
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

                    f_parts = filename.split("/")
                    segyfile = f_parts[-1]
                    name = segyfile[:-4]

                    # -----------------------------------------------------------------------------------------

                    # Criar lista de Data para este File
                    for i in range(rec_num.size):
                        all_data.append({
                            'File': name,
                            'Trace_Number': i + 1,
                            'Field_Record': rec_num[i],
                            'X_Coordinate': x_coord[i],
                            'Y_Coordinate': y_coord[i]
                        })

                print(f"Processado: {name}")
                
            except Exception as e:            
                print(f"Erro ao processar {filename}: {str(e)}")
                continue

        # Save o File CSV combinado
        if all_data:
            try:
                df = pd.DataFrame(all_data)
                df.to_csv(output_file, index=False)
                messagebox.showinfo(title='Success', 
                                message=f'Arquivo CSV salvo com {len(all_data)} registros:\n{output_file}')
                print(f"Arquivo CSV salvo: {output_file} ({len(all_data)} registros)")
            except Exception as e:
                messagebox.showerror(title='Errorrr', message=f'Erro ao salvar arquivo CSV: {str(e)}')
                print(f"Erro ao salvar: {str(e)}")
        else:
            messagebox.showwarning(title='Warning', message='None dado foi processed')

    def open_navigation_window(self):

        self.navegation = Toplevel(self.root)
        self.navegation.title("POSITIONING")
        self.navegation.geometry("300x150")
        w = 280
        h = 150
        ws = self.navegation.winfo_screenwidth()
        hs = self.navegation.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        self.navegation.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.navegation.resizable(width=False, height=False)
        self.navegation.attributes('-toolwindow', True)

        Label(self.navegation, text="Extract coordinates from SEG-Y").pack(pady=10)
        export_dxf_button = Button(self.navegation, text="Export to DXF", command=self.exportDXF, 
                                   bg="#4A4A4A", fg='white',width=20)
        export_dxf_button.pack(pady=5)

        export_csv_button = Button(self.navegation, text="Export to CSV", command=self.exportCSV, 
                                   bg="#979696", fg='white', width=20)
        export_csv_button.pack(pady=5)


if __name__ == '__main__':
    root = Tk()
    app = SeismicApp(root)
    root.mainloop() 




                
