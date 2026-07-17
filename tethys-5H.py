import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
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
from scipy.interpolate import interp1d, UnivariateSpline

import csv
import ezdxf
from dxfwrite import DXFEngine as dxf
import utm

mpl.rcParams['figure.dpi'] = 92
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'


def equalize_for_display(image_data, v_max, alpha=0.8, log_widget=None):
    """Retorna uma versão do array pronta para exibição (-100 a 100), misturando
    clip linear tradicional com equalização de histograma (rank-based).

    FUNÇÃO DE MÓDULO (não é método de nenhuma classe) DE PROPÓSITO: é usada tanto
    por SeismicApp (img_show, Interpret, Spherical Divergence, etc.) quanto por
    GainControl (interactive_gain_editor), que são classes DIFERENTES e sem
    relação de herança entre si. Já aconteceu duas vezes de eu tentar chamar isso
    como equalize_for_display(..., log_widget=getattr(self, 'txt_edit', None)) de dentro de GainControl e quebrar com
    AttributeError, porque o método só existia em SeismicApp. Como função de
    módulo livre, qualquer código pode chamar equalize_for_display(...)
    diretamente, sem depender de qual classe/instância está chamando.

    Diagnóstico: amplitude sísmica costuma ter distribuição extremamente
    concentrada perto de zero, e clip linear sozinho não resolve isso (a maior
    parte do dado cai perto do centro da paleta de cor -- branco, se a paleta for
    divergente como 'seismic'/'seismic_r', ou cinza médio, se for 'binary'/'gray').
    Detecta dados "de um lado só" (STA/LTA, envelope, AGC -- sempre não-negativos)
    e usa a paleta de cor inteira nesse caso, em vez de desperdiçar metade dela.

    Args:
        image_data: array 2D com os dados a exibir.
        v_max: escala de referência para a versão de clip linear (pode ser
               None/NaN/inválido -- cai num cálculo de fallback a partir do
               próprio dado).
        alpha: não usado diretamente aqui (mantido por compatibilidade); a
               mistura entre 'equalized' e 'linear_pct' é feita pelo chamador.
        log_widget: widget Text opcional (ex: self.txt_edit) para os
               diagnósticos aparecerem na interface, além do console.

    Retorna (equalized, linear_pct) -- os dois arrays base para permitir misturar
    com um slider (como em img_show), ou já combinar num resultado único.
    """
    image_data = np.asarray(image_data, dtype=np.float64)

    # Blindagem: deconvolução por divisão espectral (scipy.signal.deconvolve) é
    # numericamente instável e pode gerar NaN/Inf com facilidade. Se isso
    # chegasse aqui sem tratamento, o rank/percentil e a normalização abaixo
    # propagariam NaN pra imagem inteira -- o matplotlib renderiza NaN como
    # branco/transparente, ou seja, "sem imagem" e sem nenhum erro visível.
    if not np.all(np.isfinite(image_data)):
        image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)

    is_one_sided = (image_data.min() >= 0) or (np.mean(image_data < 0) < 0.01)

    # Blindagem: v_max pode chegar None, NaN, string, etc. Nesse caso cai no
    # cálculo a partir do próprio dado, em vez de quebrar silenciosamente e
    # deixar a janela em branco.
    try:
        v_max = float(v_max)
        if not np.isfinite(v_max):
            v_max = 0.0
    except (TypeError, ValueError):
        v_max = 0.0

    flat_abs = np.abs(image_data).ravel()
    order = np.argsort(flat_abs, kind='quicksort')
    rank_flat = np.empty_like(order, dtype=np.float64)
    rank_flat[order] = np.arange(1, len(flat_abs) + 1)
    rank_pct = (rank_flat / len(flat_abs) * 100.0).reshape(image_data.shape)

    if is_one_sided:
        equalized = (rank_pct - 50.0) * 2.0
    else:
        sign = np.sign(image_data)
        equalized = sign * rank_pct

    vmax_ref = abs(v_max) if abs(v_max) > 0 else np.abs(image_data).max()
    if vmax_ref == 0:
        vmax_ref = 1.0
    if is_one_sided:
        vmin_ref = image_data.min()
        span = (vmax_ref - vmin_ref) or 1.0
        linear_pct = np.clip((image_data - vmin_ref) / span * 200.0 - 100.0, -100.0, 100.0)
    else:
        linear_pct = np.clip(image_data / vmax_ref * 100.0, -100.0, 100.0)

    return equalized, linear_pct


class SeismicApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TETHYS - Hi-Res Monochannel Seismic Filters and Interpretation")
        self.root.geometry("937x540+300+50")
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

        self.sd_data = self.swell_data = None
        self.bp_data = self.ormsby_data = self.wt_data = None
        self.agc_data = self.custom_gain_data = self.stalta_data = None
        self.envelope_data = None
        self.conv_rick_data = self.conv_ormsby_data = None
        self.conv_ricker_data = None  # nome real usado por apply_ricker_wavelet (conv_rick_data acima nunca é usado)
        self.deconv_bot_data = self.deconv_pred_data = None
        self.deconv_wav_data = None

        self.bp_title = self.wt_title = None
        self.agc_title = self.custom_gain_title = self.stalta_title = None
        self.conv_rick_title = self.conv_ormsby_title = None
        self.deconv_bot_title = self.deconv_pred_title = None
        self.deconv_wav_title = None
        self.options = ['Raw Data', 'Spherical Divergence', 'Swell Filter', 'Band Pass', 'Ormsby',
                        'Spectral Whitening','AGC', 'Custom Gain', 'STALTA', 'Envelope', 'Conv Ricker', 
                        'Conv Ormsby','Deconv Bottom', 'Deconv Predictive','Deconv Wavelet']

        self.soma = None  # Para deconvolução
        self.e_ini = self.e_win = self.e_sta = self.e_lta = None
        self.swell_method1 = self.swell_method2 = None

        # Variáveis para Interpretation
        self.reflectors = {}
        self.current_reflector = None
        self.reflector_colors = {"A": "red", "B": "blue", "C": "green", "D": "orange", "E": "purple"}
        self.interpret_lines = {}
        self.modo_interpretacao = None
        self._dragging_point = None
        self._drag_reflector = None

        self._setup_ui()

    def reset_application(self):
        """Reseta todas as variáveis e fecha todas as figuras ao abrir novo arquivo"""
        
        # Fechar todas as figuras matplotlib
        plt.close('all')
        
        # Fechar janelas toplevel abertas
        for widget in self.root.winfo_children():
            if isinstance(widget, Toplevel):
                try:
                    widget.destroy()
                except:
                    pass
        
        # Reset variáveis de dados
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
        
        # Reset dados filtrados
        self.sd_data = self.swell_data = None
        self.bp_data = self.ormsby_data = self.wt_data = None
        self.agc_data = self.custom_gain_data = self.stalta_data = None
        self.envelope_data = None
        self.conv_rick_data = self.conv_ormsby_data = None
        self.conv_ricker_data = None  # nome real usado por apply_ricker_wavelet (conv_rick_data acima nunca é usado)
        self.deconv_bot_data = self.deconv_pred_data = None
        self.deconv_wav_data = None
        
        # Reset títulos
        self.bp_title = self.wt_title = None
        self.agc_title = self.custom_gain_title = self.stalta_title = None
        self.conv_rick_title = self.conv_ormsby_title = None
        self.deconv_bot_title = self.deconv_pred_title = None
        self.deconv_wav_title = None
        
        # Reset scenes DataFrame
        self.scenes = pd.DataFrame(columns=['img', 'cmin', 'cmax', 'tit', 'sigfilt', 'cmap'])
        
        # Reset variáveis de interpretação
        self.reflectors = {}
        self.current_reflector = None
        self.interpret_lines = {}
        self.modo_interpretacao = None
        self._dragging_point = None
        self._drag_reflector = None
        
        # Reset outras variáveis
        self.soma = None
        self.tri = self.trf = self.spi = self.spf = 0
        self.e_ini = self.e_win = self.e_sta = self.e_lta = None
        self.swell_method1 = self.swell_method2 = None
        
        # Limpar texto
        self.txt_edit.delete(1.0, END)
        
        # Desabilitar botões até novo arquivo ser carregado
        self.btn_corrections.config(state='disabled')
        self.btn_freq_filters.config(state='disabled')
        self.btn_ampl_filters.config(state='disabled')
        self.btn_conv.config(state='disabled')
        self.btn_deconv.config(state='disabled')
        self.btn_interpretation.config(state='disabled')
        
        # Reset paleta
        self.palet = 'binary'
        self.cmap = 'binary'
        self.pal_combobox.set('Greys')
        
        # Coletar garbage
        import gc
        gc.collect()
        
        self.txt_edit.insert(END, "Application reset - Ready for new file\n")

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

        self.btn_corrections = Button(self.fr_buttons, text="CORRECTIONS", width=12, 
                       command=self.open_corrections_window, borderwidth=3, 
                       font=('sans 9 bold'), bg="#A39C6A", fg='grey20')
        self.btn_corrections.grid(row=0, column=1, padx=5, pady=5)

        self.btn_freq_filters = Button(self.fr_buttons, text="FREQ FILTERS", 
                               width=12, command=self.open_freq_filters_window, 
                               borderwidth=3, font=('sans 9 bold'), 
                               bg="#56AF59", fg='grey20')
        self.btn_freq_filters.grid(row=0, column=2, padx=5, pady=5)

        self.btn_ampl_filters = Button(self.fr_buttons, text="AMPLIT FILTERS", 
                                command=self.open_amplitude_filters_window, font=('sans 9 bold'),
                                bg='#56AF59', fg='grey20')
        self.btn_ampl_filters.grid(row=0, column=3, padx=5, pady=5)

        self.btn_conv = Button(self.fr_buttons, text="CONVOLUTION", 
                               command=self.open_conv_window, font=('sans 9 bold'),
                            bg="#F8C374", fg='grey20')
        self.btn_conv.grid(row=0, column=4, padx=4, pady=5)

        self.btn_deconv = Button(self.fr_buttons, text="DECONVOLUTION", 
                                 command=self.open_deconv_window, font=('sans 9 bold'),
                                 bg='#F8C374', fg='grey20')
        self.btn_deconv.grid(row=0, column=5, padx=5, pady=5)

        self.btn_interpretation = Button(self.fr_buttons, text="INTERPRETATION", 
                                         command=self.open_interpretation_window, font=('sans 9 bold'),
                                         bg="#D69494", fg='grey20')
        self.btn_interpretation.grid(row=0, column=6, padx=5, pady=5)

        btn_navigation = Button(self.fr_buttons, text="NAVIGATION", 
                                command=self.open_navigation_window, font=('sans 9 bold'),
                                bg="#6C838F", fg='grey20')
        btn_navigation.grid(row=0, column=7, padx=5, pady=5)        

        self.btn_corrections.config(state='disabled')
        self.btn_freq_filters.config(state='disabled')
        self.btn_ampl_filters.config(state='disabled')
        self.btn_conv.config(state='disabled')
        self.btn_deconv.config(state='disabled')
        self.btn_interpretation.config(state='disabled')

        plt.close('all')

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
        self.reset_application() 
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

                # Lê os dados e converte explicitamente para float32
                self.data = segyio.tools.collect(f.trace[:]).astype(np.float32)
                
                # ATENÇÃO: Se os dados já estão na casa de -0.1 a 0.12 na gravação int16 original,
                # a divisão abaixo os jogaria para a casa de 0.000003, destruindo o contraste.
                # Só aplique a divisão se o dado bruto original estiver na casa dos milhares (ex: 13834).
                if np.max(np.abs(self.data)) > 1.0:
                    self.data = self.data / 32768.0
                
                self.sig = np.copy(self.data)

                self.nyquist = 0.5 / self.sr
                self.fs = 1 / self.sr
                self.dt = self.sr

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

    def img_view(self):
        """Exibe a Image sísmica na janela de plotagem com contraste corrigido."""
        try:
            tmi = int(self.e_tmi.get() or 0)
            tmf = int(self.e_tmf.get() or self.ns * self.sr * 1000)
            self.tri = int(self.e_tri.get() or 0)
            self.trf = int(self.e_trf.get() or self.nt)
            self.spi = int(tmi / self.sr / 1000)
            self.spf = int(tmf / self.sr / 1000)

            # Extração segura e limpa
            matriz_real = np.array([self.data[t][self.spi:self.spf] for t in range(self.tri, self.trf)], dtype=np.float32)
            self.sigdata = matriz_real
            
            # Usamos um percentil bicaudal simétrico robusto de 95% para dados não envelopados
            # Isso impede que picos isolados do fundo do mar esmaguem a escala visual
            vm = np.percentile(np.abs(self.sigdata), 95.0)
            
            if vm == 0: 
                vm = np.max(np.abs(self.sigdata)) if np.max(np.abs(self.sigdata)) > 0 else 1.0

            self.cmin = -float(vm)
            self.cmax = float(vm)
            self.cmap = self.palet

            title = 'Working Window'
            self.sigfilt = ' over Raw Data'
            
            self.img_show(
                self.filename, title, self.sigdata, self.tri, self.trf,
                self.spi, self.spf, self.sr, self.cmin, self.cmax, self.palet, self.sigfilt
            )
            
            self.limits = True
            self.btn_corrections.config(state='normal')
            self.btn_freq_filters.config(state='normal')
            self.btn_ampl_filters.config(state='normal')
            self.btn_conv.config(state='normal')
            self.btn_deconv.config(state='normal')
            self.btn_interpretation.config(state='normal')

        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {e}")

    def _open_singleton_toplevel(self, attr_name):
        """Fecha a janela anterior rastreada em self.<attr_name> (se ainda existir)
        e cria uma nova, evitando o acúmulo de janelas duplicadas/órfãs quando o
        usuário reabre o mesmo diálogo sem fechar o anterior. Uso:

            self.xxx_window = self._open_singleton_toplevel('xxx_window')
            self.xxx_window.title('...')
            ...

        Cada janela aberta consome memória (figura do matplotlib, canvas, widgets
        Tkinter) até ser destruída; sem isso, sessões longas acumulam janelas
        invisíveis/esquecidas e degradam a performance ao longo do tempo.
        """
        prev = getattr(self, attr_name, None)
        if prev is not None:
            try:
                if prev.winfo_exists():
                    prev.destroy()
            except Exception:
                pass
        win = Toplevel(self.root)
        setattr(self, attr_name, win)
        return win

    def img_show(self, filename, title, image_data, tri, trf, spi, spf, sr, v_min, v_max, colors, filt):
        """Exibe a imagem sísmica com imshow rápido e slider de equalização de histograma.

        IMPORTANTE: usamos FigureCanvasTkAgg embutido numa janela Toplevel real do
        Tkinter, em vez de plt.show(). O app já roda seu próprio root.mainloop();
        chamar plt.show() (janela solta do matplotlib) de dentro dele faz o
        matplotlib e o Tkinter competirem pela mesma malha de eventos, e isso
        deixa os widgets interativos do matplotlib (Slider, Button) sem resposta
        a cliques/arrastos -- a janela aparece, mas os eventos não chegam ao
        widget de forma confiável. Embutir com FigureCanvasTkAgg (o mesmo padrão
        já usado em plot_interpret_image) resolve isso de vez.
        """
        # Fecha a janela/figura anterior criada por este método, se existir
        if getattr(self, '_img_show_window', None) is not None:
            try:
                self._img_show_window.destroy()
            except Exception:
                pass
        if getattr(self, '_img_show_fig', None) is not None:
            try:
                plt.close(self._img_show_fig)
            except Exception:
                pass

        win = Toplevel(self.root)
        win.title(f'{filename} - {title}{filt}')
        win.geometry('1400x800')
        self._img_show_window = win

        fig = Figure(figsize=(15, 7))
        ax1 = fig.add_subplot(111)
        self._img_show_fig = fig
        # Deixa espaço para os sliders à direita
        fig.subplots_adjust(right=0.88, left=0.08, bottom=0.12)
        
        tmi = spi * sr * 1000
        tmf = spf * sr * 1000
        twt = np.linspace(tmi, tmf, image_data.shape[1])  
        ext = [tri, trf, twt[-1], twt[0]] # Mantém o crescimento dos tempos para baixo
        
        ax1.set_xlabel('Trace number')
        ax1.set_ylabel('TWT [ms]')
        ax1.set_title(f'{filename} - {title}{filt}')
        
        ax1_smp = ax1.twinx()
        ax1_smp.set_ylabel('Samples')
        ax1_smp.set_ylim(spf, spi)
        ax1_smp.set_xlim(tri, trf)
        
        # Plot inicial usando a matriz transposta
        # interpolation='nearest': evita que o antialiasing padrão do matplotlib
        # borre/apague reflexões finas quando o registro tem muito mais amostras
        # que pixels de tela disponíveis (comum em dados de alta resolução, como
        # este arquivo: 25984 amostras por traço espremidas em ~700px de altura).
        img = ax1.imshow(image_data.T, vmin=v_min, vmax=v_max, cmap=colors, aspect='auto',
                          extent=ext, interpolation='nearest')

        # ---- EQUALIZAÇÃO DE HISTOGRAMA (rank-based) ----
        # Diagnóstico final (confirmado com dado real do usuário): a amplitude sísmica
        # tem distribuição extremamente concentrada perto de zero -- a mediana chega a
        # ser ~36x menor que o percentil 99.9%. Isso significa que QUALQUER clip linear
        # (não importa o percentual escolhido) sempre vai espremer a maioria dos dados
        # em uma fração mínima da faixa de cinza (~3%-5%), porque a relação entre o
        # "típico" e o "extremo" é grande demais para um mapeamento linear resolver.
        # Ver _equalize_for_display() para os detalhes (inclui tratamento de dados
        # "de um lado só", como STA/LTA, que nunca são negativos).
        equalized, linear_pct = equalize_for_display(image_data, v_max, log_widget=getattr(self, 'txt_edit', None))

        ax_clip = fig.add_axes([0.92, 0.15, 0.03, 0.7])
        # Desativa a navegação da toolbar (zoom/pan/coordenadas) nesse eixo.
        # Sem isso, o NavigationToolbar2Tk tenta formatar coordenadas do mouse
        # sobre o eixo do slider (que não tem dados reais) e pode lançar
        # "AttributeError: 'FigureCanvasTkAgg' object has no attribute 'renderer'".
        ax_clip.set_navigate(False)
        s_clip = Slider(ax_clip, 'Equaliz. %', 0.0, 100.0, valinit=80.0,
                         orientation='vertical', valfmt='%.0f')

        display_data = (s_clip.val / 100.0) * equalized + (1 - s_clip.val / 100.0) * linear_pct
        img.set_data(display_data.T)
        img.set_clim([-100.0, 100.0])

        def update(val):
            alpha = s_clip.val / 100.0
            blended = alpha * equalized + (1 - alpha) * linear_pct
            img.set_data(blended.T)
            fig.canvas.draw_idle()

        s_clip.on_changed(update)

        # Força o Tkinter a calcular a geometria real da janela antes do
        # primeiro desenho -- sem isso, em alguns sistemas (ex: Windows) o
        # canvas pode desenhar com tamanho 0x0 ou incorreto, resultando numa
        # figura 'desenhada' porém invisível.
        win.update_idletasks()
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        toolbar.pack(fill=X)

        # BUG CORRIGIDO: fechar a janela pelo X do sistema destruía só o widget
        # Tkinter, mas a Figure do matplotlib continuava registrada internamente
        # (nunca chamávamos plt.close(fig) nesse caminho). Qualquer plt.show()
        # solto chamado depois em OUTRA parte do app (ex: Trace View) reexibia
        # essa figura órfã, fazendo a imagem "ressuscitar" sozinha. Agora fechamos
        # a figura de verdade e limpamos as referências quando a janela é fechada.
        def on_close():
            try:
                plt.close(fig)
            except Exception:
                pass
            self._img_show_fig = None
            self._img_show_window = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)

        return self.cmin, self.cmax, colors


    def open_work_window(self):
        """Abre a janela para definir a área de trabalho."""
        self.work_window = self._open_singleton_toplevel('work_window')
        self.work_window.title("WORKING WINDOW")
        w = 275
        h = 320
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

        Button(self.work_window, width=12, text='Trace View', command=self.trace_view,
               font=('sans 9 bold'), bg="#99A730", fg='white').place(x=90, y=180)

        Button(self.work_window, width=12, text='Wiggle View', command=self.wiggle_view,
               font=('sans 9 bold'), bg="#8B6F30", fg='white').place(x=90, y=220)

        Button(self.work_window, width=10, text='Save', command=self.save_limits,
               font=('sans 9 bold'), bg="#1EAB6C", fg='white').place(x=97, y=260)
        
    def wiggle_view(self):
        """Exibe o sinal sísmico no formato 'wiggle' (traço a traço), com
        subamostragem de traços ajustável para manter a visualização legível em
        seções com muitos traços."""
        if self.sigdata is None:
            messagebox.showerror("Error", "Please load a SEG-Y file and define a working window first.")
            return

        win = self._open_singleton_toplevel('wiggle_window')
        win.title('Wiggle View')
        win.geometry('1100x750')

        n_traces, n_samples = self.sigdata.shape
        default_step = max(1, n_traces // 40)

        ctrl_frame = Frame(win)
        ctrl_frame.pack(side=TOP, fill=X)
        Label(ctrl_frame, text='Trace step:').pack(side=LEFT, padx=5, pady=5)
        step_entry = Entry(ctrl_frame, width=6)
        step_entry.insert(0, str(default_step))
        step_entry.pack(side=LEFT, padx=5)

        fig = Figure(figsize=(10, 6))
        ax_wig = fig.add_subplot(111)
        self._wiggle_fig = fig

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        toolbar.pack(fill=X)

        def draw_wiggles():
            try:
                step = int(step_entry.get())
                if step < 1:
                    step = 1
            except ValueError:
                step = default_step

            sig = self.sigdata
            t_vector = np.arange(n_samples) * self.sr * 1000
            max_val = np.percentile(np.abs(sig), 98)
            gain = step * 0.8 / (max_val if max_val != 0 else 1)

            ax_wig.clear()
            ax_wig.set_title(f'Wiggle Display (Subsampling: 1 a cada {step} traços)', fontsize=12)
            ax_wig.set_xlabel('Trace Number')
            ax_wig.set_ylabel('TWT [ms]')

            for i in range(0, n_traces, step):
                trace_plot = sig[i] * gain + i
                ax_wig.plot(trace_plot, t_vector, color='black', linewidth=0.7)
                ax_wig.fill_betweenx(t_vector, i, trace_plot, where=(trace_plot > i), color='black', alpha=0.7)

            ax_wig.set_xlim(-step, n_traces + step)
            ax_wig.set_ylim(t_vector[-1], 0)
            ax_wig.grid(True, linestyle=':', alpha=0.5)
            canvas.draw_idle()

        Button(ctrl_frame, text='Plot', command=draw_wiggles,
               font=('sans 9 bold'), bg="#307FA7", fg='white').pack(side=LEFT, padx=10)

        win.update_idletasks()
        draw_wiggles()

        def on_close():
            try:
                plt.close(fig)
            except Exception:
                pass
            self._wiggle_fig = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)

    def trace_view(self):
        """Exibe um Trace sísmico na janela de plotagem."""

        tmi = int(self.e_tmi.get() or 0)
        tmf = int(self.e_tmf.get() or self.ns * self.sr * 1000)
        self.tri = int(self.e_tri.get() or 0)
        self.trf = int(self.e_trf.get() or self.nt)
        self.spi = int(tmi / self.sr / 1000)
        self.spf = int(tmf / self.sr / 1000)

        self.sigdata = self.data[self.tri:self.trf, self.spi:self.spf]

        try:
            times = lambda x: x * self.sr * 1000
            samples = lambda x: x / (self.sr * 1000)
            tr_centre = (self.trf-self.tri) // 2

            # Mesmo padrão de img_show(): Toplevel + FigureCanvasTkAgg em vez de
            # plt.show() solto, evitando a figura ficar órfã e "ressuscitar"
            # depois (o mesmo bug corrigido em img_show).
            if getattr(self, '_trace_view_window', None) is not None:
                try:
                    self._trace_view_window.destroy()
                except Exception:
                    pass
            if getattr(self, '_trace_view_fig', None) is not None:
                try:
                    plt.close(self._trace_view_fig)
                except Exception:
                    pass

            win_tv = Toplevel(self.root)
            win_tv.title('Trace View')
            win_tv.geometry('900x600')
            self._trace_view_window = win_tv

            fig = Figure(figsize=(12, 6))
            ax = fig.add_subplot(111)
            self._trace_view_fig = fig
            ax.set_title(f'Trace View - File:{self.filename}  - Trace: {tr_centre} (middle)', fontsize=12)
  
            x = np.arange(self.spf-self.spi)
            y = self.sigdata[tr_centre]   # plot a trace
            ax.plot(x, y, color='blue')
            ax.set_xlabel('Samples', fontsize=9)
            ax.set_ylabel('Signal', fontsize=9)
            secax = ax.secondary_xaxis('top', functions=(times, samples))
            secax.set_xlabel('Time (ms)', fontsize=9)
            ax.grid()
            fig.tight_layout()

            win_tv.update_idletasks()
            canvas = FigureCanvasTkAgg(fig, master=win_tv)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=BOTH, expand=True)
            toolbar = NavigationToolbar2Tk(canvas, win_tv)
            toolbar.update()
            toolbar.pack(fill=X)

            def on_close():
                try:
                    plt.close(fig)
                except Exception:
                    pass
                self._trace_view_fig = None
                self._trace_view_window = None
                win_tv.destroy()

            win_tv.protocol("WM_DELETE_WINDOW", on_close)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to display trace view: {e}")

    def save_limits(self):
        """saved a visualização atual na lista de cenas."""
        try:
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
                self.limits = False
            self.work_window.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Select Image Limits first")

    def select_data_combobox(self, window, px, py):
        """Cria um combobox para Select o tipo de dado para filtragem."""
        cbox = ttk.Combobox(window, state="readonly", width=13)
        cbox['values'] = self.options
        cbox.place(x=px, y=py)
        cbox.bind('<<ComboboxSelected>>', lambda event: self._update_sigdata(cbox.get()))
        Label(window, text="Select Data").place(x=px+15, y=py-22)
        return cbox  # Retorna o combobox para referência
    
    def _update_sigdata(self, selection):
        """Atualiza a variável de Data com base na seleção do combobox."""
        try:
            if selection == 'Raw Data':
                self.sigdata = self.data[self.tri:self.trf, self.spi:self.spf]
                self.sigfilt = ' over Raw Data '
            elif selection == 'Spherical Divergence':
                self.sigdata = self.sd_data
                self.cmin = self.sd_data.min()
                self.cmax = self.sd_data.max()
                self.sigfilt = ' over Spherical Divergence'
                self.cmap = self.palet
            elif selection == 'Swell Filter':
                self.sigdata = self.sw_data
                self.cmin = self.sw_data.min()
                self.cmax = self.sw_data.max()
                self.sigfilt = ' over Swell Filter'
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
            elif selection == 'Custom Gain':
                custom_gain_scene = self.scenes[self.scenes['tit'].str.contains('Custom Gain')]
                if not custom_gain_scene.empty:
                    self.sigdata = custom_gain_scene.iloc[-1]['img']
                    self.cmin = custom_gain_scene.iloc[-1]['cmin']
                    self.cmax = custom_gain_scene.iloc[-1]['cmax']
                    self.sigfilt = ' over Custom Gain'
                    self.cmap = self.palet
            elif selection == 'STALTA':
                stalta_scene = self.scenes[self.scenes['tit'].str.contains('STALTA')]
                if not stalta_scene.empty:
                    self.sigdata = stalta_scene.iloc[-1]['img']
                    self.cmin = stalta_scene.iloc[-1]['cmin']
                    self.cmax = stalta_scene.iloc[-1]['cmax']
                    self.sigfilt = ' over STALTA'
                    self.cmap = self.palet
            elif selection == 'Envelope':
                env_scene = self.scenes[self.scenes['tit'].str.contains('Envelope')]
                if not env_scene.empty:
                    self.sigdata = env_scene.iloc[-1]['img']
                    self.cmin = env_scene.iloc[-1]['cmin']
                    self.cmax = env_scene.iloc[-1]['cmax']
                    self.sigfilt = ' over Envelope'
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
        """Cria um combobox para Select cenas salvas"""
        cbox = ttk.Combobox(window, state="readonly", width=13)
        cbox['values'] = self.options
        cbox.place(x=px, y=py)
        cbox.bind('<<ComboboxSelected>>', lambda event: self._update_scene(cbox.get()))
        Label(window, text="Select Scene").place(x=px+15, y=py-22)
        return cbox  # Retorna o combobox para referência
    
    def _update_scene(self, selection):
        """Atualiza a Image exibida com base na seleção do combobox."""
        try:
            if selection == 'Raw Data':
                self.sigdata = self.data[self.tri:self.trf, self.spi:self.spf]
                self.cmin = -np.percentile(self.sigdata, 95)
                self.cmax = np.percentile(self.sigdata, 95)
                self.sigfilt = ' over Raw Data'
                self.cmap = self.palet
            elif selection == 'Spherical Divergence':
                sd_scene = self.scenes[self.scenes['tit'].str.contains('Spherical Divergence')]
                if not sd_scene.empty:
                    self.sigdata = sd_scene.iloc[-1]['img']
                    self.cmin = sd_scene.iloc[-1]['cmin']
                    self.cmax = sd_scene.iloc[-1]['cmax']
                    self.sigfilt = ' over Spherical Divergence'
                    self.cmap = self.palet
            elif selection == 'Swell Filter':
                sw_scene = self.scenes[self.scenes['tit'].str.contains('Swell Filter')]
                if not sw_scene.empty:
                    self.sigdata = sw_scene.iloc[-1]['img']
                    self.cmin = sw_scene.iloc[-1]['cmin']
                    self.cmax = sw_scene.iloc[-1]['cmax']
                    self.sigfilt = ' over Swell Filter'
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
            elif selection == 'Custom Gain':
                custom_gain_scene = self.scenes[self.scenes['tit'].str.contains('Custom Gain')]
                if not custom_gain_scene.empty:
                    self.sigdata = custom_gain_scene.iloc[-1]['img']
                    self.cmin = custom_gain_scene.iloc[-1]['cmin']
                    self.cmax = custom_gain_scene.iloc[-1]['cmax']
                    self.sigfilt = ' over Custom Gain'
                    self.cmap = self.palet
            elif selection == 'STALTA':
                stalta_scene = self.scenes[self.scenes['tit'].str.contains('STALTA')]
                if not stalta_scene.empty:
                    self.sigdata = stalta_scene.iloc[-1]['img']
                    self.cmin = stalta_scene.iloc[-1]['cmin']
                    self.cmax = stalta_scene.iloc[-1]['cmax']
                    self.sigfilt = ' over STALTA'
                    self.cmap = self.palet
            elif selection == 'Envelope':
                env_scene = self.scenes[self.scenes['tit'].str.contains('Envelope')]
                if not env_scene.empty:
                    self.sigdata = env_scene.iloc[-1]['img']
                    self.cmin = env_scene.iloc[-1]['cmin']
                    self.cmax = env_scene.iloc[-1]['cmax']
                    self.sigfilt = ' over Envelope'
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
        except Exception:
            messagebox.showerror("Error", "Filter not defined or data not available.")      

    def _save_filter_result(self, filter_type):
        """salve o resultado de um Filter na lista de cenas."""
        
        if filter_type == 'Spherical Divergence' and self.sd_data is not None:
            scene = [self.sd_data, self.cmin, self.cmax, self.sd_title, self.sigfilt, self.cmap]
            self.scenes.loc[len(self.scenes)] = scene
            # self.spherical_divergence.destroy()
        elif filter_type == 'Swell Filter' and self.sw_data is not None:
            scene = [self.sw_data, self.cmin, self.cmax, self.sw_title, self.sigfilt, self.cmap]
            self.scenes.loc[len(self.scenes)] = scene
            self.apply_swell_filter.destroy()
        elif filter_type == 'Band Pass' and self.bp_data is not None:
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
        elif filter_type == 'Custom Gain' and self.custom_gain_data is not None:
            scene = [self.custom_gain_data, self.cmin, self.cmax, self.custom_gain_title, self.sigfilt, self.cmap]
            self.scenes.loc[len(self.scenes)] = scene
        elif filter_type == 'STALTA' and self.stalta_data is not None:  
            scene = [self.stalta_data, self.cmin, self.cmax, self.stalta_title, self.sigfilt, self.cmap]
            self.scenes.loc[len(self.scenes)] = scene
            self.stalta_window.destroy()
        elif filter_type == 'Envelope' and self.envelope_data is not None:
            scene = [self.envelope_data, self.cmin, self.cmax, self.envelope_title, self.sigfilt, self.cmap]
            self.scenes.loc[len(self.scenes)] = scene
            self.envelope_window.destroy()
        elif filter_type == 'Conv Ricker' and self.conv_ricker_data is not None:
            scene = [self.conv_ricker_data, self.cmin, self.cmax, self.conv_ricker_title, self.sigfilt, self.cmap]
            self.scenes.loc[len(self.scenes)] = scene
            self.ricker_window.destroy()
        elif filter_type == 'Conv Ormsby' and self.conv_ormsby_data is not None:
            scene = [self.conv_ormsby_data, self.cmin, self.cmax, self.conv_ormsby_title, self.sigfilt, self.cmap]
            self.scenes.loc[len(self.scenes)] = scene
            self.ormsby_conv_window.destroy()
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

        # print(self.scenes)

    def open_corrections_window(self):
        """Abre a janela de Corrections."""
        self.corrections_window = self._open_singleton_toplevel('corrections_window')
        self.corrections_window.title("CORRECTIONS")
        w = 280
        h = 200
        ws = self.root.winfo_screenwidth()  
        hs = self.root.winfo_screenheight()  
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        self.corrections_window.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.corrections_window.resizable(width=False, height=False)
        self.corrections_window.attributes('-toolwindow', True)

        Label(self.corrections_window, text="All corrections are applied to the raw data").place(x=30, y=30)

        Button(self.corrections_window, text='Spherical Divergence', width=22, command=self.apply_spherical_divergence,
               bg="#A39C6A", fg='white').place(x=50, y=80)
        Button(self.corrections_window, text='Swell Filter', width=22, command=self.apply_swell_filter_window,
               bg="#5F5B3E", fg='white').place(x=50, y=120)

    def apply_spherical_divergence(self):

        def spherical_divergence_correction(dat, ntraces, nsamples):

            t = [ x*self.sr for x in range(nsamples)]
            sigout = np.zeros(dat.shape)
            sd = [1/750*x for x in t]

            for k in range(ntraces):
                sigout[k,:] = dat[k,:]*sd
                
            return sigout

        self.sd_data = spherical_divergence_correction(self.sigdata, self.sigdata.shape[0], self.sigdata.shape[1])

        vm = np.percentile(self.sd_data, 95)
        self.cmin, self.cmax = -vm, vm
        self.sd_title = 'Spherical Divergence Correction'
        self.sigfilt = ' over Spherical Divergence' 
        self.cmap = self.palet

        # Mesmo padrão de img_show(): Toplevel + FigureCanvasTkAgg em vez de
        # plt.show() solto (evita botão sem resposta e figura órfã "ressuscitando"
        # depois), e equalização de histograma para contraste de verdade.
        if getattr(self, '_sd_window', None) is not None:
            try:
                self._sd_window.destroy()
            except Exception:
                pass
        if getattr(self, '_sd_fig', None) is not None:
            try:
                plt.close(self._sd_fig)
            except Exception:
                pass

        win = Toplevel(self.root)
        win.title('Spherical Divergence Correction')
        win.geometry('1100x750')
        self._sd_window = win

        fig = Figure(figsize=(13, 7))
        axes = fig.subplots(2, 1)
        self._sd_fig = fig

        try:
            eq, lin = equalize_for_display(self.sigdata, np.percentile(np.abs(self.sigdata), 95), log_widget=getattr(self, 'txt_edit', None))
            axes[0].imshow((0.8*eq + 0.2*lin).T, cmap=self.palet, aspect='auto',
                           extent=[0, self.sigdata.shape[0], self.sigdata.shape[1]*self.sr, 0], vmin=-100.0, vmax=100.0,
                           interpolation='nearest')
            axes[0].set_title('Raw Data', fontsize=12)
            axes[0].set_ylabel('Time (s)')

            eq, lin = equalize_for_display(self.sd_data, np.percentile(np.abs(self.sd_data), 95), log_widget=getattr(self, 'txt_edit', None))
            axes[1].imshow((0.8*eq + 0.2*lin).T, cmap=self.palet, aspect='auto',
                           extent=[0, self.sd_data.shape[0], self.sd_data.shape[1]*self.sr, 0], vmin=-100.0, vmax=100.0,
                           interpolation='nearest')
            axes[1].set_title('Spherical Divergence Correction', fontsize=12)
            axes[1].set_ylabel('Time (s)')
        except Exception as e:
            import traceback
            plt.close(fig)
            win.destroy()
            self._sd_fig = None
            self._sd_window = None
            messagebox.showerror("Error", f"Failed to display Spherical Divergence:\n{e}\n\n{traceback.format_exc()}")
            return

        show_btn = fig.add_axes([0.83, 0.95, 0.1, 0.05])   # define button area
        # Ver comentário em img_show(): evita o erro de renderer do
        # NavigationToolbar2Tk ao passar o mouse sobre o eixo do botão.
        show_btn.set_navigate(False)
        # put the matplotlib button in that area
        btn = Btn(show_btn, 'Save', hovercolor='gold')

        def save_and_feedback(event):
            # Mudar cor e texto do botão
            btn.label.set_text('Saved')
            btn.color = 'green'
            btn.hovercolor = 'lightgreen'
            fig.canvas.draw_idle()
            
            # Salvar dados
            self.save_sd_correction()

        btn.on_clicked(save_and_feedback)

        win.update_idletasks()
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        toolbar.pack(fill=X)

        def on_close():
            try:
                plt.close(fig)
            except Exception:
                pass
            self._sd_fig = None
            self._sd_window = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)

    def save_sd_correction(self):
        self.sd_title = 'Spherical Divergence Correction'
        self.cmap = self.palet
        self._save_filter_result('Spherical Divergence')

    def apply_swell_filter_window(self):
        """Abre a janela para o Filter Band Pass."""
        self.apply_swell_filter = self._open_singleton_toplevel('apply_swell_filter')
        self.apply_swell_filter.title("SWELL FILTER")
        w = 300
        h = 520
        ws = self.apply_swell_filter.winfo_screenwidth()
        hs = self.apply_swell_filter.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        self.apply_swell_filter.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.apply_swell_filter.resizable(width=False, height=False)
        self.apply_swell_filter.attributes('-toolwindow', True)

        self.select_data_combobox(self.apply_swell_filter, 80, 30)

        Label(self.apply_swell_filter, text='Only applicable with full working window').place(x=30, y=60)   

        Button(self.apply_swell_filter, width=15, text="Show Raw Image", command=lambda: self.show_img(self.sigdata)).place(x=80, y=90)         
        
        Label(self.apply_swell_filter, text='Start Scan of Bottom\nTracker Capture (ms)').place(x=70, y=130)
        ini1 = StringVar(self.apply_swell_filter, value='10')
        self.e_ini = Entry(self.apply_swell_filter, width=3, textvariable= ini1)
        self.e_ini.place(x=190, y=137)

        Label(self.apply_swell_filter, text='Short Time Average/Long Time Average windows').place(x=20, y=185)         
        
        Label(self.apply_swell_filter, text='STA - number of samples').place(x=50, y=215)
        ini2 = StringVar(self.apply_swell_filter, value='10')
        self.e_sta = Entry(self.apply_swell_filter, width=3, textvariable= ini2)
        self.e_sta.place(x=190, y=215)
        
        Label(self.apply_swell_filter, text='LTA - number of samples').place(x=50, y=240)
        ini3 = StringVar(self.apply_swell_filter, value='20')
        self.e_lta = Entry(self.apply_swell_filter, width=3, textvariable= ini3)
        self.e_lta.place(x=190, y=240)


        Label(self.apply_swell_filter, text="Moving Average window size (traces)").place(x=30, y=280)
        init4 = StringVar(self.apply_swell_filter, value='100')
        self.e_win = Entry(self.apply_swell_filter, width=3, textvariable= init4)
        self.e_win.place(x=235, y=280)   

        Button(self.apply_swell_filter, width=20, text="Apply Swell Correction", command= self.apply_swell_correction).place(x=70, y=320)

        Label(self.apply_swell_filter, text="Save Only the Best Method").place(x=72, y=370)
        Button(self.apply_swell_filter, width=20, text="Save with Method 1", command= self.save_method1).place(x=70, y=395)
        Button(self.apply_swell_filter, width=20, text="Save with Method 2", command= self.save_method2).place(x=70, y=425)

    def show_img(self, data):
        if getattr(self, '_show_img_window', None) is not None:
            try:
                self._show_img_window.destroy()
            except Exception:
                pass
        if getattr(self, '_show_img_fig', None) is not None:
            try:
                plt.close(self._show_img_fig)
            except Exception:
                pass

        win_si = Toplevel(self.root)
        win_si.title('Raw Data Image')
        win_si.geometry('900x600')
        self._show_img_window = win_si

        fig = Figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        self._show_img_fig = fig
        eq, lin = equalize_for_display(data, np.percentile(np.abs(data), 95), log_widget=getattr(self, 'txt_edit', None))
        ax.imshow((0.8*eq + 0.2*lin).T, cmap=self.palet, aspect='auto',
                   extent=[0, self.nt, self.ns*self.sr, 0], vmin=-100.0, vmax=100.0,
                   interpolation='nearest')
        ax.set_title('Raw Data Image')
        ax.set_xlabel('Trace Number')
        ax.set_ylabel('Time (s)')

        win_si.update_idletasks()
        canvas = FigureCanvasTkAgg(fig, master=win_si)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, win_si)
        toolbar.update()
        toolbar.pack(fill=X)

        def on_close():
            try:
                plt.close(fig)
            except Exception:
                pass
            self._show_img_fig = None
            self._show_img_window = None
            win_si.destroy()

        win_si.protocol("WM_DELETE_WINDOW", on_close)

    def apply_swell_correction(self):
            
        def moving_average(x, w):
            return np.convolve(x, np.ones(w), 'same') / w

        ini_scan  = int(self.e_ini.get())  # start scan in ms
        smp_ini = int(ini_scan/1000/self.sr) # start scan in samples

        n_sta = int(self.e_sta.get())
        n_lta = int(self.e_lta.get())

        win = int(self.e_win.get()) # win size for moving average

        # extracts Smax (SAMPLE NUMBER) from the trace
        def extractSmax(trace, sta, lta):

            I = self.classic_sta_lta_py(trace, sta, lta)
            Ymax = max(I)
            I1 = I.tolist()
            Smax = I1.index(Ymax)
            return Smax
            
        Smax = []

        for tr in range(self.nt):  
            sigtrace = self.sigdata[tr, smp_ini:]
            smax = extractSmax(sigtrace, n_sta, n_lta)  #using the ones that worked best
            Smax = np.append(Smax, smax)
        # sample number of the max intensity of each of the nt traces

        fator = 1.5                                # 1.5 is the multiplication factor
        q75, q25 = np.percentile(Smax, [75, 25])   # returns the third and first quartile
        iqr = q75 - q25                            # calculates the iqr(interquartile range)

        lowpass = q25 - (iqr * fator)              # calculates the minimum value to apply to the filter
        highpass = q75 + (iqr * fator)             # calculates the maximum value to apply to the filter

        for i in range(self.nt):                        # interpolates the range of the outlier
            if Smax[i] > highpass:
                Smax[i] = Smax[i-1] # use the previous

            if Smax[i] < lowpass:
                Smax[i] = Smax[i-1] # use the previous 

        # applying the moving average in a window of win samples
        win = int(self.e_win.get())

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
        x = np.arange(self.nt)
        cs = interpolate.CubicSpline(x, sma, axis=0)
        cs_x = np.arange(0, self.nt)
        cs_y = cs(np.arange(0, self.nt))

        # difference between the spline and theshold at each point
        # corresponds to the number of samples between the two

        dif = (cs_y - Smax) 
        dif = dif.astype(int)

        ################ METODO 1 - using insert e del)
        
        swell_met1 = self.sigdata.tolist()  # convert array to list
        for tr in range(self.nt):

            if dif[tr] > 0:

                for i in range(int(dif[tr])):

                    # insert zero at the top of the trace
                    swell_met1[tr].insert(i, 0)         
                                
                    # remove samples from the base of the trace
                    del swell_met1[tr][-i]
            
                    
            if dif[tr] >= 0 :
                for i in range(int(dif[tr])):
                    
                    # remove samples from the top of the trace
                    del swell_met1[tr][-i]
                    
                    # insert zeros at the base of the trace
                    swell_met1[tr].insert(i, 0)
                    
        self.swell_method1 = np.array(swell_met1)  # back to array
        vm = np.percentile(self.swell_method1, 95)
        self.cmin, self.cmax = -vm, vm
        self.sw_title = f'Swell Filter Method 1'
        self.cmap = self.palet

        #####################  METODO 2 - using shift()

        shft = []
        for i in range(self.nt):
            shft.append(shift(self.sigdata[i,:], dif[i], cval=0))  # opção 2 usando função shift
        self.swell_method2 = np.nan_to_num(shft).astype(int)
        vm = np.percentile(self.swell_method2, 95)
        self.cmin, self.cmax = -vm, vm
        self.sw_title = f'Swell Filter Method 2'
        self.cmap = self.palet

        # Mesmo padrão de img_show(): Toplevel + FigureCanvasTkAgg (evita a figura
        # ficar órfã e "ressuscitar" depois) e equalização de histograma.
        if getattr(self, '_swell_window', None) is not None:
            try:
                self._swell_window.destroy()
            except Exception:
                pass
        if getattr(self, '_swell_fig', None) is not None:
            try:
                plt.close(self._swell_fig)
            except Exception:
                pass

        win_sw = Toplevel(self.root)
        win_sw.title('Swell Correction')
        win_sw.geometry('1000x700')
        self._swell_window = win_sw

        fig = Figure(figsize=(10, 6))
        axes = fig.subplots(2, 1)
        self._swell_fig = fig

        eq, lin = equalize_for_display(self.swell_method1, np.percentile(np.abs(self.swell_method1), 95), log_widget=getattr(self, 'txt_edit', None))
        axes[0].imshow((0.8*eq + 0.2*lin).T, cmap=self.palet, aspect='auto',
                       extent=[0, self.swell_method1.shape[0], self.swell_method1.shape[1]*self.sr, 0],
                       vmin=-100.0, vmax=100.0, interpolation='nearest')
        axes[0].set_title('Swell Corrected Method 1', fontsize=12)
        axes[0].set_ylabel('Time (s)')

        eq, lin = equalize_for_display(self.swell_method2, np.percentile(np.abs(self.swell_method2), 95), log_widget=getattr(self, 'txt_edit', None))
        axes[1].imshow((0.8*eq + 0.2*lin).T, cmap=self.palet, aspect='auto',
                       extent=[0, self.swell_method2.shape[0], self.swell_method2.shape[1]*self.sr, 0],
                       vmin=-100.0, vmax=100.0, interpolation='nearest')
        axes[1].set_title('Swell Corrected Method 2', fontsize=12)
        axes[1].set_ylabel('Time (s)')

        fig.tight_layout()

        win_sw.update_idletasks()
        canvas = FigureCanvasTkAgg(fig, master=win_sw)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, win_sw)
        toolbar.update()
        toolbar.pack(fill=X)

        def on_close():
            try:
                plt.close(fig)
            except Exception:
                pass
            self._swell_fig = None
            self._swell_window = None
            win_sw.destroy()

        win_sw.protocol("WM_DELETE_WINDOW", on_close)

    def save_method1(self):
        self.sw_data = self.swell_method1
        self.sw_title = f'Swell Filter Method 1'
        self.cmap = self.palet
        self._save_filter_result('Swell Filter')

    def save_method2(self):
        self.sw_data = self.swell_method2
        self.sw_title = f'Swell Filter Method 2'
        self.cmap = self.palet
        self._save_filter_result('Swell Filter')

    def export_data_segy(self, data=None, label=None):
        """Exporta um array de amplitudes filtradas como um arquivo SEG-Y.

        Usado pelo botão 'Exportar SEG-Y' em cada janela de filtro (Band Pass,
        Ormsby, AGC, Custom Gain, STALTA, Envelope, Convolução e Deconvolução).
        Cada botão passa o array ESPECÍFICO daquele filtro (ex: self.stalta_data,
        self.agc_data), não um dado genérico -- assim o arquivo exportado sempre
        reflete as amplitudes daquele filtro em particular, mesmo que o usuário
        tenha navegado por outras janelas depois de aplicá-lo.

        Args:
            data: array 2D com as amplitudes filtradas a exportar. Se None, cai
                  em self.sigdata (o dado atualmente ativo) como alternativa.
            label: nome do filtro, só para a mensagem de log/erro ficar clara
                   (ex: 'STALTA', 'AGC'). Opcional.
        """
        if data is None:
            data = self.sigdata
        if data is None:
            messagebox.showerror("Error", f"No {label or 'filtered'} data available to export. Apply the filter first.")
            return

        filepath = filedialog.asksaveasfilename(defaultextension=".sgy",
                                                 filetypes=[("SEG-Y", "*.sgy"), ("All Files", "*.*")])
        if not filepath:
            return

        try:
            data = np.ascontiguousarray(data, dtype=np.float32)
            dt_us = int(round(self.sr * 1_000_000))  # segyio espera o intervalo de amostragem em microssegundos
            # format=5 (IEEE float32): evita a perda de precisão da conversão
            # para float IBM (format=1) do padrão SEG-Y clássico, e é lido sem
            # problema por segyio e pela maioria dos softwares modernos.
            segyio.tools.from_array(filepath, data, dt=dt_us, format=5)

            # segyio.tools.from_array só grava as amostras -- os cabeçalhos de
            # traço (incluindo SourceX/SourceY) saem zerados. Copiamos as
            # coordenadas e campos de navegação do arquivo original para o
            # intervalo de traços correspondente (self.tri:self.trf), reabrindo
            # o arquivo recém-criado em modo escrita.
            coords_copied = False
            if hasattr(self, 'headers') and self.headers is not None:
                tri = int(getattr(self, 'tri', 0))
                n_traces = data.shape[0]
                trf = tri + n_traces
                if trf <= len(self.headers):
                    header_fields = ['SourceX', 'SourceY', 'GroupX', 'GroupY',
                                      'SourceGroupScalar', 'CoordinateUnits', 'ElevationScalar',
                                      'offset', 'CDP', 'FieldRecord', 'TRACE_SEQUENCE_LINE']
                    available_fields = [f for f in header_fields if f in self.headers.columns]
                    if available_fields:
                        subset = self.headers.iloc[tri:trf]
                        with segyio.open(filepath, 'r+', ignore_geometry=True) as f:
                            for i in range(n_traces):
                                row = subset.iloc[i]
                                for field in available_fields:
                                    try:
                                        f.header[i][getattr(segyio.TraceField, field)] = int(row[field])
                                    except Exception:
                                        pass
                        coords_copied = True

            label_txt = f"{label} " if label else ""
            coords_txt = " (com coordenadas do arquivo original)" if coords_copied else " (SEM coordenadas -- traços fora do intervalo original ou headers indisponíveis)"
            msg = (f"\nSEG-Y exported ({label_txt}filter){coords_txt}: {filepath} "
                   f"({data.shape[0]} traces x {data.shape[1]} samples)")
            if hasattr(self, 'txt_edit'):
                self.txt_edit.insert(END, msg)
                self.txt_edit.see(END)
            messagebox.showinfo("Export completed", f"File saved in:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export SEG-Y: {e}")

    def open_freq_filters_window(self):
        """Abre a janela de Filters de Frequency."""
        self.freq_filters_window = self._open_singleton_toplevel('freq_filters_window')
        freq_filters_window = self.freq_filters_window
        freq_filters_window.title("FREQUENCIES FILTERS")
        w = 250
        h = 380
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
        Button(freq_filters_window, text='FWHM Spectrum', width=22, command=self.open_FWHM_window,
               bg="#1E753C", fg='white').place(x=40, y=150)
        Button(freq_filters_window, text='BAND-PASS FILTER', width=22, command=self.open_band_pass_window,
               bg="#295E2B", fg='white').place(x=40, y=200)
        Button(freq_filters_window, text='ORMSBY FILTER', width=22, command=self.open_ormsby_filter_window,
               bg="#1D421E", fg='white').place(x=40, y=250)
        Button(freq_filters_window, text='SPECTRAL WHITENING', width=22, command=self.open_spectral_whitening_window,
               bg="#1D421E", fg='white').place(x=40, y=300)

    def open_amplitude_spectrum_window(self):
        self.amplit_spectrum = self._open_singleton_toplevel('amplit_spectrum')
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
        self.spectrogram_window = self._open_singleton_toplevel('spectrogram_window')
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

    def open_FWHM_window(self):
        """Abre a janela para o Spectrumgrama."""
        self.FWHM_spectrum_window = self._open_singleton_toplevel('FWHM_spectrum_window')
        self.FWHM_spectrum_window.title("FWHM Spectrum")
        w = 250
        h = 250
        ws = self.root.winfo_screenwidth()  
        hs = self.root.winfo_screenheight()  
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        self.FWHM_spectrum_window.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.FWHM_spectrum_window.resizable(width=False, height=False)
        self.FWHM_spectrum_window.attributes('-toolwindow', True)

        self.select_data_combobox(self.FWHM_spectrum_window, 70, 30)
        Button(self.FWHM_spectrum_window, text='Apply', width=10, command=self.plot_FWHM_spectrum,
               bg="#2196F3", fg='white').place(x=80, y=80)

        mess = "Full Width at Half Maximum (FWHM)\nMethod for obtaining approximate frequencies in enveloped seismic signals."
        label_mess = Label(self.FWHM_spectrum_window, text=mess, justify="center", wraplength=200)
        label_mess.place(x=25, y=130)               

    def get_fwhm(self, segment):
        if len(segment) == 0 or np.max(segment) == 0: return None
        norm_env = segment / np.max(segment)
        indices = np.where(norm_env >= 0.5)[0]
        if len(indices) < 2: return None
        return (indices[-1] - indices[0]) * self.dt
    
    def run_analysis(self, window_size, overlap, blanking_ms=12.0):
            n_traces, n_samples = self.sigdata.shape

            freq_list, amp_list, t_list = [], [], []
            
            # Cálculo do índice de blanking baseado no dt da classe
            blank_idx = int(blanking_ms / (self.dt * 1000))
            
            for i in range(n_traces):
                trace = self.sigdata[i, :]
                if len(trace) <= blank_idx + window_size: continue

                # Busca o fundo do mar após o blanking
                search_zone = trace[blank_idx:]
                bottom_idx = np.argmax(search_zone) + blank_idx
                
                step = window_size - overlap
                for start in range(bottom_idx, n_samples - window_size, step):
                    segment = trace[start : start + window_size]
                    a_max = np.max(segment)
                    w_target = self.get_fwhm(segment)
                    
                    if w_target and w_target > 0 and a_max > 1e-6: # Evita divisão por zero e log de zero
                        f_app = 1 / (2 * w_target)
                        t_target = (start + np.argmax(segment)) * self.dt
                        
                        freq_list.append(f_app)
                        amp_list.append(a_max)
                        t_list.append(t_target)

            return np.array(freq_list), np.array(amp_list), np.array(t_list)

    def plot_FWHM_spectrum(self):
        freq_max_limit = 3500 # limits of X axis 
        try:
            freqs, amplitudes, tempos = self.run_analysis(window_size=100, overlap=50)

            # Filtragem pelo limite máximo definido pelo usuário
            mask_limit = freqs <= freq_max_limit
            freqs = freqs[mask_limit]
            amplitudes = amplitudes[mask_limit]
            tempos = tempos[mask_limit]

            # if len(freqs) < 10: continue

            # --- CÁLCULO DA SPLINE DE CONTORNO ---
            sort_idx = np.argsort(freqs)
            f_s, a_s = freqs[sort_idx], amplitudes[sort_idx]

            # Criar bins para encontrar os máximos locais para a spline
            bins = np.linspace(f_s.min(), f_s.max(), 40)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            max_envelope = [np.percentile(a_s[(f_s >= bins[i]) & (f_s < bins[i+1])], 95) 
                            if np.any((f_s >= bins[i]) & (f_s < bins[i+1])) else np.nan 
                            for i in range(len(bins)-1)]
            
            valid = ~np.isnan(max_envelope)
            spline_func = UnivariateSpline(bin_centers[valid], np.array(max_envelope)[valid], s=0.5)
            f_smooth = np.linspace(f_s.min(), f_s.max(), 200)

            # --- PLOTAGEM ---
            plt.figure(figsize=(10, 6))
            t_ms = tempos * 1000
            sc = plt.scatter(freqs, amplitudes, c=t_ms, cmap='gist_rainbow', alpha=0.4, s=15, edgecolors='none')
            
            # Plot da Spline de Contorno
            plt.plot(f_smooth, spline_func(f_smooth), color='blue', lw=1.5, label='Contour (Spline 95%)')

            cbar = plt.colorbar(sc, pad=0.02)
            cbar.ax.invert_yaxis() 
            cbar.set_label('Time (ms)')
            
            plt.title(f"FWHM Spectrum Analysis (Max: {freq_max_limit} Hz)\nFile: {os.path.basename(self.filename)}")
            plt.xlabel("Estimated Frequency (Hz)")
            plt.ylabel("Peak Amplitude")
            
            # O limite em X agora respeita o valor amigável definido
            plt.xlim(np.min(freqs), freq_max_limit)
            plt.ylim(0, np.max(amplitudes) * 1.1)
            plt.legend()
            plt.grid(True, alpha=0.15)

            plt.tight_layout()
            plt.show(block=False)


        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot FWHM spectrum: {e}") 
        # self.plot_FWHM_spectrum.destroy()

    def open_band_pass_window(self):
        """Abre a janela para o Filter Band Pass."""
        self.band_pass_window = self._open_singleton_toplevel('band_pass_window')
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
        Button(self.band_pass_window, width=12, text='Export SEG-Y',
               command=lambda: self.export_data_segy(self.bp_data, 'Band Pass'),
               bg="#5C5C5C", fg='white').place(x=80, y=220)
        
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
        self.ormsby_window = self._open_singleton_toplevel('ormsby_window')
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
        Button(self.ormsby_window, width=12, text='Export SEG-Y',
               command=lambda: self.export_data_segy(self.ormsby_data, 'Ormsby'),
               bg="#5C5C5C", fg='white').place(x=90, y=280)

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
        self.whitening_window = self._open_singleton_toplevel('whitening_window')
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

        self.amp_filters_window = self._open_singleton_toplevel('amp_filters_window')
        amp_filters_window = self.amp_filters_window
        amp_filters_window.title("Amplitude FILTERS")
        w = 200
        h = 270
        ws = self.root.winfo_screenwidth()
        hs = self.root.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        amp_filters_window.geometry('%dx%d+%d+%d' % (w, h, x, y))
        amp_filters_window.resizable(width=False, height=False)
        amp_filters_window.attributes('-toolwindow', True)

        Button(amp_filters_window, width=10, text='AGC', command=self.open_AGC_window,
               bg="#4CAF50", fg='white').place(x=60, y=30)
        Button(amp_filters_window, width=12, text='Custom Gain', command=self.open_custom_gain_window,
               bg="#3A853D", fg='white').place(x=60, y=80)
        Button(amp_filters_window, width=10, text='STA/LTA', command=self.open_STALTA_window,
               bg="#214D23", fg='white').place(x=60, y=130)

        # Botão Envelope: desabilitado se o dado carregado já estiver envelopado
        # (self.datatype == 1), já que aplicar envelope sobre dado já envelopado
        # não faz sentido (não muda o resultado, só reforça um viés positivo).
        btn_envelope = Button(amp_filters_window, width=10, text='Envelope', command=self.open_envelope_window,
               bg="#0D3A0F", fg='white')
        btn_envelope.place(x=60, y=180)
        if getattr(self, 'datatype', None) == 1:
            btn_envelope.config(state=DISABLED)

    def open_AGC_window(self):
        """Abre a janela para o Filter AGC."""
        self.agc_window = self._open_singleton_toplevel('agc_window')
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
        Button(self.agc_window, width=12, text='Export SEG-Y',
               command=lambda: self.export_data_segy(self.agc_data, 'AGC'),
               bg="#5C5C5C", fg='white').place(x=75, y=220)

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

    def open_custom_gain_window(self):
        """Abre a janela para o Custom Gain."""
        self.custom_gain_window = self._open_singleton_toplevel('custom_gain_window')
        self.custom_gain_window.title('Custom Gain')
        w = 250
        h = 200
        ws = self.root.winfo_screenwidth()
        hs = self.root.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        self.custom_gain_window.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.custom_gain_window.resizable(width=False, height=False)
        self.custom_gain_window.attributes('-toolwindow', True)

        Button(self.custom_gain_window, width=12, text='Apply', command=self.apply_custom_gain,
               bg="#307FA7", fg='white').place(x=75, y=80)
        Button(self.custom_gain_window, width=12, text='Export SEG-Y',
               command=lambda: self.export_data_segy(self.custom_gain_data, 'Custom Gain'),
               bg="#5C5C5C", fg='white').place(x=75, y=130)
        
        self.select_data_combobox(self.custom_gain_window, 70, 30)

    def apply_custom_gain(self):
        """Aplica Custom Gain ao sinal sísmico."""
        try:
            # IMPORTANTE: cada chamada cria uma instância NOVA de GainControl, que
            # não tem memória de uma janela aberta por uma chamada anterior (a
            # limpeza feita dentro de interactive_gain_editor só funciona se for a
            # MESMA instância). Se o usuário testar um filtro, não fechar a janela
            # do editor, e testar outro filtro, ficam janelas empilhadas -- a nova
            # pode abrir escondida atrás da antiga, parecendo "sem imagem". Por
            # isso rastreamos e fechamos a janela anterior aqui, no nível do app.
            if getattr(self, '_custom_gain_controller', None) is not None:
                prev_win = getattr(self._custom_gain_controller, '_gain_editor_window', None)
                if prev_win is not None:
                    try:
                        prev_win.destroy()
                    except Exception:
                        pass
                prev_fig = getattr(self._custom_gain_controller, '_gain_editor_fig', None)
                if prev_fig is not None:
                    try:
                        plt.close(prev_fig)
                    except Exception:
                        pass

            nsp = self.sigdata.shape[1]  # Número de amostras
            time = np.arange(nsp) * self.sr
            signal = self.sigdata.T
            controller = GainControl(signal, time=time, fs=1/self.sr, root=self.root,
                                      log_widget=getattr(self, 'txt_edit', None))
            self._custom_gain_controller = controller

            # Chama o editor interativo e obtém o resultado
            gained_signal, gain_curve = controller.interactive_gain_editor()
            
            # Verifica se o ganho foi aplicado (usuário clicou em Save)
            if controller.gain_applied:
                self.custom_gain_data = gained_signal.T  # Transpõe de volta
                
                vm = np.percentile(self.custom_gain_data, 95)
                self.cmin, self.cmax = -vm, vm
                self.custom_gain_title = 'Custom Gain'
                self.cmap = self.palet

                # Salva o filtro aplicado
                self._save_filter_result('Custom Gain')
                
                # Mostra a imagem final
                self.img_show(
                    self.filename, self.custom_gain_title, self.custom_gain_data, 
                    self.tri, self.trf, self.spi, self.spf, self.sr, 
                    self.cmin, self.cmax, self.cmap, self.sigfilt
                )
                
                self.txt_edit.insert(END, '\nCustom Gain applied')
                self.txt_edit.see(END)
                # NOTA: a janela Custom Gain (launcher) não fecha mais sozinha aqui
                # depois do Apply -- ela precisa continuar aberta para o usuário
                # poder clicar em "Exportar SEG-Y" em seguida. O usuário fecha
                # manualmente quando terminar, como as outras janelas de filtro.
            else:
                # Usuário fechou sem aplicar
                self.txt_edit.insert(END, '\nCustom Gain cancelled by user')
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply Custom Gain: {e}")
            import traceback
            traceback.print_exc()

    def open_STALTA_window(self):
        """Abre a janela para o Filter Stalta."""
        self.stalta_window = self._open_singleton_toplevel('stalta_window')
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
        Button(self.stalta_window, width=12, text='Export SEG-Y',
               command=lambda: self.export_data_segy(self.stalta_data, 'STALTA'),
               bg="#5C5C5C", fg='white').place(x=90, y=250)

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

    def open_envelope_window(self):
        """Abre a janela para o Filter Envelope (amplitude instantânea via
        transformada de Hilbert), seguindo o mesmo padrão de Select Data +
        Apply + Save usado pelo STA/LTA."""
        if self.sigdata is None:
            raise ValueError("None data loaded or selected.")

        self.envelope_window = self._open_singleton_toplevel('envelope_window')
        self.envelope_window.title('Envelope filter')
        w = 280
        h = 240
        ws = self.envelope_window.winfo_screenwidth()
        hs = self.envelope_window.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        self.envelope_window.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.envelope_window.resizable(width=False, height=False)
        self.envelope_window.attributes('-toolwindow', True)

        Button(self.envelope_window, width=12, text='Apply', command=self.show_envelope,
               bg="#307FA7", fg='white').place(x=90, y=90)
        Button(self.envelope_window, width=12, text='Save', command=lambda: self._save_filter_result('Envelope'),
               bg="#43AB1E", fg='white').place(x=90, y=130)
        Button(self.envelope_window, width=12, text='Export SEG-Y',
               command=lambda: self.export_data_segy(self.envelope_data, 'Envelope'),
               bg="#5C5C5C", fg='white').place(x=90, y=170)

        self.select_data_combobox(self.envelope_window, 70, 30)

    def show_envelope(self):
        """Calcula e exibe o envelope de amplitude instantânea (transformada de
        Hilbert). Representa a energia/intensidade da reflexão independente da
        fase, útil para realçar refletores em dado ainda não envelopado."""
        try:
            analytic_signal = signal.hilbert(self.sigdata, axis=1)
            self.envelope_data = np.abs(analytic_signal)

            vm = np.percentile(self.envelope_data, 99)
            self.cmin, self.cmax = 0, vm
            self.envelope_title = 'Instantaneous Amplitude (Envelope)'
            self.cmap = self.palet

            self.img_show(
                self.filename, self.envelope_title, self.envelope_data, self.tri, self.trf,
                self.spi, self.spf, self.sr, self.cmin, self.cmax, self.cmap, self.sigfilt
            )
            self.txt_edit.insert(END, '\nInstantaneous Amplitude (Envelope) applied')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute Envelope: {e}")

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
        self.conv_window = self._open_singleton_toplevel('conv_window')
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
        self.ricker_window = self._open_singleton_toplevel('ricker_window')
        self.ricker_window.title('RICKER WAVELET CONVOLUTION')
        w = 250
        h = 270
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
        Button(self.ricker_window, width=12, text='Export SEG-Y',
               command=lambda: self.export_data_segy(self.conv_ricker_data, 'Conv Ricker'),
               bg="#5C5C5C", fg='white').place(x=75, y=200)

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
            # NOME CORRIGIDO: usava self.ormsby_window, o MESMO atributo do filtro
            # de frequência Ormsby (open_ormsby_filter_window) -- as duas janelas
            # se fechavam uma à outra sem querer, já que _open_singleton_toplevel
            # destrói qualquer janela anterior guardada sob o mesmo nome.
            self.ormsby_conv_window = self._open_singleton_toplevel('ormsby_conv_window')
            self.ormsby_conv_window.title('ORMSBY WAVELET CONVOLUTION')
            w = 260
            h = 360
            ws = self.ormsby_conv_window.winfo_screenwidth()
            hs = self.ormsby_conv_window.winfo_screenheight()
            x = (ws/2) - (w/2)
            y = (hs/2) - (h/2)
            self.ormsby_conv_window.geometry('%dx%d+%d+%d' % (w, h, x, y))
            self.ormsby_conv_window.resizable(width=False, height=False)
            self.ormsby_conv_window.attributes('-toolwindow', True)

            # Parâmetros do Ormsby
            Label(self.ormsby_conv_window, text="Ormsby Parameters", 
                font=('Arial', 10, 'bold')).place(x=55, y=70)

            Label(self.ormsby_conv_window, text="f1 (Hz):").place(x=70, y=110)
            self.e_ormsby_f1 = Entry(self.ormsby_conv_window, width=6)
            self.e_ormsby_f1.insert(0, "200")
            self.e_ormsby_f1.place(x=120, y=110)

            Label(self.ormsby_conv_window, text="f2 (Hz):").place(x=70, y=140)
            self.e_ormsby_f2 = Entry(self.ormsby_conv_window, width=6)
            self.e_ormsby_f2.insert(0, "1000")
            self.e_ormsby_f2.place(x=120, y=140)

            Label(self.ormsby_conv_window, text="f3 (Hz):").place(x=70, y=170)
            self.e_ormsby_f3 = Entry(self.ormsby_conv_window, width=6)
            self.e_ormsby_f3.insert(0, "3000")
            self.e_ormsby_f3.place(x=120, y=170)

            Label(self.ormsby_conv_window, text="f4 (Hz):").place(x=70, y=200)
            self.e_ormsby_f4 = Entry(self.ormsby_conv_window, width=6)
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

            Button(self.ormsby_conv_window, width=12, text='Apply', command=self.apply_ormsby_wavelet,
                   bg="#307FA7", fg='white').place(x=75, y=240)
            Button(self.ormsby_conv_window, width=12, text='Save', command=lambda: self._save_filter_result('Conv Ormsby'),
                   bg="#43AB1E", fg='white').place(x=75, y=280)
            Button(self.ormsby_conv_window, width=12, text='Export SEG-Y',
                   command=lambda: self.export_data_segy(self.conv_ormsby_data, 'Conv Ormsby'),
                   bg="#5C5C5C", fg='white').place(x=75, y=320)

            self.select_data_combobox(self.ormsby_conv_window, 70, 30)   

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
        self.deconv_window = self._open_singleton_toplevel('deconv_window')
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
        self.deconv_pred_window = self._open_singleton_toplevel('deconv_pred_window')
        self.deconv_pred_window.title('PREDICTIVE DECONVOLUTION')
        w = 220
        h = 260
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
        Button(self.deconv_pred_window, width=12, text='Export SEG-Y',
               command=lambda: self.export_data_segy(self.deconv_pred_data, 'Deconv Predictive'),
               bg="#5C5C5C", fg='white').place(x=65, y=150)

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
        self.deconv_pred_title = f'Predictive Deconvolution - Order={ordem}'
        self.cmap = self.palet
        
        self.img_show(
            self.filename, self.deconv_pred_title, self.deconv_pred_data, self.tri, self.trf,
            self.spi, self.spf, self.sr, self.cmin, self.cmax, self.cmap, self.sigfilt)
        
        self.txt_edit.insert(END, f'\nPredictive Deconvolution - Order={ordem}')


# ============= Wavelet CONVOLUTION==========================================================

    def apply_wavelet_deconvolution(self):
        """Placeholder para deConvolution Wavelet."""
        self.deconv_wav_window = self._open_singleton_toplevel('deconv_wav_window')
        self.deconv_wav_window.title('WAVELET DECONVOLUTION')
        w = 220
        h = 320
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
        Button(self.deconv_wav_window, width=12, text='Export SEG-Y',
               command=lambda: self.export_data_segy(self.deconv_wav_data, 'Deconv Wavelet'),
               bg="#5C5C5C", fg='white').place(x=65, y=250)

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
        self.deconv_bot_window = self._open_singleton_toplevel('deconv_bot_window')
        self.deconv_bot_window.title('BOTTOM PULSE DECONVOLUTION')
        w = 320
        h = 290
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
        Button(self.deconv_bot_window, width=14, text='Export SEG-Y',
               command=lambda: self.export_data_segy(self.deconv_bot_data, 'Deconv Bottom'),
               bg="#5C5C5C", fg='white').place(x=90, y=200)

    def select_window_bottom_pulse(self):
        """Seleciona janela para capturar Pulse do fundo."""
        # Mesmo problema de interatividade do img_show original: plt.show() solto
        # dentro do app Tkinter deixa widgets do matplotlib (aqui, o RectangleSelector)
        # sem resposta confiável a cliques/arrastos. Usamos o mesmo padrão de
        # Toplevel + FigureCanvasTkAgg já corrigido em img_show().
        if getattr(self, '_pulse_sel_window', None) is not None:
            try:
                self._pulse_sel_window.destroy()
            except Exception:
                pass
        if getattr(self, '_pulse_sel_fig', None) is not None:
            try:
                plt.close(self._pulse_sel_fig)
            except Exception:
                pass

        win = Toplevel(self.root)
        win.title('Select Seabottom Pulse')
        win.geometry('900x700')
        self._pulse_sel_window = win

        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        self._pulse_sel_fig = fig
        ax.set_title('with mouse, select a window to capture the bottom signal')

        # Mesma equalização de histograma usada em img_show(), pelo mesmo motivo:
        # clip linear (vmin=-vm,vmax=vm) espreme a maior parte do dado perto do
        # cinza médio, dificultando enxergar o refletor de fundo pra selecionar.
        vm = np.percentile(np.abs(self.sigdata), 95)
        equalized, linear_pct = equalize_for_display(self.sigdata, vm, log_widget=getattr(self, 'txt_edit', None))
        display_data = 0.8 * equalized + 0.2 * linear_pct
        ax.imshow(display_data.T, vmin=-100.0, vmax=100.0, cmap='binary', aspect='auto', interpolation='nearest')

        def line_select_callback(eclick, erelease):
            self.x1, self.y1 = eclick.xdata, eclick.ydata
            self.x2, self.y2 = erelease.xdata, erelease.ydata
            rect = plt.Rectangle((min(self.x1, self.x2), min(self.y1, self.y2)), 
                               np.abs(self.x1 - self.x2), np.abs(self.y1 - self.y2), 
                               fill=False, edgecolor='red')
            ax.add_patch(rect)
            fig.canvas.draw_idle()

        # IMPORTANTE: guardamos o RectangleSelector em self._pulse_rs. Sem uma
        # referência persistente, o objeto seria coletado pelo garbage collector
        # assim que esta função retornasse (já que não há mais um plt.show()
        # bloqueante segurando a referência local), e o seletor pararia de
        # responder a cliques.
        self._pulse_rs = RectangleSelector(ax, line_select_callback, useblit=True, button=[1],
                            minspanx=3, minspany=3, spancoords='pixels', interactive=True)

        win.update_idletasks()
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        toolbar.pack(fill=X)

        def on_close():
            try:
                plt.close(fig)
            except Exception:
                pass
            self._pulse_sel_fig = None
            self._pulse_sel_window = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)

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

        # Mesmo padrão de img_show(): Toplevel + FigureCanvasTkAgg em vez de
        # plt.show() solto, o que também evita a figura ficar órfã e "ressuscitar"
        # depois via algum outro plt.show() do app.
        if getattr(self, '_pulse_extract_window', None) is not None:
            try:
                self._pulse_extract_window.destroy()
            except Exception:
                pass
        if getattr(self, '_pulse_extract_fig', None) is not None:
            try:
                plt.close(self._pulse_extract_fig)
            except Exception:
                pass

        win = Toplevel(self.root)
        win.title('Bottom Pulse Extraction')
        win.geometry('1200x600')
        self._pulse_extract_window = win

        fig = Figure(figsize=(10, 5))
        self._pulse_extract_fig = fig
        fig.suptitle('Bottom Pulse Extraction - Check for good Klauder wavelet (Autocorrelation)', fontsize=12)
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_title('Bottom window')
        # Mesma equalização de histograma usada em img_show()
        equalized, linear_pct = equalize_for_display(bot_win, np.percentile(np.abs(bot_win), 95), log_widget=getattr(self, 'txt_edit', None))
        display_bot_win = 0.8 * equalized + 0.2 * linear_pct
        ax1.imshow(display_bot_win.T, vmin=-100.0, vmax=100.0, cmap='gray', interpolation='nearest')

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
            win.destroy()
            self._pulse_extract_window = None
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
        
        ax2 = fig.add_subplot(1, 3, 2)
        x_axis = np.arange(s2 - s1)
        for t in range(t1, t2):
            ax2.plot(self.sigdata[t, s1:s2], -x_axis, color='gray', alpha=0.5)
        ax2.plot(self.soma, -x_axis, '-', color='r', linewidth=3)
        ax2.axvline(x=0.0, color='k', linestyle='-')
        ax2.set_title('Sum of Traces')

        ax3 = fig.add_subplot(1, 3, 3)
        ap = self.autocorrelation(self.soma)
        ax3.plot(ap)
        ax3.set_title('Autocorrelation')

        win.update_idletasks()
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        toolbar.pack(fill=X)

        def on_close():
            try:
                plt.close(fig)
            except Exception:
                pass
            self._pulse_extract_fig = None
            self._pulse_extract_window = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)

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
        self.win = self._open_singleton_toplevel('win')
        self.win.title("Seismic Interpretation")
        self.win.geometry("1500x700")

        self.reflectors = {}
        self.current_reflector = StringVar(value="A")
        self.reflector_colors = {"A": "red", "B": "blue", "C": "green", "D": "orange", "E": "purple", 
                                  "F": "cyan", "G": "magenta", "H": "yellow", "I": "brown", "J": "pink"}
        self.interpret_lines = {}

        # Variável para armazenar a figura de interpretação
        self.interpret_fig = None
        
        # Variável para armazenar a paleta escolhida (inicializa com a paleta atual)
        self.interpret_palette = StringVar(value=self.palet)

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
        
        self.cmb_image['values'] = self.options
        if self.options:
            self.cmb_image.set(self.options[0])
        self.cmb_image.pack(pady=5)

        # ===== NOVO: Seletor de Paleta =====
        Label(top_frame, text="Palette:").pack(side=LEFT, padx=10)
        palettes = ['Greys', 'seismic', 'seismic_r', 'binary', 'Spectral', 'PuOr', 'BrBG', 'PRGn', 'RdBu']
        cmb_palette = ttk.Combobox(top_frame, values=palettes, state="readonly", 
                                   textvariable=self.interpret_palette, width=12)
        cmb_palette.pack(side=LEFT, padx=5)
        cmb_palette.bind("<<ComboboxSelected>>", lambda e: self.update_palette_interpretation())

        # ===== MODO DE Interpretation =====
        self.modo_interpretacao = StringVar(value="desenho")

        # ===== BARRA INFERIOR =====
        btn_frame = Frame(self.win)
        btn_frame.pack(fill=X, pady=5)

        btn_show = Button(btn_frame, text="Show Filter", width=12,
                        command=self.show_image_interp, borderwidth=3, 
                        font=('sans 9 bold'), bg="#7F9EB7", fg='black')
        btn_show.pack(side=LEFT, padx=10)

        Label(btn_frame, text="Modo:").pack(side=LEFT)
        Radiobutton(btn_frame, text="Draw", variable=self.modo_interpretacao, 
                    value="desenho").pack(side=LEFT)
        Radiobutton(btn_frame, text="Edit", variable=self.modo_interpretacao, 
                    value="edicao").pack(side=LEFT)
        
        def close_interpretation():
            # Fecha a figura se existir
            if self.interpret_fig is not None:
                try:
                    plt.close(self.interpret_fig)
                except:
                    pass
                self.interpret_fig = None
            # Destroi a janela
            self.win.destroy()

        btn_export = Button(btn_frame, text="Export CSV", width=12,
                        command=self.export_reflectors_csv, borderwidth=3, 
                        font=('sans 9 bold'), bg="#587A5B", fg='black')
        btn_export.pack(side=LEFT, padx=10)

        btn_export_dxf = Button(btn_frame, text="Export DXF", width=12,
                        command=self.export_reflectors_dxf, borderwidth=3, 
                        font=('sans 9 bold'), bg="#5E7C82", fg='black')
        btn_export_dxf.pack(side=LEFT, padx=10)

        btn_close = Button(btn_frame, text="CLOSE", command=close_interpretation,
                        width=12, borderwidth=3, 
                        font=('sans 9 bold'), bg="#D19A9A", fg='black')
        btn_close.pack(side=LEFT)

        # ===== CANVAS =====
        self.plot_frame = Frame(self.win)
        self.plot_frame.pack(fill=BOTH, expand=True)

        # evento quando fechar a janela pelo X
        self.win.protocol("WM_DELETE_WINDOW", close_interpretation)

    def update_palette_interpretation(self):
        """Atualiza a paleta e redesenha a imagem de interpretação"""
        if hasattr(self, 'plot_frame') and self.plot_frame.winfo_exists():
            self.plot_interpret_image()

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
        # Fecha a figura anterior se existir
        if hasattr(self, 'interpret_fig') and self.interpret_fig is not None:
            try:
                plt.close(self.interpret_fig)
            except:
                pass
        # Limpar widgets anteriores
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Verificar se há Data para plotar
        if self.sigdata is None:
            messagebox.showerror("Error", "No data available for plotting")
            return

        # Obter a paleta selecionada
        current_palette = self.interpret_palette.get()

        # Criar figura
        fig = Figure(figsize=(15, 7))
        ax = fig.add_subplot(111)
        self.interpret_fig = fig  # Armazenar a figura atual
        
        # Usar a paleta selecionada no combobox
        # Mesma equalização de histograma usada em img_show() -- essa janela usava
        # vmin/vmax lineares direto (self.cmin/self.cmax), sofrendo do mesmo
        # problema de contraste (a maior parte do dado espremida perto do cinza
        # médio). As coordenadas de clique continuam em unidades de traço/amostra,
        # então a equalização não afeta a marcação de refletores.
        try:
            equalized, linear_pct = equalize_for_display(self.sigdata, self.cmax, log_widget=getattr(self, 'txt_edit', None))
            display_data = 0.8 * equalized + 0.2 * linear_pct
        except Exception as e:
            import traceback
            messagebox.showerror("Error", f"Failed to prepare image for display:\n{e}\n\n{traceback.format_exc()}")
            return
        im = ax.imshow(display_data.T, aspect='auto', origin='upper',
                    cmap=current_palette, vmin=-100.0, vmax=100.0, interpolation='nearest')
        ax.set_xlabel("Traces")
        ax.set_ylabel("Samples")
        ax.set_title(f"Interpretation{self.sigfilt} - Palette: {current_palette}")

        # Criar canvas
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        
        # Toolbar
        toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
        toolbar.update()
        toolbar.pack(fill=X)

        # Redesenhar linhas existentes
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
        
    def _interpolate_reflector_points(self, pontos, velocity):
        """Recebe os vértices esparsos digitados pelo usuário (lista de dicts
        {"x": traço, "y": amostra}) e retorna uma lista DENSA com um ponto para
        CADA traço inteiro entre os vértices (inclusive), interpolando
        linearmente a amostra (e portanto profundidade) entre vértices
        consecutivos. Também busca a amplitude real do sinal sísmico e as
        coordenadas reais do levantamento (X/Y) em cada traço, quando disponíveis.
        """
        if len(pontos) < 2:
            return []

        pts_sorted = sorted(pontos, key=lambda p: p["x"])

        # Interpola x/y para cada traço inteiro entre vértices consecutivos
        dense_xy = []
        n_seg = len(pts_sorted) - 1
        for i in range(n_seg):
            x0, y0 = pts_sorted[i]["x"], pts_sorted[i]["y"]
            x1, y1 = pts_sorted[i + 1]["x"], pts_sorted[i + 1]["y"]

            if x1 == x0:
                # Vértices no mesmo traço -- nada a interpolar nesse segmento
                if i == n_seg - 1:
                    dense_xy.append((x0, y0))
                continue

            step = 1 if x1 > x0 else -1
            # Inclui o último traço apenas no último segmento, para não duplicar
            # o ponto de junção entre segmentos consecutivos.
            xs = range(x0, x1 + step, step) if i == n_seg - 1 else range(x0, x1, step)
            for x in xs:
                frac = (x - x0) / (x1 - x0)
                y = y0 + frac * (y1 - y0)
                dense_xy.append((x, y))

        # Calcula os campos derivados (tempo, profundidade, amplitude, X/Y reais)
        result = []
        for x_trace, y_sample in dense_xy:
            time_ms = y_sample * self.sr * 1000
            depth_m = -(time_ms * velocity / 2000)  # negativo: convenção de profundidade abaixo da superfície

            trace_idx = int(round(x_trace))
            sample_idx = int(round(y_sample))
            amplitude = None
            if self.sigdata is not None and 0 <= trace_idx < self.sigdata.shape[0] \
                    and 0 <= sample_idx < self.sigdata.shape[1]:
                amplitude = float(self.sigdata[trace_idx, sample_idx])

            x_coord, y_coord = None, None
            abs_trace = int(getattr(self, 'tri', 0)) + trace_idx
            if hasattr(self, 'headers') and self.headers is not None and 0 <= abs_trace < len(self.headers):
                try:
                    row = self.headers.iloc[abs_trace]
                    sx, sy = float(row['SourceX']), float(row['SourceY'])
                    scalar = float(row['SourceGroupScalar'])
                    units = int(row['CoordinateUnits'])
                    scaler = (-1 / scalar) if scalar < 0 else (1 / scalar if scalar > 0 else 1.0)
                    sx *= scaler
                    sy *= scaler
                    if units == 2:  # lat/long -> UTM
                        long_, lat_ = sx / 3600, sy / 3600
                        UTM = utm.from_latlon(lat_, long_)
                        sx, sy = UTM[0], UTM[1]
                    x_coord, y_coord = sx, sy
                except Exception:
                    pass

            result.append({
                "trace": x_trace, "sample": y_sample,
                "time_ms": time_ms, "depth_m": depth_m,
                "amplitude": amplitude, "X": x_coord, "Y": y_coord,
            })

        return result

    def export_reflectors_csv(self):
        """Exporta Reflectores para CSV, interpolando um valor para CADA traço
        entre os vértices digitados (em vez de só os vértices em si), com
        profundidade, amplitude do sinal e coordenadas reais quando disponíveis.
        """
        if not self.reflectors:
            messagebox.showwarning("Nothing to Export", "No Reflectors were designed.")
            return

        try:
            velocity = float(self.entry_velocity.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid velocity.")
            return

        filepath = filedialog.asksaveasfilename(defaultextension=".csv", 
                                              filetypes=[("CSV", "*.csv")])
        if not filepath:
            return

        has_coords = hasattr(self, 'headers') and self.headers is not None \
            and 'SourceX' in self.headers.columns

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["Reflector", "Trace", "Sample", "Time_ms", "Depth_m", "Amplitude"]
            if has_coords:
                header += ["X", "Y"]
            writer.writerow(header)

            total_rows = 0
            for letra, pontos in self.reflectors.items():
                dense = self._interpolate_reflector_points(pontos, velocity)
                for pt in dense:
                    row = [letra, pt["trace"], round(pt["sample"], 3),
                           round(pt["time_ms"], 2), round(pt["depth_m"], 2),
                           round(pt["amplitude"], 6) if pt["amplitude"] is not None else ""]
                    if has_coords:
                        row += [round(pt["X"], 3) if pt["X"] is not None else "",
                                round(pt["Y"], 3) if pt["Y"] is not None else ""]
                    writer.writerow(row)
                    total_rows += 1

        if total_rows == 0:
            messagebox.showwarning(
                "Nothing to Export",
                "Each reflector needs at least 2 vertices to interpolate between them."
            )
            return

        messagebox.showinfo("Exportation completed", f"File saved in:\n{filepath}")

    def export_reflectors_dxf(self):
        """Exporta Reflectores para DXF, com um vértice para CADA traço entre os
        pontos digitados (não só os vértices em si), usando a mesma
        interpolação linear do export CSV.
        """
        if not self.reflectors:
            messagebox.showwarning("Nothing to Export", "No Reflectors were designed.")
            return

        try:
            velocity = float(self.entry_velocity.get())
        except ValueError:
            velocity = 1500.0  # DXF usa tempo (ms) como Z, não profundidade -- só
                                # precisamos da velocidade porque _interpolate_reflector_points
                                # a exige, mas seu valor não afeta o Z usado aqui.

        filepath = filedialog.asksaveasfilename(defaultextension=".dxf", 
                                              filetypes=[("DXF", "*.dxf")])
        if not filepath:
            return

        try:
            doc = ezdxf.new()
            msp = doc.modelspace()

            for letra, pontos in self.reflectors.items():
                color = self.get_dxf_color(letra)
                dense = self._interpolate_reflector_points(pontos, velocity)
                points_3d = [(pt["trace"], pt["sample"], pt["time_ms"]) for pt in dense]

                if len(points_3d) >= 2:
                    layer_name = f"Refletor_{letra}"
                    if not doc.layers.has_entry(layer_name):
                        doc.layers.add(name=layer_name, color=color)
                    msp.add_polyline3d(points_3d, dxfattribs={"layer": layer_name, "color": color})

            doc.saveas(filepath)
            messagebox.showinfo("DXF export", f"file saved successfully:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export DXF: {e}")

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
            title="Save DXF file as..."
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
                    # data = f.trace.raw[:]  # Get all data into memory (could cause on big files)
                    data = segyio.tools.collect(f.trace[:])
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

                print(f"Processed: {name}")
                
            except Exception as e:            
                print(f"Error processing {filename}: {str(e)}")
                continue

        # Save o File DXF combinado uma única vez
        try:
            drawing.save()
            messagebox.showinfo(title='Success', message=f'DXF file save as: {output_file}')
            print(f"{output_file} saved")
        except Exception as e:
            messagebox.showerror(title='Error', message=f'Error saving DXF file.: {str(e)}')
            print(f"Error saving: {str(e)}") 

    def exportCSV(self):
        filenames = filedialog.askopenfilename(filetypes=[("SEG-Y", "*.sgy"), ("SEG files", "*.seg"), 
                    ("All Files", "*.*")], multiple=True)
        if not filenames:
            return
        
        # Solicitar onde Save o File CSV combinado
        output_file = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save CSV file as..."
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

                print(f"Processed: {name}")
                
            except Exception as e:            
                print(f"Error processing {filename}: {str(e)}")
                continue

        # Save o File CSV combinado
        if all_data:
            try:
                df = pd.DataFrame(all_data)
                df.to_csv(output_file, index=False)
                messagebox.showinfo(title='Success', 
                                message=f'CSV file saved with {len(all_data)} regs:\n{output_file}')
                print(f"CSV saved: {output_file} ({len(all_data)} regs)")
            except Exception as e:
                messagebox.showerror(title='Error', message=f'Error saving CSV file.: {str(e)}')
                print(f"Erro ao salvar: {str(e)}")
        else:
            messagebox.showwarning(title='Warning', message='No data was processed.')

    def open_navigation_window(self):

        self.navegation = self._open_singleton_toplevel('navegation')
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

class GainControl:
    """
    Classe para aplicar ganho customizado em sinais sísmicos.
    Permite desenhar uma curva de ganho interativamente e aplicar ao sinal.
    """
    
    def __init__(self, signal, time=None, fs=1000, root=None, log_widget=None):
        """
        Inicializa o controlador de ganho.
        
        Args:
            signal: Array 1D com o sinal sísmico ou array 2D com perfil sísmico
            time: Array com os tempos correspondentes (opcional)
            fs: Frequência de amostragem se time não for fornecido
            root: janela raiz do Tkinter (necessária para abrir o editor
                  interativo embutido). Deve ser passada pelo chamador (ex:
                  SeismicApp.apply_custom_gain passa self.root).
            log_widget: widget Text do app (self.txt_edit) para os diagnósticos
                  aparecerem no mesmo lugar que os de img_show. Opcional.
        """
        self.root = root
        self.log_widget = log_widget
        self.original_signal = np.array(signal)
        # self.is_1d = len(self.original_signal.shape) == 1
        
        # if time is None:
        #     if self.is_1d:
        #         self.time = np.arange(len(signal)) / fs
        #     else:
        #         self.time = np.arange(signal.shape[0]) / fs
        # else:
        self.time = np.array(time)
        
        # Pontos de controle para a curva de ganho
        self.gain_points_t = [self.time[0], self.time[-1]]
        self.gain_points_g = [1.0, 1.0]
        
        self.current_point = None
        self.gain_applied = False
        
    def _create_gain_curve(self):
        """Cria a curva de ganho interpolada a partir dos pontos de controle."""
        if len(self.gain_points_t) < 2:
            return np.ones_like(self.time)
        
        # Ordena os pontos por tempo
        sorted_indices = np.argsort(self.gain_points_t)
        t_sorted = [self.gain_points_t[i] for i in sorted_indices]
        g_sorted = [self.gain_points_g[i] for i in sorted_indices]
        
        # Escolhe tipo de interpolação baseado no número de pontos
        if len(t_sorted) < 3:
            kind = 'linear'
        elif len(t_sorted) == 3:
            kind = 'quadratic'
        else:
            kind = 'cubic'
        
        # Interpolação
        interp_func = interp1d(t_sorted, g_sorted, kind=kind, 
                              fill_value='extrapolate', bounds_error=False)
        gain_curve = interp_func(self.time)
        
        # Limita o ganho a valores razoáveis
        gain_curve = np.clip(gain_curve, 0.1, 10.0)
        
        return gain_curve
    
    def _apply_gain_to_signal(self, gain_curve):
        """Aplica a curva de ganho ao sinal."""
        # if self.is_1d:
        #     return self.original_signal * -gain_curve
        # else:
        # Para perfil 2D, aplica ganho em cada traço
        return self.original_signal * gain_curve[:, np.newaxis]

    def interactive_gain_editor(self):
        """Abre interface interativa para editar a curva de ganho."""
        if self.root is None:
            raise RuntimeError(
                "GainControl foi criado sem 'root' (janela Tkinter). "
                "Passe root=self.root ao instanciar GainControl(...)."
            )

        # Mesmo problema do img_show original: plt.show() solto dentro do app
        # Tkinter deixa eventos de mouse (aqui, arrastar pontos da curva de ganho)
        # sem resposta confiável. Usamos Toplevel + FigureCanvasTkAgg, e
        # win.wait_window() para manter o comportamento de bloqueio que esta
        # função tinha antes (só retorna depois que o usuário clica Save/fecha).
        if getattr(self, '_gain_editor_window', None) is not None:
            try:
                self._gain_editor_window.destroy()
            except Exception:
                pass
        if getattr(self, '_gain_editor_fig', None) is not None:
            try:
                plt.close(self._gain_editor_fig)
            except Exception:
                pass

        win = Toplevel(self.root)
        win.title('Interactive Gain Editor')
        win.geometry('1000x750')
        self._gain_editor_window = win

        fig = Figure(figsize=(9, 6))
        self._gain_editor_fig = fig
        
        # Layout dos gráficos
        ax_before = fig.add_subplot(3, 1, 1)
        ax_gain = fig.add_subplot(3, 1, 2)
        ax_after = fig.add_subplot(3, 1, 3)
        
        # Blindagem: percentil sem valor absoluto pode degenerar para valores muito
        # pequenos ou ate zero dependendo da distribuição dos dados (ex: dado bruto
        # com muitas amostras zeradas), fazendo vmin=vmax e a imagem renderizar em
        # branco sem nenhum erro. Usamos abs() e garantimos um piso mínimo.
        vm = np.percentile(np.abs(self.original_signal), 95)
        if not np.isfinite(vm) or vm <= 0:
            vm = np.abs(self.original_signal).max()
        if not np.isfinite(vm) or vm <= 0:
            vm = 1.0
        # Mesma equalização de histograma usada em img_show()/_equalize_for_display()
        # em todo o resto do app: um clip linear simples (vmin=-vm,vmax=vm) deixa a
        # esmagadora maioria da amplitude sísmica (que se concentra pertinho de
        # zero) cair perto do CENTRO da paleta 'seismic_r' -- que é branco. Isso
        # fazia o painel parecer "sem imagem" para dado bruto/maioria dos filtros,
        # enquanto STA/LTA (faixa de valores bem mais equilibrada) aparecia normal.
        eq, lin = equalize_for_display(self.original_signal, vm, log_widget=self.log_widget)
        display_before = 0.8 * eq + 0.2 * lin
        im1 = ax_before.imshow(display_before, aspect='auto',
                                cmap='seismic_r', interpolation='nearest',
                                extent=[0, self.original_signal.shape[1], 
                                        self.time[-1], self.time[0]], vmin=-100.0, vmax=100.0)
        ax_before.set_title('Seismic Section (200 traces - Before Gain)')
        ax_before.set_ylabel('Time (s)')
        ax_before.set_xlabel('Trace')
        
        # Curva de ganho
        line_gain, = ax_gain.plot(self.time, np.ones_like(self.time), 'b-', linewidth=2)
        points_gain, = ax_gain.plot(self.gain_points_t, self.gain_points_g, 
                                    'ro', markersize=8, picker=5)
        ax_gain.set_title('Gain Curve - click and drag to adjust points', )
        ax_gain.set_ylabel('Gain')
        ax_gain.set_ylim(0, 5)
        ax_gain.grid(True, alpha=0.3)
        ax_gain.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        
        im2 = ax_after.imshow(display_before, aspect='auto',
                                cmap='seismic_r', interpolation='nearest',
                                extent=[0, self.original_signal.shape[1], 
                                    self.time[-1], self.time[0]], vmin=-100.0, vmax=100.0)
        ax_after.set_title('Seismic Section with Gain')
        ax_after.set_ylabel('Time (s)')
        ax_after.set_xlabel('Trace')
        
        # Botões
        ax_apply = fig.add_axes([0.7, 0.02, 0.1, 0.04])
        ax_reset = fig.add_axes([0.81, 0.02, 0.1, 0.04])
        # Ver comentário em img_show(): evita o erro de renderer do
        # NavigationToolbar2Tk ao passar o mouse sobre os eixos dos botões.
        ax_apply.set_navigate(False)
        ax_reset.set_navigate(False)
        btn_apply = Btn(ax_apply, 'Save')
        btn_reset = Btn(ax_reset, 'Reset')

        def update_plot():
            """Atualiza todos os gráficos."""
            gain_curve = self._create_gain_curve()
            signal_with_gain = self._apply_gain_to_signal(gain_curve)
            
            line_gain.set_ydata(gain_curve)
            points_gain.set_data(self.gain_points_t, self.gain_points_g)
            eq2, lin2 = equalize_for_display(signal_with_gain, np.percentile(np.abs(signal_with_gain), 95), log_widget=self.log_widget)
            im2.set_data(0.8 * eq2 + 0.2 * lin2)
            
            fig.canvas.draw_idle()
        
        def on_click(event):
            """Adiciona novo ponto de controle."""
            if event.inaxes == ax_gain and event.button == 1:
                if self.current_point is None:
                    self.gain_points_t.append(event.xdata)
                    self.gain_points_g.append(event.ydata)
                    update_plot()
        
        def on_pick(event):
            """Seleciona ponto para arrastar."""
            if event.artist == points_gain:
                self.current_point = event.ind[0]
        
        def on_motion(event):
            """Arrasta ponto selecionado."""
            if self.current_point is not None and event.inaxes == ax_gain:
                self.gain_points_t[self.current_point] = event.xdata
                self.gain_points_g[self.current_point] = max(0.1, min(10, event.ydata))
                update_plot()
        
        def on_release(event):
            """Solta o ponto."""
            self.current_point = None
        
        def close_editor():
            try:
                plt.close(fig)
            except Exception:
                pass
            self._gain_editor_fig = None
            self._gain_editor_window = None
            win.destroy()

        def on_apply(event):
            """Aplica o ganho definitivamente."""
            gain_curve = self._create_gain_curve()
            self.original_signal = self._apply_gain_to_signal(gain_curve)
            self.gain_applied = True
            close_editor()

        def on_reset(event):
            """Reseta a curva de ganho."""
            self.gain_points_t = [self.time[0], self.time[-1]]
            self.gain_points_g = [1.0, 1.0]
            update_plot()
        
        # Conecta eventos
        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('pick_event', on_pick)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)
        btn_apply.on_clicked(on_apply)
        btn_reset.on_clicked(on_reset)
        
        fig.subplots_adjust(bottom=0.1)

        win.update_idletasks()
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        toolbar.pack(fill=X)

        win.protocol("WM_DELETE_WINDOW", close_editor)

        # Garante que a janela venha para frente e receba foco -- evita que ela
        # abra escondida atrás de uma janela anterior (ex: um teste de outro
        # filtro cuja janela do editor não foi fechada), o que pareceria
        # "sem imagem" para o usuário.
        win.lift()
        win.attributes('-topmost', True)
        win.after(100, lambda: win.attributes('-topmost', False))
        win.focus_force()

        # Bloqueia aqui (via Tkinter, não via matplotlib) até o usuário clicar
        # Save/Reset->fechar ou fechar a janela -- preserva o comportamento
        # original da função, que só retorna depois que o editor é fechado.
        win.wait_window()

        return self.original_signal, self._create_gain_curve()
    
    def apply_gain_function(self, gain_func):
        """
        Aplica uma função de ganho pré-definida.
        
        Args:
            gain_func: Função que recebe tempo e retorna ganho, ou array de ganhos
        """
        if callable(gain_func):
            gain_curve = gain_func(self.time)
        else:
            gain_curve = np.array(gain_func)
        
        return self._apply_gain_to_signal(gain_curve)


if __name__ == '__main__':
    root = Tk()
    app = SeismicApp(root)
    root.mainloop() 




                
