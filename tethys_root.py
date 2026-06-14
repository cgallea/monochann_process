import numpy as np
from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import segyio

def extract_coords(file):
    with segyio.open(file, mode='r', ignore_geometry=True) as f:
        headers = segyio.tracefield.keys
        trace_headers = {}
        target_headers = ['TRACE_SEQUENCE_LINE', 'SourceX', 'SourceY', 'LagTsigeA', 'ElevationScalar']
        
        for k, v in headers.items():
            if k in target_headers:
                trace_headers[k] = f.attributes(v)[:]
    return trace_headers

def open_segy():
    # Importações tardias para inicialização veloz do executável
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button as Btn, Slider, RadioButtons, TextBox
    import matplotlib
    matplotlib.use('TkAgg') 

    segyname = askopenfilename(filetypes=[("SEG-Y", "*.sgy"), ("SEG files", "*.seg"), ("All Files", "*.*")])   
    if not segyname: return

    with segyio.open(segyname, ignore_geometry=True) as f:
        n_traces = f.tracecount
        sample_rate = segyio.tools.dt(f) / 1000000
        n_samples = f.samples.size
        data = f.trace.raw[:] 
        
        raw_header = segyio.tools.wrap(f.text[0])
        head_txt.delete('1.0', END) 
        head_txt.insert(END, '\t EBCDIC header\n')
        head_txt.insert(END, raw_header)
        head_txt.insert(END, f'\n\t File name: {segyname}')
        head_txt.insert(END, f'\n\t Number of traces: {n_traces}')
        head_txt.insert(END, f'\n\t Number of samples per trace: {n_samples}')
        head_txt.insert(END, f'\n\t Sample rate: {sample_rate*1000000} microseconds')
        head_txt.insert(END, f'\n\t Frequency: {int(1/sample_rate)} Hz\n\n')
        head_txt.insert(END, str(f.bin))

        sig = np.copy(data)

    # --- Janela de Frequências ---
    def showfreq(event):
        t = n_traces // 2
        fig_freq = plt.figure(figsize=(10, 4), constrained_layout=True)
        plt.suptitle(f'Trace {t} - Spectral Analysis', fontsize=11)

        for i, title in enumerate(['Spectrum Before', 'Spectrum After'], 1):
            ax = fig_freq.add_subplot(1, 2, i)
            xf_mag = np.abs(np.fft.fft(sig[t]))
            freqs = np.fft.fftfreq(len(xf_mag), d=sample_rate)
            
            pos_mask = freqs >= 0
            ax.plot(freqs[pos_mask], xf_mag[pos_mask], color='crimson')
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Amplitude')
            ax.set_title(title, fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    # --- Janela de Wiggles ---
    def showwiggles(event):
        # Captura o valor atual da caixa de texto
        try:
            step = int(txt_step.text)
            if step < 1: step = 1
        except ValueError:
            step = max(1, n_traces // 40) # Fallback caso o usuário apague ou digite algo inválido

        fig_wig, ax_wig = plt.subplots(figsize=(10, 6))
        ax_wig.set_title(f'Wiggle Display (Subsampling: 1 a cada {step} traços)', fontsize=12)
        ax_wig.set_xlabel('Trace Number')
        ax_wig.set_ylabel('TWT [ms]')
        
        t_vector = np.arange(n_samples) * sample_rate * 1000
        
        # Fator de ganho adaptativo baseado no step escolhido
        max_val = np.percentile(np.abs(sig), 98)
        gain = step * 0.8 / (max_val if max_val != 0 else 1)
        
        for i in range(0, n_traces, step):
            trace_plot = sig[i] * gain + i
            ax_wig.plot(trace_plot, t_vector, color='black', linewidth=0.7)
            ax_wig.fill_betweenx(t_vector, i, trace_plot, where=(trace_plot > i), color='black', alpha=0.7)
            
        ax_wig.set_xlim(-step, n_traces + step)
        ax_wig.set_ylim(t_vector[-1], 0) 
        ax_wig.grid(True, linestyle=':', alpha=0.5)
        plt.show()

    # --- Plotagem da Seção Principal ---
    fig, ax1 = plt.subplots(figsize=(13, 6)) 
    plt.subplots_adjust(left=0.08, right=0.82, bottom=0.18) # Aumentado margem inferior para os novos componentes
    
    tmf = n_samples * sample_rate * 1000
    ext = [0, n_traces, tmf, 0] 
    
    ax1.set_xlabel('Trace number')
    ax1.set_ylabel('TWT [ms]')
    
    vm = np.percentile(sig, 99)
    img = ax1.imshow(sig.T, vmin=-vm, vmax=vm, cmap='gray', aspect='auto', extent=ext)

    # --- Sliders de Ganho ---
    ax_clr_min = plt.axes([0.86, 0.2, 0.015, 0.6])
    ax_clr_max = plt.axes([0.89, 0.2, 0.015, 0.6])
    s_clr_min = Slider(ax_clr_min, 'Min', -vm, 0, valinit=-vm, orientation='vertical')
    s_clr_max = Slider(ax_clr_max, 'Max', 0, vm, valinit=vm, orientation='vertical')

    def update(val):
        img.set_clim([s_clr_min.val, s_clr_max.val])
        fig.canvas.draw_idle()

    s_clr_min.on_changed(update)
    s_clr_max.on_changed(update)

    # --- Seletor Interativo de Paletas (CMAPS) ---
    ax_radio = plt.axes([0.92, 0.4, 0.06, 0.2])
    colormaps = ('gray', 'seismic', 'RdBu', 'viridis', 'plasma')
    radio_cmap = RadioButtons(ax_radio, colormaps, active=0)

    def change_cmap(label):
        img.set_cmap(label)
        fig.canvas.draw_idle()
        
    radio_cmap.on_clicked(change_cmap)

    # --- Botão: Show Frequencies ---
    ax_btn_freq = plt.axes([0.15, 0.03, 0.18, 0.05])
    btn_freq = Btn(ax_btn_freq, 'Show Frequencies')
    btn_freq.on_clicked(showfreq)

    # --- Caixa de Texto: Subsampling Step (Ao lado do botão de Wiggles) ---
    ax_txt_step = plt.axes([0.55, 0.03, 0.06, 0.05])
    initial_step = str(max(1, n_traces // 40))
    txt_step = TextBox(ax_txt_step, 'Step: ', initial=initial_step)

    # --- Botão: Show Wiggles ---
    ax_btn_wig = plt.axes([0.63, 0.03, 0.18, 0.05])
    btn_wig = Btn(ax_btn_wig, 'Show Wiggles')
    btn_wig.on_clicked(showwiggles)

    plt.show()

# --- Interface Principal (Tkinter) ---
root = Tk()
root.title("TETHYS - SEG-Y Viewer")
root.geometry('900x520')

head_txt = Text(root, bg='white')
head_txt.pack(expand=True, fill=BOTH, side=TOP)

fr_buttons = Frame(root, relief=RAISED, bd=2, bg='#ADEAEA')
fr_buttons.pack(fill=X, side=BOTTOM)

btn_open = Button(fr_buttons, text="OPEN SEG-Y", command=open_segy, font=('sans 9 bold'))
btn_open.pack(padx=10, pady=10)

root.mainloop()