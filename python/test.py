from butterworth_filter import butter
from scipy.signal import freqz
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import scipy.fftpack

"""
std::vector<double> filterSignal(const std::vector<double>& signal, const std::vector<double>& b, const std::vector<double>& a)
{
    std::vector<double> output(signal.size());
    std::vector<double> state(a.size() - 1, 0.0);

    for (size_t n = 0; n < signal.size(); ++n) {
        // Calcular la salida en el instante n
        output[n] = b[0] * signal[n];

        for (size_t i = 1; i < b.size() && n >= i; ++i)
            output[n] += b[i] * signal[n - i];

        for (size_t i = 1; i < a.size() && n >= i; ++i)
            output[n] -= a[i] * state[state.size() - i];

        // Actualizar los estados anteriores
        for (size_t i = state.size() - 1; i > 0; --i)
            state[i] = state[i - 1];

        state[0] = output[n];
    }

    return output;
}
"""

def filter_signal(signal, b, a):

    output = np.zeros(len(signal))

    for n in range(len(signal)):
        output[n] = b[0] * signal[n]

        for i in range(1, len(b)):
            if n >= i:
                output[n] += b[i] * signal[n-i]
            else:
                break

        for i in range(1, len(a)):
            if (n>0) & (n>=i):
                output[n] -= a[i]*output[n-i]
            else:
                break

    return output

def pure_tone(f, sr, duration):
    return (np.sin(2 * np.pi * np.arange(sr * duration) * f / sr)).astype(np.float32)

if __name__ == '__main__':

    # Generación del filtro:
    N = 3 #Orden del filtro
    fc = [6000, 12000] #Frecuencia de corte
    btype = 'bs'
    analog = False
    output = 'ba'
    fs = 44100

    #Obtengo los coeficientes de los filtros:
    b, a = butter(N, fc, btype, analog, output, fs)
    print(b)
    print(a)

    #Grafico el filtro:
    w, h = freqz(b, a, fs=fs, worN=2048)

    plt.plot(w, abs(h))
    plt.axhline(y=0.707, color='r', linestyle='--')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)

    plt.show()

    # Filtrado de la señal:
    f = 1000
    duration = 1.0

    data = pure_tone(f, fs, duration)
    sd.play(data, fs)
    sd.wait()

    filter_data = filter_signal(data, b, a)
    sd.play(filter_data, fs)
    sd.wait()

    plt.plot(data)
    plt.plot(filter_data)
    plt.show()

    T = 1/fs
    N = len(filter_data)
    yf = scipy.fftpack.fft(filter_data)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

    fig, ax = plt.subplots()
    ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    plt.show()