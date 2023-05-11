from butterworth_filter import butter
from scipy.signal import freqz
import matplotlib.pyplot as plt

if __name__ == '__main__':

    N = 10 #Orden del filtro
    fc = [6000] #Frecuencia de corte
    btype = 'hp'
    analog = False
    output = 'ba'
    fs = 44100

    #Obtengo los coeficientes de los filtros:
    b, a = butter(N, fc, btype, analog, output, fs)
    #print(b)
    #print(a)

    #Grafico el filtro:
    w, h = freqz(b, a, fs=fs, worN=2048)

    plt.plot(w, abs(h))
    plt.axhline(y=0.707, color='r', linestyle='--')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)

    plt.show()
