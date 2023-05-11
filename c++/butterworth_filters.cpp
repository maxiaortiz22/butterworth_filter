#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include "butterworth_filters.h"

using namespace std;

extern double *filtered_signal;

// Función para generar coeficientes de filtro pasa bajos Butterworth de segundo orden

void butterworthLowpass(double fs, double fc, double *b, double *a)
{
    const double omega_c = 2.0 * M_PI * fc / fs;
    const double sin_omega_c = std::sin(omega_c);
    const double cos_omega_c = std::cos(omega_c);
    const double sqrt_2 = std::sqrt(2.0);
    const double beta = 1.0 / (1.0 + sqrt_2 * sin_omega_c + sin_omega_c * sin_omega_c);
    const double a0 = (1.0 - cos_omega_c) * 0.5;
    const double a1 = 1.0 - cos_omega_c;
    const double a2 = (1.0 - cos_omega_c) * 0.5;
    const double b0 = (1.0 - cos_omega_c) * 0.5 * beta;
    const double b1 = (1.0 - cos_omega_c) * beta;
    const double b2 = (1.0 - cos_omega_c) * 0.5 * beta;
    
    // Assign the coefficients to the output arrays
    b[0] = b0;
    b[1] = b1;
    b[2] = b2;
    a[0] = 1.0;
    a[1] = -a1;
    a[2] = -a2;
}

// Función para generar coeficientes de filtro pasa altos Butterworth de segundo orden
void butterworthHighpass(double fs, double fc, double *b, double *a){

    double omega = 2.0 * M_PI * fc / fs;
    double c = 1.0 / tan(omega);
    double a0 = 1.0 + sqrt(2) * c + pow(c, 2);

    b[0] = 1.0 / a0;
    b[1] = -2.0 / a0;
    b[2] = 1.0 / a0;
    a[0] = 1.0;
    a[1] = 2.0 * (pow(c, 2.0) - 1.0) / a0;
    a[2] = (1.0 - sqrt(2.0) * c + pow(c, 2.0)) / a0;
}

/*
// Función para generar coeficientes de filtro pasa banda Butterworth de segundo orden
void butterworthBandpass(int fs, int f_low, int f_high, double* b, double* a) {

    double w_low = 2 * M_PI * f_low / fs; // Frecuencia angular de corte inferior
    double w_high = 2 * M_PI * f_high / fs; // Frecuencia angular de corte superior
    double BW = w_high - w_low; // Ancho de banda
    
    double Q = sqrt(2) / 2; // Factor de calidad (Q) para filtros de Butterworth de segundo orden
    double alpha = sin(BW) / (2 * Q); // Factor de realimentación
    
    b[0] = alpha;
    b[1] = 0;
    b[2] = -alpha;
    a[0] = 1 + alpha;
    a[1] = -2 * cos(BW);
    a[2] = 1 - alpha;

    // Normalización
    for (int i = 0; i < 3; i++) {
        b[i] /= a[0];
        a[i] /= a[0];
    }
}
*/

void butterworthBandpass(int fs, int f_low, int f_high, double* b, double* a) {
    // Calcular las frecuencias de corte normalizadas a1 y a2
    double a1 = tan(M_PI * f_low / fs);
    double a2 = tan(M_PI * f_high / fs);
    
    // Calcular los factores de calidad normalizados Q1 y Q2
    double bw_oct = log2(f_high / f_low);
    double Q1 = 1.0 / (2.0 * sin(M_PI * bw_oct / 2.0));
    double Q2 = Q1;
    
    // Calcular los coeficientes del filtro pasabajo
    double alpha1 = a1 * a1 * Q1;
    double b1_0 = a1 * sqrt(Q1);
    double a1_0 = 1.0 + alpha1;
    double a1_1 = -2.0 * cos(M_PI / 4.0) * a1_0;
    double a1_2 = alpha1 - a1_0;
    
    // Calcular los coeficientes del filtro pasaalto
    double alpha2 = a2 * a2 * Q2;
    double b2_0 = a2 * sqrt(Q2);
    double a2_0 = 1.0 + alpha2;
    double a2_1 = -2.0 * cos(M_PI / 4.0) * a2_0;
    double a2_2 = alpha2 - a2_0;
    
    // Calcular los coeficientes del filtro pasabanda
    double k = b2_0 / b1_0;
    b[0] = k;
    b[1] = 0.0;
    b[2] = -k;
    a[0] = a1_0 * a2_0;
    a[1] = a1_0 * a2_1 + a1_1 * a2_0;
    a[2] = a1_0 * a2_2 + a1_1 * a2_1 + a1_2 * a2_0;
}

void filter_signal(std::vector<double> signal, double *filtered_signal, double* b, double* a){

    // Inicializar las variables de delay
    double x_z1 = 0.0;
    double x_z2 = 0.0;
    double y_z1 = 0.0;
    double y_z2 = 0.0;


    // Filtrar la señal de entrada
    for (int n = 0; n < signal.size(); n++) {
        //Muestra actual
        double x_n = signal[n];
        // y[n] = b0 * x[n] + b1 * x[n-1] + b2 * x[n-2] - a1 * y[n-1] - a2 * y[n-2]  Ecuación del filtro de segundo orden
        double y_n = b[0] * x_n + b[1] * x_z1 + b[2] * x_z2 - a[1] * y_z1 - a[2] * y_z2;

        // Guardo el sample de la señal filtrada:
        filtered_signal[n] = y_n;

        // Actualizar las variables de delay
        if (n==0){
            x_z1 = x_n;
            y_z1 = y_n;
        }

        else {
            x_z2 = x_z1;
            x_z1 = x_n;
            
            y_z2 = y_z1;
            y_z1 = y_n;
        }
    }
}

// Tests functions:

void test_lowpass(double *filtered_signal, int buffer_size) {
    // Frecuencia de muestreo
    int fs = 44100;

    // Frecuencia de corte
    int fc = 1000;

    // Coeficientes del filtro
    double b[3], a[3];

    // Calcular los coeficientes del filtro
    butterworthLowpass(fs, fc, b, a);

    // Inicializar variables para el filtrado
    double x = 0; // entrada del filtro
    double y = 0; // salida del filtro

    // Frecuencia del tono puro
    double f_tone = 3000;

    // Generar una señal de tono puro
    std::vector<double> signal(44100);
    for (int i = 0; i < signal.size(); i++) {
        signal[i] = sin(2 * M_PI * f_tone * i / fs);
    }

    filter_signal(signal, filtered_signal, b, a);
}

void test_highpass(double *filtered_signal, int buffer_size) {
    // Frecuencia de muestreo
    int fs = 44100;

    // Frecuencia de corte
    int fc = 1000;

    // Coeficientes del filtro
    double b[3], a[3];

    // Calcular los coeficientes del filtro
    butterworthHighpass(fs, fc, b, a);

    // Inicializar variables para el filtrado
    double x = 0; // entrada del filtro
    double y = 0; // salida del filtro

    // Frecuencia del tono puro
    double f_tone = 250;

    // Generar una señal de tono puro
    std::vector<double> signal(44100);
    for (int i = 0; i < signal.size(); i++) {
        signal[i] = sin(2 * M_PI * f_tone * i / fs);
    }

    filter_signal(signal, filtered_signal, b, a);
}

void test_bandpass(double *filtered_signal, int buffer_size) {
    // Frecuencia de muestreo
    int fs = 44100;

    // Frecuencia de corte inferior
    int f_low = 500;

    // Frecuencia de corte superior
    int f_high = 1000;

    // Coeficientes del filtro
    double b[3], a[3];

    // Calcular los coeficientes del filtro
    butterworthBandpass(fs, f_low, f_high, b, a);

    // Inicializar variables para el filtrado
    double x = 0; // entrada del filtro
    double y = 0; // salida del filtro

    // Frecuencia del tono puro
    double f_tone = 3000;

    // Generar una señal de tono puro
    std::vector<double> signal(44100);
    for (int i = 0; i < signal.size(); i++) {
        signal[i] = sin(2 * M_PI * f_tone * i / fs);
    }

    filter_signal(signal, filtered_signal, b, a);
}
