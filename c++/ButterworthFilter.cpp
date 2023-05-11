#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <string>
#include <tuple>
#include <complex>
#include <stdexcept>
#include <numeric>

using namespace std::complex_literals;

/*
using namespace std;    
std::vector<double> signal(44100);
signal.size();

booleanos: true false

// Coeficientes del filtro
double b[3], a[3];

Para devolver muchos valores de una función: https://stackoverflow.com/questions/321068/returning-multiple-values-from-a-c-function

Mi forma de usar tuplas sirve de C++17 en adelante
*/

void butter(int N, std::vector<double> Wn, std::string btype, bool analog, double fs, double* b, double* a){
    /*
    Butterworth digital and analog filter design.
    Design an Nth-order digital or analog Butterworth filter and return
    the filter coefficients.
    Parameters
    ----------
    N : int
        The order of the filter. For 'bandpass' and 'bandstop' filters,
        the resulting order of the final second-order sections ('sos')
        matrix is ``2*N``, with `N` the number of biquad sections
        of the desired system.
    Wn : vector
        The critical frequency or frequencies. For lowpass and highpass
        filters, Wn is a scalar; for bandpass and bandstop filters,
        Wn is a length-2 sequence.
        For a Butterworth filter, this is the point at which the gain
        drops to 1/sqrt(2) that of the passband (the "-3 dB point").
        For digital filters, if `fs` is not specified, `Wn` units are
        normalized from 0 to 1, where 1 is the Nyquist frequency (`Wn` is
        thus in half cycles / sample and defined as 2*critical frequencies
        / `fs`). If `fs` is specified, `Wn` is in the same units as `fs`.
        For analog filters, `Wn` is an angular frequency (e.g. rad/s).
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}
    analog : bool
        When True, return an analog filter, otherwise a digital filter is
        returned.
    output : {'ba', 'zpk'}
        Type of output:  numerator/denominator ('ba') or pole-zero ('zpk')
    fs : float
        The sampling frequency of the digital system.
    b  : double*
        Coefficients of the numerator of the filter transfer function
    a  : double*
        Coefficients of the denominator of the filter transfer function
    */

   iirfilter(N, Wn, btype, analog, fs, b, a);

}

void iirfilter(int N, std::vector<double> Wn, std::string btype, bool analog, double fs, double* b, double* a){
    /*
    IIR digital and analog filter design given order and critical points.
    Design an Nth-order digital or analog filter and return the filter
    coefficients.
    Parameters
    ----------
    N : int
        The order of the filter. For 'bandpass' and 'bandstop' filters,
        the resulting order of the final second-order sections ('sos')
        matrix is ``2*N``, with `N` the number of biquad sections
        of the desired system.
    Wn : vector
        The critical frequency or frequencies. For lowpass and highpass
        filters, Wn is a scalar; for bandpass and bandstop filters,
        Wn is a length-2 sequence.
        For a Butterworth filter, this is the point at which the gain
        drops to 1/sqrt(2) that of the passband (the "-3 dB point").
        For digital filters, if `fs` is not specified, `Wn` units are
        normalized from 0 to 1, where 1 is the Nyquist frequency (`Wn` is
        thus in half cycles / sample and defined as 2*critical frequencies
        / `fs`). If `fs` is specified, `Wn` is in the same units as `fs`.
        For analog filters, `Wn` is an angular frequency (e.g. rad/s).
        In any case, Wn must be greater than 0. If the filter is a bandpass
        or bandstop filter, the cutoff frequencies must be passed from low to 
        high, eg: <6000, 12000>
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}
    analog : bool
        When True, return an analog filter, otherwise a digital filter is
        returned.
    output : {'ba', 'zpk'}
        Type of output:  numerator/denominator ('ba') or pole-zero ('zpk')
    fs : float
        The sampling frequency of the digital system.
    b  : double*
        Coefficients of the numerator of the filter transfer function
    a  : double*
        Coefficients of the denominator of the filter transfer function
    */

    std::vector<double> warped;

    // Calculate zeros, poles and gain of the butterworth filter of N order.
    auto [z, p, k] = buttap(N);

    // Pre-warp frequencies for digital filter design
    if (!analog){
        
        for (int i = 0; i < Wn.size(); i++) {
            Wn[i] = 2*Wn[i]/fs;
        }

        fs = 2.0;
        for (auto val : Wn) {
            warped.push_back(2 * fs * std::tan(M_PI * val / fs));
        }

    }
        
    else{
        warped = Wn;
    }
        

    // transform to lowpass, bandpass, highpass, or bandstop
    if (btype == "lowpass" || btype == "highpass"){

        if (btype == "lowpass"){
            auto [z, p, k] = lp2lp_zpk(z, p, k, warped[0]);
        }
            
        else{
            auto [z, p, k] = lp2hp_zpk(z, p, k, warped[0]);
        }

    }
        
    else{

        if (btype == "bandpass" || btype == "bandstop"){

            double bw = warped[1] - warped[0];
            double wo = std::sqrt(warped[0] * warped[1]);

            if (btype == "bandpass"){
                auto [z, p, k] = lp2bp_zpk(z, p, k, wo, bw);
            }
            else{
                auto [z, p, k] = lp2bs_zpk(z, p, k, wo, bw);
            }
        }
        
        else{
            throw std::invalid_argument("Invalid filter type (btype), it must be 'lowpass', 'highpass', 'bandpass' or 'bandstop'.");
        }
    }

    // Find discrete equivalent for digital filters:
    if (!analog){
        auto [z, p, k] = bilinear_zpk(z, p, k, fs);
    }

    // Transform to numerator/denominator ('ba') output:
    zpk2tf(z, p, k, b, a);

}

std::tuple<std::vector<double>, std::vector<std::complex<double>>, double> buttap(int N){
    /*
    Return (z,p,k) for analog prototype of Nth-order Butterworth filter.
    The filter will have an angular (e.g., rad/s) cutoff frequency of 1.
    */

    std::vector<double> z;               //Zeros
    std::vector<std::complex<double>> p; //poles
    std::vector<double> m;
    double k;                            //gain

    m = arange<double>(-N+1, N, 2.0);
    // Middle value is 0 to ensure an exactly real pole
    for(auto val : m){
        //p = -exp(1i * M_PI * m / (2 * N))
        double aux = val / (2 * N);
        p.push_back(-std::exp(1i * M_PI * aux));
    }
    
    k = 1.0;

    return std::make_tuple(z, p, k);
}

std::tuple<std::vector<double>, std::vector<std::complex<double>>, double> bilinear_zpk(std::vector<double> z, 
                                                                                        std::vector<std::complex<double>> p, 
                                                                                        double k, 
                                                                                        double fs){
    /*
    """
    Return a digital IIR filter from an analog one using a bilinear transform.
    Transform a set of poles and zeros from the analog s-plane to the digital
    z-plane using Tustin's method, which substitutes ``(z-1) / (z+1)`` for
    ``s``, maintaining the shape of the frequency response.
    Parameters
    ----------
    z : vector<double>
        Zeros of the analog filter transfer function.
    p : vector<complex>
        Poles of the analog filter transfer function.
    k : double
        System gain of the analog filter transfer function.
    fs : double
        Sample rate, as ordinary frequency (e.g., hertz). No prewarping is
        done in this function.
    Returns
    -------
    z : ndarray
        Zeros of the transformed digital filter transfer function.
    p : ndarray
        Poles of the transformed digital filter transfer function.
    k : float
        System gain of the transformed digital filter.
    */

    std::vector<double> z_z;               //Zeros
    std::vector<std::complex<double>> p_z; //poles
    double k_z;                            //Gain

    int degree = _relative_degree(z, p);

    double fs2 = 2.0 * fs;

    // Bilinear transform the poles and zeros
    for (int i = 0; i < z.size(); i++) {
        z_z[i] = (fs2 + z[i]) / (fs2 - z[i]);
    }
    for (int i = 0; i < p.size(); i++) {
        p_z[i] = (fs2 + p[i]) / (fs2 - p[i]);
    }

    // Any zeros that were at infinity get moved to the Nyquist frequency
    for (int i = 0; i < degree; i++){
        z_z.push_back(-1.0); //Es probable que esto deba ser 1 y no -1, probar después!!!!!!!!!
    }

    // Compensate for gain change
    for (int i = 0; i < z.size(); i++){
        z[i] = fs2 - z[i];
    }
    for (int i = 0; i < p.size(); i++){
        p[i] = fs2 - p[i];
    }

    auto z_prod = std::accumulate(z.begin(), z.end(), 1, std::multiplies<double>());
    auto p_prod = std::accumulate(p.begin(), p.end(), 1, std::multiplies<std::complex<double>>());

    k_z = k * std::real(z_prod / p_prod);

    return std::make_tuple(z_z, p_z, k_z);
}

std::tuple<std::vector<double>, std::vector<std::complex<double>>, double> lp2lp_zpk(std::vector<double> z, 
                                                                                     std::vector<std::complex<double>> p, 
                                                                                     double k, 
                                                                                     double wo){
    /*
    Transform a lowpass filter prototype to a different frequency.
    Return an analog low-pass filter with cutoff frequency `wo`
    from an analog low-pass filter prototype with unity cutoff frequency,
    using zeros, poles, and gain ('zpk') representation.
    Parameters
    ----------
    z : vector<double>
        Zeros of the analog filter transfer function.
    p : avector<complex>
        Poles of the analog filter transfer function.
    k : double
        System gain of the analog filter transfer function.
    wo: double
        Desired cutoff, as angular frequency (e.g., rad/s).
        Defaults to no change.
    Returns
    -------
    z : vector<double>
        Zeros of the transformed low-pass filter transfer function.
    p : vector<complex>
        Poles of the transformed low-pass filter transfer function.
    k : double
        System gain of the transformed low-pass filter.
    */

    std::vector<double> z_lp;               //Zeros
    std::vector<std::complex<double>> p_lp; //poles
    double k_lp;                            //Gain

    int degree = _relative_degree(z, p);

    // Scale all points radially from origin to shift cutoff frequency
    for (int i = 0; i < z.size(); i++){
        z_lp[i] = wo * z[i];
    }
    for (int i = 0; i < p.size(); i++){
        p_lp[i] = wo * p[i];
    }

    // Each shifted pole decreases gain by wo, each shifted zero increases it.
    // Cancel out the net change to keep overall gain the same
    k_lp = k * std::pow(wo, degree);

    return std::make_tuple(z_lp, p_lp, k_lp);
}

template<typename T>
std::vector<T> arange(T start, T stop, T step = 1) {
    std::vector<T> values;
    for (T value = start; value < stop; value += step)
        values.push_back(value);
    return values;
}

int _relative_degree(std::vector<double> z, std::vector<std::complex<double>> p){
    /*
    Return relative degree of transfer function from zeros and poles
    */
    int degree = p.size() - z.size();
    if (degree < 0){
        throw std::out_of_range("Improper transfer function. Must have at least as many poles as zeros.");
    }
    else{
        return degree;
    }
}