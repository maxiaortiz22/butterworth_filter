#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <string>
#include <tuple>
#include <complex>
#include <stdexcept>
#include <numeric>
#include <algorithm>

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

std::tuple<std::vector<double>, std::vector<double>> butter(int N, 
                                                            std::vector<double> Wn, 
                                                            std::string btype, 
                                                            bool analog, 
                                                            double fs){
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

   std::vector<double> b; 
   std::vector<double> a;

   auto [b, a] = iirfilter(N, Wn, btype, analog, fs, b, a);

   return std::make_tuple(b, a);

}

std::tuple<std::vector<double>, std::vector<double>> iirfilter(int N, 
                                                               std::vector<double> Wn, 
                                                               std::string btype, 
                                                               bool analog, 
                                                               double fs, 
                                                               std::vector<double> b, 
                                                               std::vector<double> a){
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
    auto [b, a] = zpk2tf(z, p, k, b, a);

    return std::make_tuple(b, a);

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

    auto z_prod = std::accumulate(z.begin(), z.end(), 1.0, std::multiplies<double>());
    auto p_prod = std::accumulate(p.begin(), p.end(), 1.0, std::multiplies<std::complex<double>>());

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
    p : vector<complex>
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

std::tuple<std::vector<double>, std::vector<std::complex<double>>, double> lp2hp_zpk(std::vector<double> z, 
                                                                                     std::vector<std::complex<double>> p, 
                                                                                     double k, 
                                                                                     double wo) {
    /*
    Transform a lowpass filter prototype to a highpass filter.
    Return an analog high-pass filter with cutoff frequency `wo`
    from an analog low-pass filter prototype with unity cutoff frequency,
    using zeros, poles, and gain ('zpk') representation.
    Parameters
    ----------
    z : vector<double>
        Zeros of the analog filter transfer function.
    p : vector<complex>
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired cutoff, as angular frequency (e.g., rad/s).
        Defaults to no change.
    Returns
    -------
    z : vector
        Zeros of the transformed high-pass filter transfer function.
    p : vector
        Poles of the transformed high-pass filter transfer function.
    k : double
        System gain of the transformed high-pass filter.
    */

    std::vector<double> z_hp;               //Zeros
    std::vector<std::complex<double>> p_hp; //poles
    double k_hp;                            //Gain

    int degree = _relative_degree(z, p);

    // Invert positions radially about unit circle to convert LPF to HPF
    // Scale all points radially from origin to shift cutoff frequency
    for (int i = 0; i < z.size(); i++){
        z_hp[i] = wo / z[i];
    }
    for (int i = 0; i < p.size(); i++){
        p_hp[i] = wo / p[i];
    }

    // If lowpass had zeros at infinity, inverting moves them to origin.
    for (int i = 0; i < degree; i++){
        z_hp.push_back(0.0);
    }

    // Cancel out gain change caused by inversion
    auto z_prod = std::accumulate(z.begin(), z.end(), -1.0, std::multiplies<double>());
    auto p_prod = std::accumulate(p.begin(), p.end(), -1.0, std::multiplies<std::complex<double>>());

    k_hp = k * std::real( z_prod / p_prod);

    return std::make_tuple(z_hp, p_hp, k_hp);
}

std::tuple<std::vector<std::complex<double>>, std::vector<std::complex<double>>, double> lp2bp_zpk(std::vector<double> z, 
                                                                                                   std::vector<std::complex<double>> p, 
                                                                                                   double k, 
                                                                                                   double wo, 
                                                                                                   double bw) {
    /*
    Transform a lowpass filter prototype to a bandpass filter.
    Return an analog band-pass filter with center frequency `wo` and
    bandwidth `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, using zeros, poles, and gain ('zpk') representation.
    Parameters
    ----------
    z : vector<double>
        Zeros of the analog filter transfer function.
    p : vector<complex>
        Poles of the analog filter transfer function.
    k : double
        System gain of the analog filter transfer function.
    wo : double
        Desired passband center, as angular frequency (e.g., rad/s).
        Defaults to no change.
    bw : double
        Desired passband width, as angular frequency (e.g., rad/s).
        Defaults to 1.
    Returns
    -------
    z : vector
        Zeros of the transformed band-pass filter transfer function.
    p : vector
        Poles of the transformed band-pass filter transfer function.
    k : double
        System gain of the transformed band-pass filter.
    */

    std::vector<std::complex<double>> z_lp;      //Zeros
    std::vector<std::complex<double>> z_bp;
    std::vector<std::complex<double>> z_aux_neg;
    std::vector<std::complex<double>> z_aux_pos;
    std::vector<std::complex<double>> p_lp;      //poles
    std::vector<std::complex<double>> p_bp;
    std::vector<std::complex<double>> p_aux_neg;
    std::vector<std::complex<double>> p_aux_pos;
    double k_bp;                                 //Gain

    double degree = _relative_degree(z, p);

    // Scale poles and zeros to desired bandwidth
    for (auto val : z) {
        z_lp.push_back(val * bw/2);
    }
    for (auto val : p) {
        p_lp.push_back(val * (bw/2));
    }

    // Duplicate poles and zeros and shift from baseband to +wo and -wo
    for (int i = 0; i < z_lp.size(); i++){
        z_aux_pos.push_back(z_lp[i] + std::sqrt(std::pow(z_lp[i], 2) - std::pow(wo, 2)));
        z_aux_neg.push_back(z_lp[i] - std::sqrt(std::pow(z_lp[i], 2) - std::pow(wo, 2)));
    }
    for (int i = 0; i < p_lp.size(); i++){
        p_aux_pos.push_back(p_lp[i] + std::sqrt(std::pow(p_lp[i], 2) - std::pow(wo, 2)));
        p_aux_neg.push_back(p_lp[i] - std::sqrt(std::pow(p_lp[i], 2) - std::pow(wo, 2)));
    }

    std::merge(z_aux_pos.begin(), z_aux_pos.end(), z_aux_neg.begin(), z_aux_neg.end(), std::back_inserter(z_bp));
    std::merge(p_aux_pos.begin(), p_aux_pos.end(), p_aux_neg.begin(), p_aux_neg.end(), std::back_inserter(p_bp));

    // Move degree zeros to origin, leaving degree zeros at infinity for BPF
    for (int i = 0; i < degree; i++){
        z_bp.push_back(0.0);
    }

    // Cancel out gain change from frequency scaling
    k_bp = k * std::pow(bw, degree);

    return std::make_tuple(z_bp, p_bp, k_bp);
}

std::tuple<std::vector<std::complex<double>>, std::vector<std::complex<double>>, double> lp2bs_zpk(std::vector<double> z, 
                                                                                                   std::vector<std::complex<double>> p, 
                                                                                                   double k, 
                                                                                                   double wo, 
                                                                                                   double bw){
    /*
    Transform a lowpass filter prototype to a bandstop filter.
    Return an analog band-stop filter with center frequency `wo` and
    stopband width `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, using zeros, poles, and gain ('zpk') representation.
    Parameters
    ----------
    z : vector
        Zeros of the analog filter transfer function.
    p : vector
        Poles of the analog filter transfer function.
    k : double
        System gain of the analog filter transfer function.
    wo : double
        Desired stopband center, as angular frequency (e.g., rad/s).
        Defaults to no change.
    bw : double
        Desired stopband width, as angular frequency (e.g., rad/s).
        Defaults to 1.
    Returns
    -------
    z : vector
        Zeros of the transformed band-stop filter transfer function.
    p : vector
        Poles of the transformed band-stop filter transfer function.
    k : double
        System gain of the transformed band-stop filter.
    */

    std::vector<std::complex<double>> z_hp;      //Zeros
    std::vector<std::complex<double>> z_bs;
    std::vector<std::complex<double>> z_aux_neg;
    std::vector<std::complex<double>> z_aux_pos;
    std::vector<std::complex<double>> p_hp;      //poles
    std::vector<std::complex<double>> p_bs;
    std::vector<std::complex<double>> p_aux_neg;
    std::vector<std::complex<double>> p_aux_pos;
    double k_bs;                                 //Gain

    double degree = _relative_degree(z, p);

    // Invert to a highpass filter with desired bandwidth
    for (auto val : z) {
        z_hp.push_back((bw/2) / val);
    }
    for (auto val : p) {
        p_hp.push_back((bw/2) / val);
    }

    // Duplicate poles and zeros and shift from baseband to +wo and -wo
    for (int i = 0; i < z_hp.size(); i++){
        z_aux_pos.push_back(z_hp[i] + std::sqrt(std::pow(z_hp[i], 2) - std::pow(wo, 2)));
        z_aux_neg.push_back(z_hp[i] - std::sqrt(std::pow(z_hp[i], 2) - std::pow(wo, 2)));
    }
    for (int i = 0; i < p_hp.size(); i++){
        p_aux_pos.push_back(p_hp[i] + std::sqrt(std::pow(p_hp[i], 2) - std::pow(wo, 2)));
        p_aux_neg.push_back(p_hp[i] - std::sqrt(std::pow(p_hp[i], 2) - std::pow(wo, 2)));
    }

    std::merge(z_aux_pos.begin(), z_aux_pos.end(), z_aux_neg.begin(), z_aux_neg.end(), std::back_inserter(z_bs));
    std::merge(p_aux_pos.begin(), p_aux_pos.end(), p_aux_neg.begin(), p_aux_neg.end(), std::back_inserter(p_bs));

    // Move any zeros that were at infinity to the center of the stopband
    for (int i = 0; i < degree; i++){
        z_bs.push_back(1i*wo);
    }
    for (int i = 0; i < degree; i++){
        z_bs.push_back(-1i*wo);
    }

    // Cancel out gain change caused by inversion
    auto z_prod = std::accumulate(z.begin(), z.end(), -1.0, std::multiplies<double>());
    auto p_prod = std::accumulate(p.begin(), p.end(), -1.0, std::multiplies<std::complex<double>>());

    k_bs = k * std::real(z_prod / p_prod);

    return std::make_tuple(z_bs, p_bs, k_bs);
}

std::tuple<std::vector<double>, std::vector<double>> zpk2tf(std::vector<std::complex<double>> z, 
                                                            std::vector<std::complex<double>> p, 
                                                            double k, 
                                                            std::vector<double> b, 
                                                            std::vector<double> a){
    /*
    Return polynomial transfer function representation from zeros and poles
    Parameters
    ----------
    z : vector
        Zeros of the transfer function.
    p : vector
        Poles of the transfer function.
    k : double
        System gain.
    b : double pointer
    a : double pointer
    Returns
    -------
    b : double pointer
        Numerator polynomial coefficients.
    a : double pointer
        Denominator polynomial coefficients.
    */
    
   std::vector<double> poly_z;

    poly_z = poly(z);
    a = poly(p);

    for(int i=0; i<poly_z.size(); i++){
        b.push_back(k*poly_z[i]);
    }

    if (a.size() != b.size()){
        throw std::out_of_range("The total numbers of coefficients b and a do not match");
    }

    return std::make_tuple(b, a);
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

std::vector<double> poly(std::vector<std::complex<double>> seq_of_zeros){
    /*
    Find the coefficients of a polynomial with the given sequence of roots.

    Returns the coefficients of the polynomial whose leading coefficient
    is one for the given sequence of zeros (multiple roots must be included
    in the sequence as many times as their multiplicity; see Examples).
    A square matrix (or array, which will be treated as a matrix) can also
    be given, in which case the coefficients of the characteristic polynomial
    of the matrix are returned.

    Parameters
    ----------
    seq_of_zeros : vector
        A sequence of polynomial roots

    Returns
    -------
    c : vector
        1D array of polynomial coefficients from highest to lowest degree:

        ``c[0] * x**(N) + c[1] * x**(N-1) + ... + c[N-1] * x + c[N]``
        where c[0] always equals 1.

    Notes
    -----
    Specifying the roots of a polynomial still leaves one degree of
    freedom, typically represented by an undetermined leading
    coefficient. [1]_ In the case of this function, that coefficient -
    the first one in the returned array - is always taken as one. (If
    for some reason you have one other point, the only automatic way
    presently to leverage that information is to use ``polyfit``.)

    The characteristic polynomial, :math:`p_a(t)`, of an `n`-by-`n`
    matrix **A** is given by

        :math:`p_a(t) = \\mathrm{det}(t\\, \\mathbf{I} - \\mathbf{A})`,

    where **I** is the `n`-by-`n` identity matrix. [2]_

    References
    ----------
    .. [1] M. Sullivan and M. Sullivan, III, "Algebra and Trignometry,
       Enhanced With Graphing Utilities," Prentice-Hall, pg. 318, 1996.

    .. [2] G. Strang, "Linear Algebra and Its Applications, 2nd Edition,"
       Academic Press, pg. 182, 1980.

    */

   std::vector<std::complex<double>> a{1.0};
   std::vector<std::complex<double>> conv_result;
   std::vector<double> output;

    int sh = seq_of_zeros.size();

    //dt = seq_of_zeros.dtype

    for(auto zero : seq_of_zeros){
        
        std::vector<std::complex<double>> aux{1.0, -zero};
        
        a = convolve(a, aux); //Revisar si puedo hacer esta asignación!
    }

    // if complex roots are all complex conjugates, the roots are real.
    for(auto val : a){
        output.push_back(std::real(val));
    }

    return output;

}

std::vector<std::complex<double>> convolve(std::vector<std::complex<double>> h, std::vector<std::complex<double>> x) {

    std::vector<std::complex<double>> y; //Final signal

    int conv_len = h.size() + x.size() - 1; //Len of the convolution

    //Primero invierto una de las señales:
    std::vector<std::complex<double>> c_h;

    int count = h.size()-1;
    for(int i=0; i<h.size(); i++){
        c_h.push_back(h[count]);
        count--;
    }

    //zero padding a la otra señal: Agrego h.size()-1 ceros al principio y al final
    std::vector<std::complex<double>> c_x;

    c_x = x; //Copio la señal

    for(int i=0; i<h.size()-1; i++){
        c_x.push_back(0);
        c_x.insert(begin(c_x), 0);
    }

    //Realizo la convolución:
    for(int i=0; i<conv_len; i++){
        std::complex<double> aux = 0;
        for(int j=0; j<h.size(); j++){
            aux += c_h[j] * c_x[i+j]; 
        }

        y.push_back(aux);
    }

    return y;

}