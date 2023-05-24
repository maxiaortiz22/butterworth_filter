import numpy as np



def iirfilter(N, Wn, btype='band', analog=False,
              ftype='butter', output='ba', fs=None):
    """
    IIR digital and analog filter design given order and critical points.
    Design an Nth-order digital or analog filter and return the filter
    coefficients.
    Parameters
    ----------
    N : int
        The order of the filter.
    Wn : array_like
        A scalar or length-2 sequence giving the critical frequencies.
        For digital filters, `Wn` are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`Wn` is thus in
        half-cycles / sample.)
        For analog filters, `Wn` is an angular frequency (e.g., rad/s).
        When Wn is a length-2 sequence, ``Wn[0]`` must be less than ``Wn[1]``.
    rp : float, optional
        For Chebyshev and elliptic filters, provides the maximum ripple
        in the passband. (dB)
    rs : float, optional
        For Chebyshev and elliptic filters, provides the minimum attenuation
        in the stop band. (dB)
    btype : {'bandpass', 'lowpass', 'highpass', 'bandstop'}, optional
        The type of filter.  Default is 'bandpass'.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    ftype : str, optional
        The type of IIR filter to design:
            - Butterworth   : 'butter'
            - Chebyshev I   : 'cheby1'
            - Chebyshev II  : 'cheby2'
            - Cauer/elliptic: 'ellip'
            - Bessel/Thomson: 'bessel'
    output : {'ba', 'zpk', 'sos'}, optional
        Filter form of the output:
            - second-order sections (recommended): 'sos'
            - numerator/denominator (default)    : 'ba'
            - pole-zero                          : 'zpk'
        In general the second-order sections ('sos') form  is
        recommended because inferring the coefficients for the
        numerator/denominator form ('ba') suffers from numerical
        instabilities. For reasons of backward compatibility the default
        form is the numerator/denominator form ('ba'), where the 'b'
        and the 'a' in 'ba' refer to the commonly used names of the
        coefficients used.
        Note: Using the second-order sections form ('sos') is sometimes
        associated with additional computational costs: for
        data-intense use cases it is therefore recommended to also
        investigate the numerator/denominator form ('ba').
    fs : float, optional
        The sampling frequency of the digital system.
        .. versionadded:: 1.2.0
    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output='sos'``.
    See Also
    --------
    butter : Filter design using order and critical points
    cheby1, cheby2, ellip, bessel
    buttord : Find order and critical points from passband and stopband spec
    cheb1ord, cheb2ord, ellipord
    iirdesign : General filter design using passband and stopband spec
    Notes
    -----
    The ``'sos'`` output parameter was added in 0.16.0.
    Examples
    --------
    Generate a 17th-order Chebyshev II analog bandpass filter from 50 Hz to
    200 Hz and plot the frequency response:
    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> b, a = signal.iirfilter(17, [2*np.pi*50, 2*np.pi*200], rs=60,
    ...                         btype='band', analog=True, ftype='cheby2')
    >>> w, h = signal.freqs(b, a, 1000)
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(1, 1, 1)
    >>> ax.semilogx(w / (2*np.pi), 20 * np.log10(np.maximum(abs(h), 1e-5)))
    >>> ax.set_title('Chebyshev Type II bandpass frequency response')
    >>> ax.set_xlabel('Frequency [Hz]')
    >>> ax.set_ylabel('Amplitude [dB]')
    >>> ax.axis((10, 1000, -100, 10))
    >>> ax.grid(which='both', axis='both')
    >>> plt.show()
    Create a digital filter with the same properties, in a system with
    sampling rate of 2000 Hz, and plot the frequency response. (Second-order
    sections implementation is required to ensure stability of a filter of
    this order):
    >>> sos = signal.iirfilter(17, [50, 200], rs=60, btype='band',
    ...                        analog=False, ftype='cheby2', fs=2000,
    ...                        output='sos')
    >>> w, h = signal.sosfreqz(sos, 2000, fs=2000)
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(1, 1, 1)
    >>> ax.semilogx(w, 20 * np.log10(np.maximum(abs(h), 1e-5)))
    >>> ax.set_title('Chebyshev Type II bandpass frequency response')
    >>> ax.set_xlabel('Frequency [Hz]')
    >>> ax.set_ylabel('Amplitude [dB]')
    >>> ax.axis((10, 1000, -100, 10))
    >>> ax.grid(which='both', axis='both')
    >>> plt.show()
    """
    ftype, btype, output = [x.lower() for x in (ftype, btype, output)]
    Wn = np.asarray(Wn)
    if fs is not None:
        if analog:
            raise ValueError("fs cannot be specified for an analog filter")
        Wn = 2*Wn/fs

    if np.any(Wn <= 0):
        raise ValueError("filter critical frequencies must be greater than 0")

    if Wn.size > 1 and not Wn[0] < Wn[1]:
        raise ValueError("Wn[0] must be less than Wn[1]")

    try:
        btype = band_dict[btype]
    except KeyError as e:
        raise ValueError("'%s' is an invalid bandtype for filter." % btype) from e

    try:
        typefunc = filter_dict[ftype][0]
    except KeyError as e:
        raise ValueError("'%s' is not a valid basic IIR filter." % ftype) from e

    if output not in ['ba', 'zpk', 'sos']:
        raise ValueError("'%s' is not a valid output form." % output)

    # Get analog lowpass prototype
    if typefunc == buttap:
        z, p, k = typefunc(N)
    else:
        raise NotImplementedError("'%s' not implemented in iirfilter." % ftype)

    # Pre-warp frequencies for digital filter design
    if not analog:
        if np.any(Wn <= 0) or np.any(Wn >= 1):
            if fs is not None:
                raise ValueError("Digital filter critical frequencies must "
                                 f"be 0 < Wn < fs/2 (fs={fs} -> fs/2={fs/2})")
            raise ValueError("Digital filter critical frequencies "
                             "must be 0 < Wn < 1")
        fs = 2.0
        warped = 2 * fs * np.tan(np.pi * Wn / fs)
    else:
        warped = Wn

    # transform to lowpass, bandpass, highpass, or bandstop
    if btype in ('lowpass', 'highpass'):
        if np.size(Wn) != 1:
            raise ValueError('Must specify a single critical frequency Wn '
                             'for lowpass or highpass filter')

        if btype == 'lowpass':
            z, p, k = lp2lp_zpk(z, p, k, wo=warped)
        elif btype == 'highpass':
            z, p, k = lp2hp_zpk(z, p, k, wo=warped)
    elif btype in ('bandpass', 'bandstop'):
        try:
            bw = warped[1] - warped[0]
            wo = np.sqrt(warped[0] * warped[1])
        except IndexError as e:
            raise ValueError('Wn must specify start and stop frequencies for '
                             'bandpass or bandstop filter') from e

        if btype == 'bandpass':
            z, p, k = lp2bp_zpk(z, p, k, wo=wo, bw=bw)
        elif btype == 'bandstop':
            z, p, k = lp2bs_zpk(z, p, k, wo=wo, bw=bw)
    else:
        raise NotImplementedError("'%s' not implemented in iirfilter." % btype)

    # Find discrete equivalent if necessary
    if not analog:
        z, p, k = bilinear_zpk(z, p, k, fs=fs)

    # Transform to proper out type (pole-zero, state-space, numer-denom)
    if output == 'zpk':
        return z, p, k
    elif output == 'ba':
        return zpk2tf(z, p, k)
    

def butter(N, Wn, btype='low', analog=False, output='ba', fs=None):
    """
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
    Wn : array_like
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
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        The type of filter.  Default is 'lowpass'.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    output : {'ba', 'zpk', 'sos'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        second-order sections ('sos'). Default is 'ba' for backwards
        compatibility, but 'sos' should be used for general-purpose filtering.
    fs : float, optional
        The sampling frequency of the digital system.
        .. versionadded:: 1.2.0
    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output='sos'``.
    See Also
    --------
    buttord, buttap
    Notes
    -----
    The Butterworth filter has maximally flat frequency response in the
    passband.
    The ``'sos'`` output parameter was added in 0.16.0.
    If the transfer function form ``[b, a]`` is requested, numerical
    problems can occur since the conversion between roots and
    the polynomial coefficients is a numerically sensitive operation,
    even for N >= 4. It is recommended to work with the SOS
    representation.
    Examples
    --------
    Design an analog filter and plot its frequency response, showing the
    critical points:
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> b, a = signal.butter(4, 100, 'low', analog=True)
    >>> w, h = signal.freqs(b, a)
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.title('Butterworth filter frequency response')
    >>> plt.xlabel('Frequency [radians / second]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.margins(0, 0.1)
    >>> plt.grid(which='both', axis='both')
    >>> plt.axvline(100, color='green') # cutoff frequency
    >>> plt.show()
    Generate a signal made up of 10 Hz and 20 Hz, sampled at 1 kHz
    >>> t = np.linspace(0, 1, 1000, False)  # 1 second
    >>> sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)
    >>> fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    >>> ax1.plot(t, sig)
    >>> ax1.set_title('10 Hz and 20 Hz sinusoids')
    >>> ax1.axis([0, 1, -2, 2])
    Design a digital high-pass filter at 15 Hz to remove the 10 Hz tone, and
    apply it to the signal. (It's recommended to use second-order sections
    format when filtering, to avoid numerical error with transfer function
    (``ba``) format):
    >>> sos = signal.butter(10, 15, 'hp', fs=1000, output='sos')
    >>> filtered = signal.sosfilt(sos, sig)
    >>> ax2.plot(t, filtered)
    >>> ax2.set_title('After 15 Hz high-pass filter')
    >>> ax2.axis([0, 1, -2, 2])
    >>> ax2.set_xlabel('Time [seconds]')
    >>> plt.tight_layout()
    >>> plt.show()
    """
    return iirfilter(N, Wn, btype=btype, analog=analog,
                     output=output, ftype='butter', fs=fs)

def buttap(N):
    """Return (z,p,k) for analog prototype of Nth-order Butterworth filter.
    The filter will have an angular (e.g., rad/s) cutoff frequency of 1.
    See Also
    --------
    butter : Filter design function using this prototype
    """
    print('\nbuttap:')
    if abs(int(N)) != N:
        raise ValueError("Filter order must be a nonnegative integer")
    z = np.array([])
    m = np.arange(-N+1, N, 2)
    # Middle value is 0 to ensure an exactly real pole
    p = -np.exp(1j * np.pi * m / (2 * N))
    k = 1
    print(f'z: {z}')
    print(f'p: {p}')
    print(f'k: {k}')
    return z, p, k


def bilinear_zpk(z, p, k, fs):
    r"""
    Return a digital IIR filter from an analog one using a bilinear transform.
    Transform a set of poles and zeros from the analog s-plane to the digital
    z-plane using Tustin's method, which substitutes ``(z-1) / (z+1)`` for
    ``s``, maintaining the shape of the frequency response.
    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    fs : float
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
    See Also
    --------
    lp2lp_zpk, lp2hp_zpk, lp2bp_zpk, lp2bs_zpk
    bilinear
    Notes
    -----
    .. versionadded:: 1.1.0
    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> fs = 100
    >>> bf = 2 * np.pi * np.array([7, 13])
    >>> filts = signal.lti(*signal.butter(4, bf, btype='bandpass', analog=True,
    ...                                   output='zpk'))
    >>> filtz = signal.lti(*signal.bilinear_zpk(filts.zeros, filts.poles,
    ...                                         filts.gain, fs))
    >>> wz, hz = signal.freqz_zpk(filtz.zeros, filtz.poles, filtz.gain)
    >>> ws, hs = signal.freqs_zpk(filts.zeros, filts.poles, filts.gain,
    ...                           worN=fs*wz)
    >>> plt.semilogx(wz*fs/(2*np.pi), 20*np.log10(np.abs(hz).clip(1e-15)),
    ...              label=r'$|H_z(e^{j \omega})|$')
    >>> plt.semilogx(wz*fs/(2*np.pi), 20*np.log10(np.abs(hs).clip(1e-15)),
    ...              label=r'$|H(j \omega)|$')
    >>> plt.legend()
    >>> plt.xlabel('Frequency [Hz]')
    >>> plt.ylabel('Magnitude [dB]')
    >>> plt.grid(True)
    """

    print('\nbilinear_zpk:')
    z = np.atleast_1d(z)
    p = np.atleast_1d(p)

    degree = _relative_degree(z, p)

    fs2 = 2.0*fs

    # Bilinear transform the poles and zeros
    z_z = (fs2 + z) / (fs2 - z)
    p_z = (fs2 + p) / (fs2 - p)

    #print(f'z_z: {len(z_z)}')
    #print(z_z)
    #print(f'p: {len(p_z)}')
    #print(p_z)

    # Any zeros that were at infinity get moved to the Nyquist frequency
    z_z = np.append(z_z, -np.ones(degree))

    # Compensate for gain change
    k_z = k * np.real(np.prod(fs2 - z) / np.prod(fs2 - p))

    #print(f'z: {len(z_z)}')
    #print(z_z)

    #print(f'z: {len(z_hp)}')
    #print(z_hp)
    #print(f'p: {len(p_hp)}')
    #print(p_hp)
    print(f'z: {z_z}')
    print(f'p: {p_z}')
    print(f'k: {k_z}')

    return z_z, p_z, k_z

def lp2lp_zpk(z, p, k, wo=1.0):
    r"""
    Transform a lowpass filter prototype to a different frequency.
    Return an analog low-pass filter with cutoff frequency `wo`
    from an analog low-pass filter prototype with unity cutoff frequency,
    using zeros, poles, and gain ('zpk') representation.
    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired cutoff, as angular frequency (e.g., rad/s).
        Defaults to no change.
    Returns
    -------
    z : ndarray
        Zeros of the transformed low-pass filter transfer function.
    p : ndarray
        Poles of the transformed low-pass filter transfer function.
    k : float
        System gain of the transformed low-pass filter.
    See Also
    --------
    lp2hp_zpk, lp2bp_zpk, lp2bs_zpk, bilinear
    lp2lp
    Notes
    -----
    This is derived from the s-plane substitution
    .. math:: s \rightarrow \frac{s}{\omega_0}
    .. versionadded:: 1.1.0
    """

    print('\nlp2lp_zpk:')
    z = np.atleast_1d(z)
    p = np.atleast_1d(p)
    wo = float(wo)  # Avoid int wraparound

    degree = _relative_degree(z, p)

    # Scale all points radially from origin to shift cutoff frequency
    z_lp = wo * z
    p_lp = wo * p

    # Each shifted pole decreases gain by wo, each shifted zero increases it.
    # Cancel out the net change to keep overall gain the same
    k_lp = k * wo**degree

    print(f'z: {z_lp}')
    print(f'p: {p_lp}')
    print(f'k: {k_lp}')

    return z_lp, p_lp, k_lp

def lp2hp_zpk(z, p, k, wo=1.0):
    r"""
    Transform a lowpass filter prototype to a highpass filter.
    Return an analog high-pass filter with cutoff frequency `wo`
    from an analog low-pass filter prototype with unity cutoff frequency,
    using zeros, poles, and gain ('zpk') representation.
    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired cutoff, as angular frequency (e.g., rad/s).
        Defaults to no change.
    Returns
    -------
    z : ndarray
        Zeros of the transformed high-pass filter transfer function.
    p : ndarray
        Poles of the transformed high-pass filter transfer function.
    k : float
        System gain of the transformed high-pass filter.
    See Also
    --------
    lp2lp_zpk, lp2bp_zpk, lp2bs_zpk, bilinear
    lp2hp
    Notes
    -----
    This is derived from the s-plane substitution
    .. math:: s \rightarrow \frac{\omega_0}{s}
    This maintains symmetry of the lowpass and highpass responses on a
    logarithmic scale.
    .. versionadded:: 1.1.0
    """
    print('\nlp2hp_zpk:')
    z = np.atleast_1d(z)
    p = np.atleast_1d(p)
    wo = float(wo)

    degree = _relative_degree(z, p)

    # Invert positions radially about unit circle to convert LPF to HPF
    # Scale all points radially from origin to shift cutoff frequency
    z_hp = wo / z
    p_hp = wo / p

    # If lowpass had zeros at infinity, inverting moves them to origin.
    z_hp = np.append(z_hp, np.zeros(degree))

    # Cancel out gain change caused by inversion
    k_hp = k * np.real(np.prod(-z) / np.prod(-p))

    #print(f'z: {len(z_hp)}')
    #print(z_hp)
    #print(f'p: {len(p_hp)}')
    #print(p_hp)

    print(f'z: {z_hp}')
    print(f'p: {p_hp}')
    print(f'k: {k_hp}')


    return z_hp, p_hp, k_hp

def lp2bp_zpk(z, p, k, wo=1.0, bw=1.0):
    r"""
    Transform a lowpass filter prototype to a bandpass filter.
    Return an analog band-pass filter with center frequency `wo` and
    bandwidth `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, using zeros, poles, and gain ('zpk') representation.
    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired passband center, as angular frequency (e.g., rad/s).
        Defaults to no change.
    bw : float
        Desired passband width, as angular frequency (e.g., rad/s).
        Defaults to 1.
    Returns
    -------
    z : ndarray
        Zeros of the transformed band-pass filter transfer function.
    p : ndarray
        Poles of the transformed band-pass filter transfer function.
    k : float
        System gain of the transformed band-pass filter.
    See Also
    --------
    lp2lp_zpk, lp2hp_zpk, lp2bs_zpk, bilinear
    lp2bp
    Notes
    -----
    This is derived from the s-plane substitution
    .. math:: s \rightarrow \frac{s^2 + {\omega_0}^2}{s \cdot \mathrm{BW}}
    This is the "wideband" transformation, producing a passband with
    geometric (log frequency) symmetry about `wo`.
    .. versionadded:: 1.1.0
    """
    print('\nlp2bp_zpk:')
    z = np.atleast_1d(z)
    p = np.atleast_1d(p)
    wo = float(wo)
    bw = float(bw)

    degree = _relative_degree(z, p)

    # Scale poles and zeros to desired bandwidth
    z_lp = z * bw/2
    p_lp = p * bw/2

    # Square root needs to produce complex result, not NaN
    z_lp = z_lp.astype(complex)
    p_lp = p_lp.astype(complex)

    # Duplicate poles and zeros and shift from baseband to +wo and -wo
    z_bp = np.concatenate((z_lp + np.sqrt(z_lp**2 - wo**2),
                        z_lp - np.sqrt(z_lp**2 - wo**2)))
    p_bp = np.concatenate((p_lp + np.sqrt(p_lp**2 - wo**2),
                        p_lp - np.sqrt(p_lp**2 - wo**2)))

    # Move degree zeros to origin, leaving degree zeros at infinity for BPF
    z_bp = np.append(z_bp, np.zeros(degree))

    # Cancel out gain change from frequency scaling
    k_bp = k * bw**degree

    print(f'z: {z_bp}')
    print(f'p: {p_bp}')
    print(f'k: {k_bp}')

    return z_bp, p_bp, k_bp

def lp2bs_zpk(z, p, k, wo=1.0, bw=1.0):
    r"""
    Transform a lowpass filter prototype to a bandstop filter.
    Return an analog band-stop filter with center frequency `wo` and
    stopband width `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, using zeros, poles, and gain ('zpk') representation.
    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired stopband center, as angular frequency (e.g., rad/s).
        Defaults to no change.
    bw : float
        Desired stopband width, as angular frequency (e.g., rad/s).
        Defaults to 1.
    Returns
    -------
    z : ndarray
        Zeros of the transformed band-stop filter transfer function.
    p : ndarray
        Poles of the transformed band-stop filter transfer function.
    k : float
        System gain of the transformed band-stop filter.
    See Also
    --------
    lp2lp_zpk, lp2hp_zpk, lp2bp_zpk, bilinear
    lp2bs
    Notes
    -----
    This is derived from the s-plane substitution
    .. math:: s \rightarrow \frac{s \cdot \mathrm{BW}}{s^2 + {\omega_0}^2}
    This is the "wideband" transformation, producing a stopband with
    geometric (log frequency) symmetry about `wo`.
    .. versionadded:: 1.1.0
    """
    print('\nlp2bs_zpk:')
    z = np.atleast_1d(z)
    p = np.atleast_1d(p)
    wo = float(wo)
    bw = float(bw)

    degree = _relative_degree(z, p)

    # Invert to a highpass filter with desired bandwidth
    z_hp = (bw/2) / z
    p_hp = (bw/2) / p

    # Square root needs to produce complex result, not NaN
    z_hp = z_hp.astype(complex)
    p_hp = p_hp.astype(complex)

    # Duplicate poles and zeros and shift from baseband to +wo and -wo
    z_bs = np.concatenate((z_hp + np.sqrt(z_hp**2 - wo**2),
                        z_hp - np.sqrt(z_hp**2 - wo**2)))
    p_bs = np.concatenate((p_hp + np.sqrt(p_hp**2 - wo**2),
                        p_hp - np.sqrt(p_hp**2 - wo**2)))

    # Move any zeros that were at infinity to the center of the stopband
    z_bs = np.append(z_bs, np.full(degree, +1j*wo))
    z_bs = np.append(z_bs, np.full(degree, -1j*wo))

    # Cancel out gain change caused by inversion
    k_bs = k * np.real(np.prod(-z) / np.prod(-p))

    print(f'z: {z_bs}')
    print(f'p: {p_bs}')
    print(f'k: {k_bs}')

    return z_bs, p_bs, k_bs

def zpk2tf(z, p, k):
    """
    Return polynomial transfer function representation from zeros and poles
    Parameters
    ----------
    z : array_like
        Zeros of the transfer function.
    p : array_like
        Poles of the transfer function.
    k : float
        System gain.
    Returns
    -------
    b : ndarray
        Numerator polynomial coefficients.
    a : ndarray
        Denominator polynomial coefficients.
    """
    print('\nlp2bs_zpk:')
    z = np.atleast_1d(z)
    k = np.atleast_1d(k)

    print(f'z: {z}')
    print(f'p: {p}')
    print(f'k: {k}')


    if len(z.shape) > 1:
        temp = poly(z[0])
        b = np.empty((z.shape[0], z.shape[1] + 1), temp.dtype.char)
        if len(k) == 1:
            k = [k[0]] * z.shape[0]
        for i in range(z.shape[0]):
            b[i] = k[i] * poly(z[i])
    else:
        b = k * poly(z)
    a = np.atleast_1d(poly(p))
    
    # Use real output if possible. Copied from numpy.poly, since
    # we can't depend on a specific version of numpy.
    if issubclass(b.dtype.type, np.complexfloating):
        print('pasé por b')
        # if complex roots are all complex conjugates, the roots are real.
        roots = np.asarray(z, complex)
        pos_roots = np.compress(roots.imag > 0, roots)
        neg_roots = np.conjugate(np.compress(roots.imag < 0, roots))
        if len(pos_roots) == len(neg_roots):
            if np.all(np.sort_complex(neg_roots) ==
                      np.sort_complex(pos_roots)):
                b = b.real.copy()

    if issubclass(a.dtype.type, np.complexfloating):
        print('pasé por a')
        # if complex roots are all complex conjugates, the roots are real.
        roots = np.asarray(p, complex)
        pos_roots = np.compress(roots.imag > 0, roots)
        neg_roots = np.conjugate(np.compress(roots.imag < 0, roots))
        if len(pos_roots) == len(neg_roots):
            if np.all(np.sort_complex(neg_roots) ==
                      np.sort_complex(pos_roots)):
                a = a.real.copy()

    print(f'b: {b}')
    print(f'a: {a}')

    return b, a

def band_stop_obj(wp, ind, passb, stopb, gpass, gstop, type):
    """
    Band Stop Objective Function for order minimization.
    Returns the non-integer order for an analog band stop filter.
    Parameters
    ----------
    wp : scalar
        Edge of passband `passb`.
    ind : int, {0, 1}
        Index specifying which `passb` edge to vary (0 or 1).
    passb : ndarray
        Two element sequence of fixed passband edges.
    stopb : ndarray
        Two element sequence of fixed stopband edges.
    gstop : float
        Amount of attenuation in stopband in dB.
    gpass : float
        Amount of ripple in the passband in dB.
    type : {'butter', 'cheby', 'ellip'}
        Type of filter.
    Returns
    -------
    n : scalar
        Filter order (possibly non-integer).
    """

    _validate_gpass_gstop(gpass, gstop)

    passbC = passb.copy()
    passbC[ind] = wp
    nat = (stopb * (passbC[0] - passbC[1]) /
           (stopb ** 2 - passbC[0] * passbC[1]))
    nat = min(abs(nat))

    if type == 'butter':
        GSTOP = 10 ** (0.1 * abs(gstop))
        GPASS = 10 ** (0.1 * abs(gpass))
        n = (np.log10((GSTOP - 1.0) / (GPASS - 1.0)) / (2 * np.log10(nat)))
    
    else:
        raise ValueError("Incorrect type: %s" % type)
    return n

def _validate_gpass_gstop(gpass, gstop):

    if gpass <= 0.0:
        raise ValueError("gpass should be larger than 0.0")
    elif gstop <= 0.0:
        raise ValueError("gstop should be larger than 0.0")
    elif gpass > gstop:
        raise ValueError("gpass should be smaller than gstop")

def _relative_degree(z, p):
    """
    Return relative degree of transfer function from zeros and poles
    """
    
    degree = len(p) - len(z)
    if degree < 0:
        raise ValueError("Improper transfer function. "
                         "Must have at least as many poles as zeros.")
    else:
        return degree


def poly(seq_of_zeros):
    """
    Find the coefficients of a polynomial with the given sequence of roots.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    Returns the coefficients of the polynomial whose leading coefficient
    is one for the given sequence of zeros (multiple roots must be included
    in the sequence as many times as their multiplicity; see Examples).
    A square matrix (or array, which will be treated as a matrix) can also
    be given, in which case the coefficients of the characteristic polynomial
    of the matrix are returned.

    Parameters
    ----------
    seq_of_zeros : array_like, shape (N,) or (N, N)
        A sequence of polynomial roots, or a square array or matrix object.

    Returns
    -------
    c : ndarray
        1D array of polynomial coefficients from highest to lowest degree:

        ``c[0] * x**(N) + c[1] * x**(N-1) + ... + c[N-1] * x + c[N]``
        where c[0] always equals 1.

    Raises
    ------
    ValueError
        If input is the wrong shape (the input must be a 1-D or square
        2-D array).

    See Also
    --------
    polyval : Compute polynomial values.
    roots : Return the roots of a polynomial.
    polyfit : Least squares polynomial fit.
    poly1d : A one-dimensional polynomial class.

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

    Examples
    --------
    Given a sequence of a polynomial's zeros:

    >>> np.poly((0, 0, 0)) # Multiple root example
    array([1., 0., 0., 0.])

    The line above represents z**3 + 0*z**2 + 0*z + 0.

    >>> np.poly((-1./2, 0, 1./2))
    array([ 1.  ,  0.  , -0.25,  0.  ])

    The line above represents z**3 - z/4

    >>> np.poly((np.random.random(1)[0], 0, np.random.random(1)[0]))
    array([ 1.        , -0.77086955,  0.08618131,  0.        ]) # random

    Given a square array object:

    >>> P = np.array([[0, 1./3], [-1./2, 0]])
    >>> np.poly(P)
    array([1.        , 0.        , 0.16666667])

    Note how in all cases the leading coefficient is always 1.

    """
    seq_of_zeros = np.atleast_1d(seq_of_zeros)
    sh = seq_of_zeros.shape

    if len(sh) == 2 and sh[0] == sh[1] and sh[0] != 0:
        seq_of_zeros = np.linalg.eigvals(seq_of_zeros)
    elif len(sh) == 1:
        dt = seq_of_zeros.dtype
        # Let object arrays slip through, e.g. for arbitrary precision
        if dt != object:
            seq_of_zeros = seq_of_zeros.astype(np.lib.type_check.mintypecode(dt.char))
    else:
        raise ValueError("input must be 1d or non-empty square 2d array.")

    if len(seq_of_zeros) == 0:
        return 1.0
    dt = seq_of_zeros.dtype
    a = np.ones((1,), dtype=dt)
    for zero in seq_of_zeros:
        a = np.convolve(a, np.array([1, -zero], dtype=dt), mode='full')

    print("roots:")
    print(a)

    if issubclass(a.dtype.type, np.complexfloating):
        # if complex roots are all complex conjugates, the roots are real.
        roots = np.asarray(seq_of_zeros, complex)
        if np.all(np.sort(roots) == np.sort(roots.conjugate())):
            print("Son complejas y conjugadas:")
            a = a.real.copy()
            print(a)

    return a

filter_dict = {'butter': [buttap],
               'butterworth': [buttap],
               }


band_dict = {'band': 'bandpass',
             'bandpass': 'bandpass',
             'pass': 'bandpass',
             'bp': 'bandpass',

             'bs': 'bandstop',
             'bandstop': 'bandstop',
             'bands': 'bandstop',
             'stop': 'bandstop',

             'l': 'lowpass',
             'low': 'lowpass',
             'lowpass': 'lowpass',
             'lp': 'lowpass',

             'high': 'highpass',
             'highpass': 'highpass',
             'h': 'highpass',
             'hp': 'highpass',
             }