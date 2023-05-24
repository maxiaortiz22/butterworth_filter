#include <vector>
#include <string>
#include <tuple>
#include <complex>

std::tuple<std::vector<double>, std::vector<double>> butter(int N, 
                                                            std::vector<double> Wn, 
                                                            std::string btype, 
                                                            bool analog, 
                                                            double fs);

std::tuple<std::vector<double>, std::vector<double>> iirfilter(int N, 
                                                               std::vector<double> Wn, 
                                                               std::string btype, 
                                                               bool analog, 
                                                               double fs);

std::tuple<std::vector<std::complex<double>>, std::vector<std::complex<double>>, double> buttap(int N);

std::tuple<std::vector<std::complex<double>>, std::vector<std::complex<double>>, double> bilinear_zpk(std::vector<std::complex<double>> z, 
                                                                                                      std::vector<std::complex<double>> p, 
                                                                                                      double k, 
                                                                                                      double fs);

std::tuple<std::vector<std::complex<double>>, std::vector<std::complex<double>>, double> lp2lp_zpk(std::vector<std::complex<double>> z, 
                                                                                                   std::vector<std::complex<double>> p, 
                                                                                                   double k, 
                                                                                                   double wo);

std::tuple<std::vector<std::complex<double>>, std::vector<std::complex<double>>, double> lp2hp_zpk(std::vector<std::complex<double>> z, 
                                                                                                   std::vector<std::complex<double>> p, 
                                                                                                   double k, 
                                                                                                   double wo);

std::tuple<std::vector<std::complex<double>>, std::vector<std::complex<double>>, double> lp2bp_zpk(std::vector<std::complex<double>> z, 
                                                                                                   std::vector<std::complex<double>> p, 
                                                                                                   double k, 
                                                                                                   double wo, 
                                                                                                   double bw);

std::tuple<std::vector<std::complex<double>>, std::vector<std::complex<double>>, double> lp2bs_zpk(std::vector<std::complex<double>> z, 
                                                                                                   std::vector<std::complex<double>> p, 
                                                                                                   double k, 
                                                                                                   double wo, 
                                                                                                   double bw);

std::tuple<std::vector<double>, std::vector<double>> zpk2tf(std::vector<std::complex<double>> z, 
                                                            std::vector<std::complex<double>> p, 
                                                            double k);

std::vector<double> arange(double start, double stop, double step = 1);

int _relative_degree(std::vector<std::complex<double>> z, std::vector<std::complex<double>> p);

std::vector<double> poly(std::vector<std::complex<double>> seq_of_zeros);

std::vector<std::complex<double>> convolve(std::vector<std::complex<double>> h, std::vector<std::complex<double>> x);

std::vector<double> filter_signal(std::vector<double> signal, std::vector<double> b, std::vector<double>a);