#include <iostream>
#include <vector>
#include <algorithm>
#include <tuple>
#include <complex>

using namespace std;
using namespace std::complex_literals;

int main()
{
    //DATA
    tuple<vector<complex<double>>, vector<complex<double>>, double> tupla;
    vector<complex<double>> v1{4,2i,8,6i};
    vector<complex<double>> v2{14,12i,18,16i};
    vector<complex<double>> v3;
    vector<complex<double>> v4;
    double v5;

    
    tupla = make_tuple(v1, v2, 3.0);
    
    tie(v3, v4, v5) = tupla;
    
    //MERGE
    //vector<int> dst;
    //merge(v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(dst));

    //PRINT

    cout<< "V3 size: " << v3.size() << endl;
    cout<< "V4 size: " << v4.size() << endl;

    for(auto item : v3){
        cout<<item.real()<<" ";
    }
    cout << endl;

    for(auto item : v4){
        cout<<item.imag()<<" ";
    }
    cout << endl;

    cout<< v5 << endl;
    

    cout << "Finalizado!";
    return 0;
}