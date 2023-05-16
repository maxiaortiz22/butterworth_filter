#include <iostream>
#include <vector>
#include <algorithm>

int main()
{
    //DATA
    std::vector<int> v1{4,2,8,6};
    std::vector<int> v2{14,12,18,16};

    std::sort(v1.begin(), v1.end()); 
    std::sort(v2.begin(), v2.end()); 

    //MERGE
    std::vector<int> dst;
    std::merge(v1.begin(), v1.end(), v2.begin(), v2.end(), std::back_inserter(dst));

    //PRINT
    for(auto item : dst)
        std::cout<<item<<" ";

    return 0;
}