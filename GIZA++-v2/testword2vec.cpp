#include "Word2Vec.h"

using namespace std;

int main(int argc, char* argv[]){

    Word2Vec w;
    w.readWord2Vec(argv[1], argv[2]);
    cout << argv[1] << " " << argv[2] << endl;
    string s;
    while(cin >> s){
        vector<string> wordVec;
        vector<float> distVec;
        w.getVector(s, wordVec, distVec);

        for(int i = 0; i < wordVec.size(); i++)
            cout << wordVec[i] << " " << distVec[i] << endl;

    }

    return 0;

}
