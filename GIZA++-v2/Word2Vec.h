#ifndef WORD2VEC_H
#define WORD2VEC_H

#include<vector>
#include<string>
#include <map>
#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <iostream>
#include <stdlib.h>
#include <boost/algorithm/string.hpp>
#include "vocab.h"

class Word2Vec
{

public:
    static const long long max_size = 2000;         // max length of strings
    long long N;                  // number of closest words that will be shown
    static const long long max_w = 200;              // max length of vocabulary entries
    long long wordsSrc, sizeSrc, wordsTrg, sizeTrg;
    float *MSrc, *MTrg;
    char *vocabSrc, *vocabTrg;
    //std::map<std::string, std::map<std::string, float> > cosineDistance;
    std::map<std::string, long long> dicSrc, dicTrg;

    std::vector<std::map<WordIndex, bool > > similarWords;
    char **bestw;
    float *bestd;
    float*vec;

public:
    Word2Vec(int n);
    void readWord2Vec(std::string vectorsFileSrc, std::string vectorsFileTrg, vcbList* elist=0, vcbList* flist=0);
    void readWord2Vec(std::string vectorsFile, int isSrc);
    void computeSimilarWords(vcbList* elist=0, vcbList* flist=0);
    void getVector(std::string s, std::vector<std::string> &wordVec, std::vector<float> &distVec);
    std::vector<std::string> getVector(std::string s);
    std::map<WordIndex, bool> getVectorMap(WordIndex s);
};

#endif // WORD2VEC_H
