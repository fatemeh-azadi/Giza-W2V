#include "Word2Vec.h"

Word2Vec::Word2Vec(int n = 1000)
{
    N = n;
    wordsSrc = sizeSrc = wordsTrg = sizeTrg = 0;
    bestw = new char*[N];
}

void Word2Vec::readWord2Vec(std::string vectorsFileSrc, std::string vectorsFileTrg, vcbList *elist, vcbList *flist)
{

    std::cout << "reading source word2vec ...." << std::endl;
    readWord2Vec(vectorsFileSrc, 1);
    std::cout << "reading target word2vec ...." << std::endl;
    readWord2Vec(vectorsFileTrg, 0);
    std::cout << "reading word2vecs finished" << std::endl;


    for (int a = 0; a < N; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
    bestd = (float *)malloc(N * sizeof(float));
    vec = (float*)malloc(max_size * sizeof(float));

    std::cout << "computing similarities for words ...." << std::endl;
    computeSimilarWords(elist, flist);
   // std::cout << wordsSrc << " "<< wordsTrg << std::endl;
   // std::cout << sizeSrc << " " << sizeTrg << std::endl;

}

void Word2Vec::readWord2Vec(std::string vectorsFile, int isSrc){
    FILE *f;
    long long a, b;
    float len;
    char *vocab;
    float *M;
    long long words, size;
    std::map<std::string, long long> &dic = dicSrc;
    f = fopen(vectorsFile.c_str(), "rb");
    if (f == NULL) {
        printf("Word2Vec file not found\n");
        exit(EXIT_FAILURE);
    }
    if(isSrc){


        fscanf(f, "%lld", &wordsSrc);
        fscanf(f, "%lld", &sizeSrc);
        vocabSrc = (char *)malloc((long long)wordsSrc * max_w * sizeof(char));

        MSrc = (float *)malloc((long long)wordsSrc * (long long)sizeSrc * sizeof(float));

        words = wordsSrc;
        size = sizeSrc;
        vocab = vocabSrc;
        M = MSrc;
        dic = dicSrc;
    }else{

        fscanf(f, "%lld", &wordsTrg);
        fscanf(f, "%lld", &sizeTrg);
        vocabTrg = (char *)malloc((long long)wordsTrg * max_w * sizeof(char));

        MTrg = (float *)malloc((long long)wordsTrg * (long long)sizeTrg * sizeof(float));

        words = wordsTrg;
        size = sizeTrg;
        vocab = vocabTrg;
        M = MTrg;
        dic = dicTrg;
    }

    if (M == NULL) {
        printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
        exit(EXIT_FAILURE);
    }

    for (b = 0; b < words; b++) {
        a = 0;
        while (1) {
            vocab[b * max_w + a] = fgetc(f);
            if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
            if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
        }

        vocab[b * max_w + a] = 0;
        for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
        len = 0;
        for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
        len = sqrt(len);
        for (a = 0; a < size; a++) M[a + b * size] /= len;
    }
    fclose(f);
    char *tmp = (char *)malloc(max_size * sizeof(char));


    for (b = 0; b < words; b++) {
        //cout << b << endl;
        strcpy(tmp, &vocab[b * max_w]);
        dic[tmp] = b;
    }
    free(tmp);
}

void Word2Vec::getVector(std::string s, std::vector<std::string> &wordVec, std::vector<float> &distVec){
    wordVec.clear();
    distVec.clear();
    char **bestw;
    bestw = new char*[N];
    float *bestd;
    float*vec;
    float dist,len;
    long long a, b, c, d, bi;

    for (a = 0; a < N; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
    bestd = (float *)malloc(N * sizeof(float));
    vec = (float*)malloc(max_size * sizeof(float));

    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;

    for (b = 0; b < wordsSrc; b++)  if (!strcmp(&vocabSrc[b * max_w], s.c_str())) break;

    if (b == wordsSrc) b = -1;
    bi = b;
    if (bi == -1)
        return;

    for (a = 0; a < sizeSrc; a++) vec[a] = 0;
    for (a = 0; a < sizeSrc; a++) vec[a] += MSrc[a + bi * sizeSrc];

    len = 0;
    for (a = 0; a < sizeSrc; a++) len += vec[a] * vec[a];
    len = sqrt(len);
    //for (a = 0; a < size; a++) vec[a] /= len;
    for (a = 0; a < N; a++) bestd[a] = -1;
    for (a = 0; a < N; a++) bestw[a][0] = 0;

    for (c = 0; c < wordsTrg; c++) {
        a = 0;
       // if (bi == c) continue;
        dist = 0;
        for (a = 0; a < sizeTrg; a++) dist += vec[a] * MTrg[a + c * sizeTrg];
        for (a = 0; a < N; a++) {
            if (dist > bestd[a]) {
                for (d = N - 1; d > a; d--) {
                    bestd[d] = bestd[d - 1];
                    strcpy(bestw[d], bestw[d - 1]);
                }
                bestd[a] = dist;
                strcpy(bestw[a], &vocabTrg[c * max_w]);
                break;
            }
        }
    }

    for (a = 0; a < N; a++)
        if(bestd[a] != -1){
            wordVec.push_back(bestw[a]);
            distVec.push_back(bestd[a]);
        }
    for (a = 0; a < N; a++) free(bestw[a]);
    free(bestd);
    free(vec);
    free(bestw);

    return;
}

void Word2Vec::computeSimilarWords(vcbList* elist, vcbList* flist){
    similarWords.resize(elist->size());
    for(int i = 0; i < wordsSrc; i++){
       WordIndex sIdx = elist->getVocabId(&vocabSrc[i * max_w]);
       if(sIdx == -1)
           continue;
       std::vector<std::string> sim = getVector(&vocabSrc[i * max_w]);

       for(int j = 0; j < sim.size(); j++){
            WordIndex tIdx = flist->getVocabId(sim[j]);
            if(tIdx == -1)
                continue;
            similarWords[sIdx][tIdx] = true;
       }
     if( i % 1000 == 0)  cout << i << endl;

    }
}

std::vector<std::string> Word2Vec::getVector(std::string s){
    std::vector<std::string> wordVec;
    //std::vector<float> distVec;

    float dist,len;
    long long a, b, c, d, bi;


    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;

    for (b = 0; b < wordsSrc; b++)  if (!strcmp(&vocabSrc[b * max_w], s.c_str())) break;

    if (b == wordsSrc) b = -1;
    bi = b;
    if (bi == -1)
        return wordVec;

    for (a = 0; a < sizeSrc; a++) vec[a] = 0;
    for (a = 0; a < sizeSrc; a++) vec[a] += MSrc[a + bi * sizeSrc];

    len = 0;
    for (a = 0; a < sizeSrc; a++) len += vec[a] * vec[a];
    len = sqrt(len);
    //for (a = 0; a < size; a++) vec[a] /= len;
    for (a = 0; a < N; a++) bestd[a] = -1;
    for (a = 0; a < N; a++) bestw[a][0] = 0;

    for (c = 0; c < wordsTrg; c++) {
        a = 0;
       // if (bi == c) continue;
        dist = 0;
        for (a = 0; a < sizeTrg; a++) dist += vec[a] * MTrg[a + c * sizeTrg];
        for (a = 0; a < N; a++) {
            if (dist > bestd[a]) {
                for (d = N - 1; d > a; d--) {
                    bestd[d] = bestd[d - 1];
                    strcpy(bestw[d], bestw[d - 1]);
                }
                bestd[a] = dist;
                strcpy(bestw[a], &vocabTrg[c * max_w]);
                break;
            }
        }
    }

    for (a = 0; a < N; a++)
        if(bestd[a] != -1){
            wordVec.push_back(bestw[a]);
          //  distVec.push_back(bestd[a]);
        }

    return wordVec;
}

std::map<WordIndex, bool> Word2Vec::getVectorMap(WordIndex s){

    return similarWords[s];
}
