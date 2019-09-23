//
// Created by vlad on 15.04.19.
//

#include "CoocBasedBuilder.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cmath>

//producer
void CoocBasedBuilder::readWords(std::ifstream &reader) {
    {

        //std::unique_lock<std::mutex> l(mt);
        std::string word;
        std::vector<int32_t> pos_queue;
        int32_t offset = 0;
        std::vector<int64_t> res;
        size_t acc_idx = 0;
        accumulators.resize(accs_count);
        int i = 0;
        int words_count = 0;
        for (reader >> word;  !reader.eof() && i < bufferSize; reader >> word) {
            if (reader.eof()) {
                return;
            }
            words_count++;
            int32_t idx = wordsIndex[normalize(word)]; //TODO there can be null idx, careful
            int32_t pos = pos_queue.size();
            int64_t out[windowRight + windowLeft];
            int32_t outIndex = 0;
            for (int j = offset; j < pos; j++) {
                int8_t distance = pos - j;
                if (distance == 0) {
                    std::cout << "Zero distance occured! pos: " << pos << "i: " << j << std::endl;
                }
                if (distance <= windowRight) {
                    out[outIndex++] = pack(pos_queue[j], idx, distance);
                }
                if (distance <= windowLeft) {
                    out[outIndex++] = pack(idx, pos_queue[j], -distance);
                }
                i++;
            }
            pos_queue.push_back(idx);
            if (pos_queue.size() > std::max(windowLeft, windowRight)) {
                offset++;
                if (offset > 1000 - std::max(windowLeft, windowRight)) {
                    pos_queue.erase(pos_queue.begin(), pos_queue.begin() + offset);
                    offset = 0;
                }
            }
            for (int k = 0; k < outIndex; k++) {
                if (accumulators[acc_idx].size() > capacity) {
                    acc_idx++;
                    if (acc_idx > accs_count - 1) {
                        return;
                    }
                }
                accumulators[acc_idx].push_back(out[k]);
            }
            if (reader.eof()) {
                return;
            }
        }
    }
    //cv.notify_one();
}

//consumer
void CoocBasedBuilder::sendWords() {
    {
        //std::unique_lock<std::mutex> l(mt);
        /*while (buffer.empty()) {
            //cv.wait(l);
        }*/
        processWords(buffer);
    }
}

void CoocBasedBuilder::processWords(std::deque<std::string> &buf) {
    //std::for_each(accumulators.begin(), accumulators.end(), &CoocBasedBuilder::merge);
    auto func = std::bind(&CoocBasedBuilder::merge, this, std::placeholders::_1);
    std::for_each(accumulators.begin(), accumulators.end(), func);
    for (auto acc : accumulators) {
        acc.clear();
    }
    accumulators.clear();
}

void CoocBasedBuilder::acquireCoocurrences() {
    if (!coocReady) {
        std::cout << "Generating cooccurences for " << path_ << std::endl;



        auto start = std::chrono::steady_clock::now();
        cooc_.resize(wordsList.size());


        std::cout << "Streaming words from corpus " << std::endl;
        std::ifstream source(path_);



       // std::vector<std::lock_guard<std::mutex>> rowLocks;
//


    /*   std::thread t1(&CoocBasedBuilder::readWords, this, std::ref(source));
       std::thread t2(&CoocBasedBuilder::sendWords, this);
       t1.join();
       t2.join();*/

        while (!source.eof()) {
            readWords(source);
            sendWords();
            auto end = std::chrono::steady_clock::now();
            int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "Processed buffers for " << elapsed << "ms" << std::endl;

        }

        auto end = std::chrono::steady_clock::now();
        int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Acquired cooccurences for " << elapsed << "ms" << std::endl;
        coocReady = true;
    }
}

void CoocBasedBuilder::merge(std::vector<int64_t> &acc) {//TODO memory bug
    std::sort(acc.begin(), acc.end());
    const int size = acc.size();
    float weights[256];
    for (int i = 0; i < 256; i++) {
        int d = i > 126 ? -256 + i : i;
        weights[i] = d == 0 ? 0 : 1./std::abs(d); //TODO hard-coded linear window, fix later
    }
    std::vector<int64_t> prevRow{};
    std::vector<int64_t> updatedRow{};
    updatedRow.reserve(wordsList.size());
    int prevA = -1;
    int pos = 0; // insertion point
    int prevLength = 0;
    {
        for (int i = 0; i < size; i++) {
            int64_t next = acc[i];
            int64_t currentPairMasked = next & 0xFFFFFFFFFFFFFF00L;
            const int a = unpackA(next);
            const int b = unpackB(next);
            float weight = weights[unpackDist(next)];
            while ((++i < size) && (((next = acc[i]) & 0xFFFFFFFFFFFFFF00L) == currentPairMasked)) {
                weight += weights[unpackDist(next)];
            }
            if (i < size) {
                i--;
            }

            if (a != prevA) {
                if (prevA >= 0){
                    updatedRow.insert(updatedRow.end(), prevRow.begin() + pos, prevRow.begin() + prevLength);

                    if (cooc_[prevA].size() < updatedRow.size()) {
                        cooc_[prevA].resize(updatedRow.size());
                    }
                    std::copy(updatedRow.begin(), updatedRow.end(), cooc_[prevA].begin());
                    //cooc_[prevA] = result;
                    int prev[] = {-1};
                    int prevAFinal = prevA;
                    //unlock rowLocks[prevA];
                }
                prevA = a;
                prevRow = cooc_[a];
                prevLength = prevRow.size();
                pos = 0;
                updatedRow.resize(0);
            }
            //lock rowLocks[prevA];


            int64_t  prevPacked;
            while (pos < prevLength) { // merging previous version of the cooc row with current data
                prevPacked = prevRow[pos];
                auto prevB = static_cast<int32_t >(prevPacked >> 32);
                if (prevB >= b) {
                    if (prevB == b) {
                        auto int_to_float = static_cast<int32_t >(prevPacked & 0xFFFFFFFFL);
                        weight += (*reinterpret_cast<float*>(&int_to_float));
                        pos++;
                    }
                    break;
                }

                updatedRow.push_back(prevPacked);
                pos++;
            }
            int64_t repacked = (static_cast<int64_t >(b) << 32) | *reinterpret_cast<int*>(&weight);
            updatedRow.push_back(repacked);
        }

        updatedRow.insert(updatedRow.end(), prevRow.begin() + pos, prevRow.begin() + prevLength);

        if (prevA != -1) {
            //cooc_[prevA].insert(cooc_[prevA].begin(), updatedRow.begin(), updatedRow.end());
            if (cooc_[prevA].size() < updatedRow.size()) {
                cooc_[prevA].resize(updatedRow.size());
            }
            std::copy(updatedRow.begin(), updatedRow.end(), cooc_[prevA].begin());
        }
    }
    //unlock rowLocks[prevA]
}

void CoocBasedBuilder::fit() {
    std::cout << "fit" << std::endl;
    acquireDictionary();
    acquireCoocurrences();
}

std::vector<std::string> CoocBasedBuilder::dict() {
    return EmbeddingBuilderBase::dict();
}

int32_t CoocBasedBuilder::index(std::string word) {
    return EmbeddingBuilderBase::index(word);
}

const std::vector<int64_t> &CoocBasedBuilder::cooc(size_t i) const {
    return cooc_[i];
}

float CoocBasedBuilder::unpackWeight(std::vector<int64_t> &cooc, int32_t v) {
    return 0;
}


void CoocBasedBuilder::build() {
    
}

CoocBasedBuilder::CoocBasedBuilder(std::string &dict_path) : EmbeddingBuilderBase(dict_path) {
    std::cout << "CoocBasedBuilder" << std::endl;
}



