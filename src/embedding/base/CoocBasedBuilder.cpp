//
// Created by vlad on 15.04.19.
//

#include "CoocBasedBuilder.h"
#include <thread>
#include <mutex>
#include <condition_variable>

//producer
void CoocBasedBuilder::readWords(std::ifstream &reader) {
    {

        std::unique_lock<std::mutex> l(mt);
        std::string word;
        int i = 0;
        for (reader >> word;  !reader.eof() && i < bufferSize; reader >> word) {
            buffer.push_back(word);
            i++;
        }
    }
    cv.notify_one();
}

//consumer
void CoocBasedBuilder::sendWords() {
    {
        std::unique_lock<std::mutex> l(mt);
        while (buffer.empty()) {
            cv.wait(l);
        }
        processWords(buffer);
    }
}

void CoocBasedBuilder::processWords(std::deque<std::string> &buf) {
    std::string word;
    while (!buf.empty()) {
        word = buf.front();
        buf.pop_front();
    }
}

void CoocBasedBuilder::acquireCoocurrences() {
    if (!coocReady) {
        std::cout << "Generating cooccurences for " << path_ << std::endl;

        auto start = std::chrono::steady_clock::now();


        //TODO make stream of positions here!
        std::cout << "Streaming words from corpus " << std::endl;
        std::ifstream source(path_);
        //std::string line;
        //std::string newLine = "777newline777";
        /*int64_t nLine = 0;

        std::vector<int32_t> pos_queue;
        int32_t offset = 0;


        std::vector<std::lock_guard<std::mutex>> rowLocks;
//        rowLocks.resize(wordsList.size());

        std::vector<int64_t> res;
        std::string word;
        int64_t i = 0;
        for (source >> word; !source.eof(); source >> word) {
            if (i % 100000 == 0) {
                auto end = std::chrono::steady_clock::now();
                int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                std::cout << i << " words processed " << " for " << elapsed << "ms" << std::endl;
            }
            int32_t idx = wordsIndex[normalize(word)];

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
            }
            pos_queue.push_back(idx);
            if (pos_queue.size() > std::max(windowLeft, windowRight)) {
                offset++;
                if (offset > 1000 - std::max(windowLeft, windowRight)) {
                    pos_queue.erase(pos_queue.begin(), pos_queue.begin() + offset);
                    offset = 0;
                }
            }
            for (auto entry: out) {
                res.push_back(entry);
            }
            i++;
        }


        std::vector<std::vector<int64_t>> accumulators;
        cooc_.resize(wordsList.size());
*/


       std::thread t1(&CoocBasedBuilder::readWords, this, std::ref(source));
       std::thread t2(&CoocBasedBuilder::sendWords, this);
       t1.join();
       t2.join();

        /*while (!source.eof()) {
            readWords(source);
            sendWords();
        }*/

        auto end = std::chrono::steady_clock::now();
        int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Acquired cooccurences for " << elapsed << "ms" << std::endl;

    }
}

void CoocBasedBuilder::merge(std::vector<int64_t> &acc) {
    std::sort(acc.begin(), acc.end());
    const int size = acc.size();
    float weights[256];
    for (int i = 0; i < 256; i++) {
        //TODO это болванка, надо прикрутить тип окна и потом исправить эту строчку
        weights[i] = i > 126 ? -256 + i : i;
    }
    std::vector<int64_t> prevRow;
    std::vector<int64_t> updatedRow(wordsList.size(), 0);
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
            while (++i < size && ((next = acc[i]) & 0xFFFFFFFFFFFFFF00L) == currentPairMasked) {
                weight += weights[unpackDist(next)];
            }
            if (i < size) {
                i--;
            }

            if (a != prevA) {
                if (prevA >= 0){
                    std::vector<int64_t> longSeqRow = prevRow;
                    updatedRow.insert(updatedRow.end(), longSeqRow.begin() + pos, longSeqRow.begin() + prevLength);
                    std::vector<int64_t > result;
                    if (updatedRow.size() <= longSeqRow.size()) {
                        result = longSeqRow;
                        result.insert(result.begin(), updatedRow.begin(), updatedRow.end());
                    }
                    else {
                        result.resize(updatedRow.size());
                        result.insert(result.begin(), updatedRow.begin(), updatedRow.end());
                    }
                    cooc_[prevA] = result; //TODO ask IK about LongSeq.build() (вроде как тут надо заинсертить longseqrow в updatedrow. переделать!
                    int prev[] = {-1};
                    int prevAFinal = prevA;
                }
                //unlock rowLocks[prevA];
                prevA = a;
                prevRow = cooc_[a];
                prevLength = prevRow.size();
                pos = 0;
            }
            //lock rowLocks[prevA];


            int64_t  prevPacked;
            while (pos < prevLength) { // merging previous version of the cooc row with current data
                prevPacked = prevRow[pos];
                int32_t prevB = static_cast<int32_t >(prevPacked >> 32);
                if (prevB >= b) {
                    if (prevB == b) {
                        int32_t int_to_float = static_cast<int32_t >(prevPacked & 0xFFFFFFFFL);
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

        std::vector<int64_t >longSeqRow = prevRow;
        updatedRow.insert(updatedRow.end(), longSeqRow.begin() + pos, longSeqRow.begin() + prevLength);
        //        cooc.set(prevA, updatedRow.build(longSeqRow.data(), 0.2, 100));
        std::vector<int64_t > result;
        if (updatedRow.size() <= longSeqRow.size()) {
            result = longSeqRow;
            result.insert(result.begin(), updatedRow.begin(), updatedRow.end());
        }
        else {
            result.resize(updatedRow.size());
            result.insert(result.begin(), updatedRow.begin(), updatedRow.end());
        }
        cooc_[prevA] = result;
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

std::vector<int64_t> CoocBasedBuilder::cooc(size_t i) {
    return std::vector<int64_t>();
}

float CoocBasedBuilder::unpackWeight(std::vector<int64_t> &cooc, int32_t v) {
    return 0;
}


void CoocBasedBuilder::build() {

}

CoocBasedBuilder::CoocBasedBuilder(std::string &dict_path) : EmbeddingBuilderBase(dict_path) {
    std::cout << "CoocBasedBuilder" << std::endl;
}




//std::vector<uint64_t> CoocBasedBuilder::positionsStream() {
    std::vector<uint64_t> coocStream;

//}


