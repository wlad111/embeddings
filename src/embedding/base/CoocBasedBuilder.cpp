//
// Created by vlad on 15.04.19.
//

#include "CoocBasedBuilder.h"
#include <thread>
//#include <mutex>
//#include <condition_variable>

//producer
void CoocBasedBuilder::readWords(std::ifstream &reader) {
    {

        //std::unique_lock<std::mutex> l(mt);
        std::string word;
        int i = 0;
        for (reader >> word;  !reader.eof() && i < bufferSize; reader >> word) {
            buffer.push_back(word);
            i++;
        }
    }
    //cv.notify_one();
}

//consumer
void CoocBasedBuilder::sendWords() {
    {
        std::unique_lock<std::mutex> l(mt);
        /*while (buffer.empty()) {
            cv.wait(l);
        }*/
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


/*
       std::thread t1(&CoocBasedBuilder::readWords, source);
       std::thread t2(&CoocBasedBuilder::sendWords);
       t1.join();
       t2.join();*/

        while (!source.eof()) {
            readWords(source);
            sendWords();
        }

        auto end = std::chrono::steady_clock::now();
        int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Acquired cooccurences for " << elapsed << "ms" << std::endl;

    }
}

void CoocBasedBuilder::merge(std::vector<int64_t> acc) {

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

int32_t CoocBasedBuilder::unpackB(std::vector<int64_t> &cooc, int32_t v) {
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


