#pragma once
#include "pooledchannel.hpp"
#include <memory>
#include <vector>
#include <mutex> 

template <typename T>
class ChannelWrapper{

public:

	ChannelWrapper(bool & stopper, int size): stopper_(stopper), size_(size) {}
	std::shared_ptr<PooledChannel<std::shared_ptr<T>>> getNewChannel();
	std::vector<std::shared_ptr<PooledChannel<std::shared_ptr<T>>>> & getChannels();
	void write(std::shared_ptr<T> t);
	void shutDown();

private:

	std::mutex mutex_;
	std::vector<std::shared_ptr<PooledChannel<std::shared_ptr<T>>>> pc_;
	bool & stopper_;
	int size_;

};

template <typename T>
std::vector<std::shared_ptr<PooledChannel<std::shared_ptr<T>>>> & ChannelWrapper<T>::getChannels(){
	return pc_;
}

template <typename T>
std::shared_ptr<PooledChannel<std::shared_ptr<T>>> ChannelWrapper<T>::getNewChannel(){

	mutex_.lock();
	pc_.push_back(std::make_shared<PooledChannel<std::shared_ptr<T>>>(size_, true, false));
	pc_.back()->setTermination(&stopper_);
	mutex_.unlock();
	return pc_.back();

}

template <typename T>
void ChannelWrapper<T>::shutDown() {

	mutex_.lock();
	for(auto & ch : pc_)
			ch->notify_all();
	mutex_.unlock();
}

template <typename T>
void ChannelWrapper<T>::write(std::shared_ptr<T> t) {

	mutex_.lock();
	for(auto & ch : pc_)
			ch->write(t);
	mutex_.unlock();
}
