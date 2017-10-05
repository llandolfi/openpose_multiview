#include <stdint.h>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <vector>
#include <iostream>
#include <tuple>

// header
struct ImageHeader
{
	ImageHeader(int w, int h) : width(w),height(h) {}
	int width;
	int height;
	int frame;
	uint64_t time;
};

// ipc support
struct IPCData
{
	boost::interprocess::interprocess_mutex      mutex;

	//Condition to wait when the queue is empty
	boost::interprocess::interprocess_condition  cond_empty;

	//Condition to wait when the queue is full
	boost::interprocess::interprocess_condition  cond_full;
	bool message_in;
};

struct ImageRGBD: public ImageHeader
{
	ImageRGBD(int w, int h) : ImageHeader(w,h)
	{
		std::cout << "ctor " << w << " " << h << "\n"; //" buffers " << (void*)depthData() << " " << (void*)rgbData() << std::endl;
	}

	// to the color
	template <class Up>
	uint8_t * rgbData(Up *p) { return p->data(); }

	// depth after color
	template <class Up>
	uint8_t * depthData(Up *p) { return p->data() + width*height*3; }

	// whole size
	static int computeVariableSize(int w, int h)
	{
		return w*h*3+w*h*2;
	}
};

template <class Base>
struct OnShared: public Base
{
	IPCData ipc;
	uint8_t body[]; // variable (0 sized object used for extending the buffer)	

	template <class... T>
	OnShared(T... args) : Base(args...)
	{

	}

	uint8_t * data() { return body; }

	template <class... T>
	static int computeSize(T... args)
	{
		return sizeof(OnShared) + Base::computeVariableSize(args...);
	}
};