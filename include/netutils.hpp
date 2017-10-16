#pragma once 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "boost/asio.hpp"
#include <boost/system/error_code.hpp>

using namespace boost::asio;

struct UDPStreamer{


	UDPStreamer(int port, const std::string & address) : port_(port), socket_(udp_service_)
	{
		socket_.open(ip::udp::v4());
		remote_endpoint_ = ip::udp::endpoint(ip::address::from_string(address.c_str()), port_);
		std::cout << "remote endpoint " << remote_endpoint_ << std::endl;
	}

	void close()
	{
		socket_.close();
	}

	boost::system::error_code sendMessage(const std::string & message)
	{
		boost::system::error_code err;
		socket_.send_to(buffer(message.c_str(), message.size()), remote_endpoint_, 0, err);
		return err;
	}



	int port_;
	io_service udp_service_;
	ip::udp::socket socket_;
	ip::udp::endpoint remote_endpoint_;

};

