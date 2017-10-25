#include <libfreenect/libfreenect.hpp>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#ifdef HAVE_OMP
#include "omp.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif // HAVE_OMP


class Mutex {
public:
        Mutex() {
                pthread_mutex_init( &m_mutex, NULL );
        }
        void lock() {
                pthread_mutex_lock( &m_mutex );
        }
        void unlock() {
                pthread_mutex_unlock( &m_mutex );
        }
private:
        pthread_mutex_t m_mutex;
};


class MyFreenectDevice : public Freenect::FreenectDevice {
public:


        std::vector<uint16_t> m_buffer_depth;
        std::vector<uint8_t> m_buffer_video;
        std::vector<uint16_t> m_gamma;
        Mutex m_rgb_mutex;
        Mutex m_depth_mutex;
        bool m_new_rgb_frame;
        bool m_new_depth_frame;

        uint16_t getDepthBufferSize16() 
        {
            return getDepthBufferSize()/2;
        }

        MyFreenectDevice(freenect_context *_ctx, int _index)
                : Freenect::FreenectDevice(_ctx, _index), m_buffer_depth(getDepthBufferSize()),m_buffer_video(getVideoBufferSize()), m_gamma(2048), m_new_rgb_frame(false), m_new_depth_frame(false)
        {
    
            for( unsigned int i = 0 ; i < 2048 ; i++) {
                    float v = i/2048.0;
                    v = std::pow(v, 3)* 6;
                    m_gamma[i] = v*6*256;
            }


        }
        // Do not call directly even in child
        void VideoCallback(void* _rgb, uint32_t timestamp) {
                //std::cout << "RGB callback" << std::endl;
                m_rgb_mutex.lock();
                uint8_t* rgb = static_cast<uint8_t*>(_rgb);
                std::copy(rgb, rgb+getVideoBufferSize(), m_buffer_video.begin());
                m_new_rgb_frame = true;
                m_rgb_mutex.unlock();
        };
        // Do not call directly even in child
        void DepthCallback(void* _depth, uint32_t timestamp) {

                m_depth_mutex.lock();
                uint16_t* depth = static_cast<uint16_t*>(_depth);
                // was getVideoBufferSize()
                std::copy(depth, depth+getDepthBufferSize(), m_buffer_depth.begin());
                m_new_depth_frame = true;
                m_depth_mutex.unlock();
        }
        bool getRGB(std::vector<uint8_t> &buffer) {
                m_rgb_mutex.lock();
                if(m_new_rgb_frame) {
                        buffer.swap(m_buffer_video);
                        m_new_rgb_frame = false;
                        m_rgb_mutex.unlock();
                        return true;
                } else {
                        m_rgb_mutex.unlock();
                        return false;
                }
        }
 
        bool getDepth(std::vector<uint16_t> &buffer) {
                m_depth_mutex.lock();
                if(m_new_depth_frame) {
                        buffer.swap(m_buffer_depth);
                        m_new_depth_frame = false;
                        m_depth_mutex.unlock();
                        return true;
                } else {
                        m_depth_mutex.unlock();
                        return false;
                }
        }
};


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class freenectServer
{
public:
    Freenect::Freenect myfreenect;
    MyFreenectDevice* device;
    std::vector<uint16_t> depth_map;
    std::vector<uint8_t> rgb;


    freenectServer(){
        depth_map.resize(640*480*4);
        rgb.resize(640*480*4);
        freenect_video_format requested_format = FREENECT_VIDEO_RGB;
        device = &myfreenect.createDevice<MyFreenectDevice>(0);
        device->setDepthFormat(FREENECT_DEPTH_REGISTERED);
        device->setVideoFormat(requested_format);
        device->startDepth();
        device->startVideo();
    }

    ~freenectServer(){
        device->stopVideo();
        device->stopDepth();
    }

    void postRGBD(cv::Mat & RGB, cv::Mat & depth) 
    {

        //get rgb and depth data
        while(!device -> getDepth(depth_map)){}
        while(!device -> getRGB(rgb)){}
        
        int depth_width = 640;
        int depth_height = 480;

        RGB = cv::Mat(480,640,CV_8UC3);
        memcpy(RGB.data, rgb.data(), (640*480*3)*sizeof(uint8_t));
        cvtColor(RGB,RGB,CV_RGB2BGR);

        depth = cv::Mat(480,640,CV_16UC1);
        memcpy(depth.data, depth_map.data(), (640*480)*sizeof(uint16_t));
    }
};