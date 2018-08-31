//
// Created by pesong on 18-8-31.
//

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../include/stb_image_resize.h"

#include "fp16.h"
#include <mvnc.h>



// somewhat arbitrary buffer size for the device name
#define NAME_SIZE 100

// graph file name - assume we are running in this directory: ncsdk/examples/caffe/GoogLeNet/cpp
#define GRAPH_FILE_NAME "../graph"

// image file name - assume we are running in this directory: ncsdk/examples/caffe/GoogLeNet/cpp
#define IMAGE_FILE_NAME "../../../data/images/nps_electric_guitar.png"


// 16 bits.  will use this to store half precision floats since C++ has no
// built in support for it.
typedef unsigned short half;

// GoogleNet image dimensions, network mean values for each channel in BGR order.
const int networkDim = 224;
float networkMean[] = {71.60167789, 82.09696889, 72.30608881};


bool g_graph_Success = false;
// movidius 设备预处理
mvncStatus retCode;
void *deviceHandle;
char devName[NAME_SIZE];
void* graphHandle;

typedef struct {
    int w;
    int h;
    int c;
    float *data;
} image;


// Load a graph file
// caller must free the buffer returned.
void *LoadFile(const char *path, unsigned int *length)
{
    FILE *fp;
    char *buf;

    fp = fopen(path, "rb");
    if(fp == NULL)
        return 0;
    fseek(fp, 0, SEEK_END);
    *length = ftell(fp);
    rewind(fp);
    if(!(buf = (char*) malloc(*length)))
    {
        fclose(fp);
        return 0;
    }
    if(fread(buf, 1, *length, fp) != *length)
    {
        fclose(fp);
        free(buf);
        return 0;
    }
    fclose(fp);
    return buf;
}


half *LoadImage(unsigned char *img, int reqsize, int width, int height)
{
    int i;
    unsigned char *imgresized;
    float *imgfp32;
    half *imgfp16;

    if(!img)
    {
        printf("The picture  could not be loaded\n");
        return 0;
    }
    imgresized = (unsigned char*) malloc(3*reqsize*reqsize);
    if(!imgresized)
    {
        free(img);
        perror("malloc");
        return 0;
    }
    //std::cout << "img: " << img << std::endl;
    stbir_resize_uint8(img, width, height, 0, imgresized, reqsize, reqsize, 0, 3);
    free(img);
    imgfp32 = (float*) malloc(sizeof(*imgfp32) * reqsize * reqsize * 3);
    if(!imgfp32)
    {
        free(imgresized);
        perror("malloc");
        return 0;
    }
    for(i = 0; i < reqsize * reqsize * 3; i++)
        imgfp32[i] = imgresized[i];
    free(imgresized);
    imgfp16 = (half*) malloc(sizeof(*imgfp16) * reqsize * reqsize * 3);
    if(!imgfp16)
    {
        free(imgfp32);
        perror("malloc");
        return 0;
    }
    //adjust values to range between -1.0 and + 1.0
    //change color channel
    for(i = 0; i < reqsize*reqsize; i++)
    {
        float blue, green, red;
        blue = imgfp32[3*i+2];
        green = imgfp32[3*i+1];
        red = imgfp32[3*i+0];
        imgfp32[3*i+0] = (blue-127.5)*0.007843;
        imgfp32[3*i+1] = (green-127.5)*0.007843;
        imgfp32[3*i+2] = (red-127.5)*0.007843;
        // uncomment to see what values are getting passed to mvncLoadTensor() before conversion to half float
        //printf("Blue: %f, Grean: %f,  Red: %f \n", imgfp32[3*i+0], imgfp32[3*i+1], imgfp32[3*i+2]);
    }
    floattofp16((unsigned char *)imgfp16, imgfp32, 3*reqsize*reqsize);
    free(imgfp32);
    return imgfp16;
}

unsigned char* image_to_stb(image in)
{
    int i,j,k;
    int w = in.w;
    int h = in.h;
    int c =3;
    unsigned char *img = (unsigned char*) malloc(c*w*h);
    for(k = 0; k < c; ++k){
        for(j=0; j <h; ++j){
            for(i=0; i<w; ++i){
                int src_index = i + w*j + w*h*k;
                int dst_index = k + c*i + c*w*j;
                img[dst_index] = (unsigned char)(255*in.data[src_index]);
            }
        }
    }
    // std::cout << "xxxxx" << std::endl;
    return img;
}

image make_empty_image(int w, int h, int c)
{
    image out;
//  out.data = 0;
    out.data = new float[h*w*c]();//calloc(h*w*c, sizeof(float));
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}


image make_image(int w, int h, int c)
{
    image out = make_empty_image(w,h,c);
//  out.data = calloc(h*w*c, sizeof(float));
    return out;
}



void ipl_into_image(IplImage* src, image im)
{
    unsigned char *data = (unsigned char *)src->imageData;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    int step = src->widthStep;
    int i, j, k;
    // std::cout << "h: " << h << " w: " << w << " c: " << c << std::endl;
    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){

                //std::cout << "index: "<< k*w*h + i*w + j << " data: " << data[i*step + j*c + k]/255. << " i: " << i << " k："<< k << " j:" << j << " w: "<< w <<" h: "<<h <<" c: "<< c << std::endl;
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
}


image ipl_to_image(IplImage* src)
{
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image out = make_image(w, h, c);
    ipl_into_image(src, out);
    return out;
}

unsigned char* cvMat_to_charImg(cv::Mat pic)
{
    IplImage copy = pic;
    IplImage *pic_Ipl = &copy;
    //std::cout << "pic_Ipl.width: " << pic_Ipl->width << " pic_Ipl.height: " << pic_Ipl->height << std::endl;
    //cvSaveImage("/home/ziwei/human_track_ssd/aaa.jpg",pic_Ipl);
//    ipl_into_image(pic_Ipl, buff_);
    unsigned char* pic_final = image_to_stb(ipl_to_image(pic_Ipl));

    return pic_final;
}


// callback for inference
void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {
        cv::Mat ROS_img = cv_bridge::toCvShare(msg, "bgr8")->image;
        cv::imshow("view", ROS_img);

        unsigned char *img = cvMat_to_charImg(ROS_img);

        unsigned int graphFileLen;
        half *imageBufFp16 = LoadImage(img, networkDim, ROS_img.cols, ROS_img.rows);


        // calculate the length of the buffer that contains the half precision floats.
        // 3 channels * width * height * sizeof a 16bit float
        unsigned int lenBufFp16 = 3 * networkDim * networkDim * sizeof(*imageBufFp16);

        //std::cout << "networkDim: " << networkDim << " imageBufFp16: " << sizeof(*imageBufFp16) << " lenBufFp16: " << lenBufFp16 << std::endl;
        retCode = mvncLoadTensor(graphHandle, imageBufFp16, lenBufFp16, NULL);

        if (retCode != MVNC_OK) {     // error loading tensor
            perror("Could not load ssd tensor\n");
            printf("Error from mvncLoadTensor is: %d\n", retCode);
        }
        printf("Successfully loaded the tensor for image\n");

        // 判断 inference Graph的状态
        if (g_graph_Success == true)
        {
            void *resultData16;
            void *userParam;
            unsigned int lenResultData;
            // 执行inference
            retCode = mvncGetResult(graphHandle, &resultData16, &lenResultData, &userParam);
        }

    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
    cv::waitKey(10);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "image_listener");
    ros::NodeHandle nh;


    retCode = mvncGetDeviceName(0, devName, NAME_SIZE);
    if (retCode != MVNC_OK)
    {   // failed to get device name, maybe none plugged in.
        printf("No NCS devices found\n");
        exit(-1);
    }

    // Try to open the NCS device via the device name
    retCode = mvncOpenDevice(devName, &deviceHandle);
    if (retCode != MVNC_OK)
    {   // failed to open the device.
        printf("Could not open NCS device\n");
        exit(-1);
    }

    // deviceHandle is ready to use now.
    // Pass it to other NC API calls as needed and close it when finished.
    printf("Successfully opened NCS device!\n");

    // Now read in a graph file
    unsigned int graphFileLen;
    void* graphFileBuf = LoadFile(GRAPH_FILE_NAME, &graphFileLen);

    // allocate the graph

    retCode = mvncAllocateGraph(deviceHandle, &graphHandle, graphFileBuf, graphFileLen);
    if (retCode != MVNC_OK)
    {   // error allocating graph
        printf("Could not allocate graph for file: %s\n", GRAPH_FILE_NAME);
        printf("Error from mvncAllocateGraph is: %d\n", retCode);
    } else {
        printf("Successfully allocate graph for file: %s\n", GRAPH_FILE_NAME);
        g_graph_Success = true;
    }

    cv::namedWindow("view");
    cv::startWindowThread();
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub = it.subscribe("camera/image", 1, imageCallback);
    ros::spin();
    cv::destroyWindow("view");
}