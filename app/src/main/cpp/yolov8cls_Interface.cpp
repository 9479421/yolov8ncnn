#include <jni.h>
#include <net.h>
#include <android/asset_manager_jni.h>
#include <cpu.h>
#include "yolov8cls_Interface.h"


bool yolov8cls_Interface::init(AAssetManager* mgr, bool use_gpu) {

    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);

    yolo.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolo.opt = ncnn::Option();

#if NCNN_VULKAN
    yolo.opt.use_vulkan_compute = use_gpu;
#endif

    yolo.opt.num_threads = ncnn::get_big_cpu_count();
    yolo.opt.blob_allocator = &blob_pool_allocator;
    yolo.opt.workspace_allocator = &workspace_pool_allocator;


    {
        int ret = yolo.load_param(mgr, "yolov8n-cls.param");
        if (ret != 0) {
            return JNI_FALSE;
        }
    }

    {
        int ret = yolo.load_model(mgr, "yolov8n-cls.bin");
        if (ret != 0) {
            return JNI_FALSE;
        }
    }

    return JNI_TRUE;
}


int yolov8cls_Interface::classDetect(ncnn::Mat in) {
    float prob_threshold = 0.8;   //最低置信度


    in.substract_mean_normalize(0, norm_vals);
    ncnn::Extractor ex = yolo.create_extractor();
    ex.input("images", in);

    ncnn::Mat out;
    ex.extract("output0", out);

    int maxIdx = -1;
    float maxVal = 0;
    float *output_data = out.row(0);
    for (int i = 0; i < out.w; i++) { //循环1000次，class的数量
        if (output_data[i] > maxVal) {
            maxVal = output_data[i];
            maxIdx = i;
        }
    }

    if (maxVal >= prob_threshold) {
        return maxIdx; //返回class的下标
    }

    return -1;
}