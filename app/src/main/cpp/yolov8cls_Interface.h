#include <allocator.h>
#include <net.h>

class yolov8cls_Interface {
private:
//    float mean_vals[3] = {103.53f, 116.28f, 123.675f};
    float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
    ncnn::Net yolo;

public:
    bool init(AAssetManager* mgr, bool use_gpu);
    int classDetect(ncnn::Mat in); //传入填充后的224*224图片
};