
#include <jni.h>
#include <android/asset_manager_jni.h>
#include "yolov8cls_Interface.h"


static yolov8cls_Interface yolov8cls_Interface;

extern "C"
JNIEXPORT jboolean JNICALL
Java_vip_wqby_yolov8ncnn_yolov8cls_init(JNIEnv *env, jobject thiz, jobject manager, jboolean use_gpu) {
    AAssetManager *mgr = AAssetManager_fromJava(env, manager);

    return yolov8cls_Interface.init(mgr, use_gpu);
}




extern "C"
JNIEXPORT jint JNICALL
Java_vip_wqby_yolov8ncnn_yolov8cls_classDetect(JNIEnv *env, jobject thiz, jobject bitmap) {
    int target_size = 224;


    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);

    const int width = info.width;
    const int height = info.height;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return NULL;

    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h) {
        scale = (float) target_size / w;
        w = target_size;
        h = h * scale;
    } else {
        scale = (float) target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_android_bitmap_resize(env, bitmap, ncnn::Mat::PIXEL_RGB, w, h);

    // pad to target_size rectangle
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2,
                           ncnn::BORDER_CONSTANT, 0.f);

    return yolov8cls_Interface.classDetect(in_pad);
}