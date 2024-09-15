package vip.wqby.yolov8ncnn;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class yolov8cls {

    public native boolean init(AssetManager manager, boolean use_gpu);
    public native int classDetect(Bitmap bitmap);

    static {
        System.loadLibrary("yolov8ncnn");
    }
}
