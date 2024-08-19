package vip.wqby.yolov8ncnn;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

import androidx.annotation.NonNull;

public class yolov8 {

    public yolov8(){

    }

    public class Obj
    {
        public int x;
        public int y;
        public int w;
        public int h;
        public int label;
        public float prob;

    }


    public native boolean init(AssetManager manager, boolean use_gpu);
    public native Obj[] detect(Bitmap bitmap);

    static {
        System.loadLibrary("yolov8ncnn");
    }
}
