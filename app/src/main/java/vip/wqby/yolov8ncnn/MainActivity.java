package vip.wqby.yolov8ncnn;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.widget.TextView;

import java.io.InputStream;

import vip.wqby.yolov8ncnn.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'yolov8ncnn' library on application startup.
    static {
        System.loadLibrary("yolov8ncnn");
    }

    private ActivityMainBinding binding;


    yolov8 yolov8 = new yolov8();
    yolov8cls yolov8cls = new yolov8cls();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());


        yolov8.init(getAssets(), true);
        yolov8cls.init(getAssets(), true);

        binding.sampleText.setOnClickListener(v -> {
            //识别
            try {
                InputStream open = getAssets().open("1008.png");
                yolov8.Obj[] detect = yolov8.detect(BitmapFactory.decodeStream(open));
                for (yolov8.Obj obj : detect) {
                    System.out.println(obj.label + " " + obj.prob + " " + obj.x + " " + obj.y + " " + obj.w + " " + obj.h);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
            //分类
            try {
                InputStream open = getAssets().open("fish.png");
                int result = yolov8cls.classDetect(BitmapFactory.decodeStream(open));
                System.out.println("clsResult:" + result);
            }catch (Exception e){
                e.printStackTrace();
            }

        });


    }

}