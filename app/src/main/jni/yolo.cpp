#include "yolo.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cpu.h"


SSD::SSD() {
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int SSD::load(const char *modeltype, int _target_size, const float *_norm_vals, bool use_gpu) {
    mobilenetssd.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    mobilenetssd.opt = ncnn::Option();

#if NCNN_VULKAN
    mobilenetssd.opt.use_vulkan_compute = use_gpu;
#endif

    mobilenetssd.opt.num_threads = ncnn::get_big_cpu_count();
    mobilenetssd.opt.blob_allocator = &blob_pool_allocator;
    mobilenetssd.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    mobilenetssd.load_param(parampath);
    mobilenetssd.load_model(modelpath);

    target_size = _target_size;
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

int SSD::load(AAssetManager *mgr, const char *modeltype, int _target_size, const float *_norm_vals,
               bool use_gpu) {
    mobilenetssd.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    mobilenetssd.opt = ncnn::Option();
#if NCNN_VULKAN
    mobilenetssd.opt.use_vulkan_compute = use_gpu;
#endif
    mobilenetssd.opt.num_threads = ncnn::get_big_cpu_count();
    mobilenetssd.opt.blob_allocator = &blob_pool_allocator;
    mobilenetssd.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    mobilenetssd.load_param(mgr, parampath);
    mobilenetssd.load_model(mgr, modelpath);


    target_size = _target_size;
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

int SSD::detect(const cv::Mat &rgb, std::vector<Object> &objects, float prob_threshold,
                 float nms_threshold) {
    double start_time = ncnn::get_current_time();
    int img_w = rgb.cols;
    int img_h = rgb.rows;
    const int target_size = 300;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, img_w, img_h, target_size,
                                                 target_size);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0/127.5,1.0/127.5,1.0/127.5};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = mobilenetssd.create_extractor();
    ex.input("data", in);


    {
        ncnn::Mat out;
        ex.extract("detection_out", out);

        //后处理
        for(int i =0; i < out.h; i++){
            const float* values = out.row(i);

            Object object;
            object.label = values[0];
            object.prob = values[1];
            object.x = values[2] * img_w;
            object.y = values[3] * img_h;
            object.w = values[4] * img_w - object.x;
            object.h = values[5] * img_h - object.y;

            objects.push_back(object);
        }
        double elasped = ncnn::get_current_time() - start_time;
        NCNN_LOGE("infer_times= %.2fms",elasped);

    }
    return 0;
}

int SSD::draw(cv::Mat &rgb, const std::vector<Object> &objects) {
    static const char *class_names[] = {
            "background",
            "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair",
            "cow", "diningtable", "dog", "horse",
            "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"
    };
    static const unsigned char colors[19][3] = {
            {54,  67,  244},
            {99,  30,  233},
            {176, 39,  156},
            {183, 58,  103},
            {181, 81,  63},
            {243, 150, 33},
            {244, 169, 3},
            {212, 188, 0},
            {136, 150, 0},
            {80,  175, 76},
            {74,  195, 139},
            {57,  220, 205},
            {59,  235, 255},
            {7,   193, 255},
            {0,   152, 255},
            {34,  87,  255},
            {72,  85,  121},
            {158, 158, 158},
            {139, 125, 96}
    };

    int color_index = 0;

    for (size_t i = 0; i < objects.size(); i++) {
        const Object &obj = objects[i];

        const unsigned char *color = colors[color_index % 19];
        color_index++;

        cv::Scalar cc(color[0], color[1], color[2]);

        cv::rectangle(rgb, cv::Point(obj.x,obj.y), cv::Point(obj.x+obj.w,obj.y+obj.h),cc, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.x;
        int y = obj.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;
        cv::rectangle(rgb, cv::Rect(cv::Point(x, y),
                                    cv::Size(label_size.width, label_size.height + baseLine)), cc,
                      -1);
        cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381) ? cv::Scalar(0, 0, 0)
                                                                    : cv::Scalar(255, 255, 255);
        cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    textcc, 1);
    }

    return 0;
}
