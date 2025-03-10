#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>

#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.5

constexpr int INPUT_W = 416;
constexpr int INPUT_H = 416;
constexpr int batchSize = 1;
constexpr int NUM_CLASSES = 1;
constexpr bool SHOW_IMG = true;

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

cv::Mat static_resize(cv::Mat &img) {
  float r = std::min(INPUT_W / (img.cols * 1.0), INPUT_H / (img.rows * 1.0));
  int unpad_w = r * img.cols;
  int unpad_h = r * img.rows;
  cv::Mat re(unpad_h, unpad_w, CV_8UC3);
  cv::resize(img, re, re.size());
  cv::Mat out(INPUT_W, INPUT_H, CV_8UC3, cv::Scalar(114, 114, 114));
  re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
  return out;
}

struct Object {
  cv::Rect_<float> rect;
  int label;
  float prob;
};

struct GridAndStride {
  int grid0;
  int grid1;
  int stride;
};

static void generate_grids_and_stride(const int target_size, std::vector<int> &strides,
                          std::vector<GridAndStride> &grid_strides) {
  for (auto stride : strides) {
    int num_grid = target_size / stride;
    for (int g1 = 0; g1 < num_grid; g1++) {
      for (int g0 = 0; g0 < num_grid; g0++) {
        grid_strides.push_back((GridAndStride){g0, g1, stride});
      }
    }
  }
}

static void generate_yolox_proposals(std::vector<GridAndStride> grid_strides,
                                     const float *feat_ptr,
                                     float prob_threshold,
                                     std::vector<Object> &objects) {
  const int num_class = NUM_CLASSES;
  const int num_anchors = grid_strides.size();

  for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
    const int grid0 = grid_strides[anchor_idx].grid0;
    const int grid1 = grid_strides[anchor_idx].grid1;
    const int stride = grid_strides[anchor_idx].stride;

    const int basic_pos = anchor_idx * 6;

    float x_center = (feat_ptr[basic_pos + 0] + grid0) * stride;
    float y_center = (feat_ptr[basic_pos + 1] + grid1) * stride;
    float w = exp(feat_ptr[basic_pos + 2]) * stride;
    float h = exp(feat_ptr[basic_pos + 3]) * stride;
    float x0 = x_center - w * 0.5f;
    float y0 = y_center - h * 0.5f;

    float box_objectness = feat_ptr[basic_pos + 4];
    for (int class_idx = 0; class_idx < num_class; class_idx++) {
      float box_cls_score = feat_ptr[basic_pos + 5 + class_idx];
      float box_prob = box_objectness * box_cls_score;
      if (box_prob > prob_threshold) {
        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = w;
        obj.rect.height = h;
        obj.label = class_idx;
        obj.prob = box_prob;

        objects.push_back(obj);
      }

    } // class loop

  } // point anchor loop
}

static inline float intersection_area(const Object &a, const Object &b) {
  cv::Rect_<float> inter = a.rect & b.rect;
  return inter.area();
}

static void qsort_descent_inplace(std::vector<Object> &faceobjects, int left,
                                  int right) {
  int i = left;
  int j = right;
  float p = faceobjects[(left + right) / 2].prob;

  while (i <= j) {
    while (faceobjects[i].prob > p)
      i++;

    while (faceobjects[j].prob < p)
      j--;

    if (i <= j) {
      // swap
      std::swap(faceobjects[i], faceobjects[j]);

      i++;
      j--;
    }
  }

#pragma omp parallel sections
  {
#pragma omp section
    {
      if (left < j)
        qsort_descent_inplace(faceobjects, left, j);
    }
#pragma omp section
    {
      if (i < right)
        qsort_descent_inplace(faceobjects, i, right);
    }
  }
}

static void qsort_descent_inplace(std::vector<Object> &objects) {
  if (objects.empty())
    return;

  qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object> &faceobjects,
                              std::vector<int> &picked, float nms_threshold) {
  picked.clear();

  const int n = faceobjects.size();

  std::vector<float> areas(n);
  for (int i = 0; i < n; i++) {
    areas[i] = faceobjects[i].rect.area();
  }

  for (int i = 0; i < n; i++) {
    const Object &a = faceobjects[i];

    int keep = 1;
    for (int j = 0; j < (int)picked.size(); j++) {
      const Object &b = faceobjects[picked[j]];

      // intersection over union
      float inter_area = intersection_area(a, b);
      float union_area = areas[i] + areas[picked[j]] - inter_area;
      // float IoU = inter_area / union_area
      if (inter_area / union_area > nms_threshold)
        keep = 0;
    }

    if (keep)
      picked.push_back(i);
  }
}

static void decode_outputs(const float *prob, std::vector<Object> &objects,
                           float scale, const int img_w, const int img_h) {
  std::vector<Object> proposals;
  std::vector<int> strides = {8, 16, 32};
  std::vector<GridAndStride> grid_strides;

  
  generate_grids_and_stride(INPUT_W, strides, grid_strides);
  generate_yolox_proposals(grid_strides, prob, BBOX_CONF_THRESH, proposals);
  qsort_descent_inplace(proposals);

  std::vector<int> picked;

  // for (int i=0;i<proposals.size();i++){

  //   std::cout << "proposals:"<< proposals[i].prob << std::endl;

  // }
  // std::cout << "picked:" << picked.size() << std::endl;

  nms_sorted_bboxes(proposals, picked, NMS_THRESH);
  
  int count = picked.size();
  objects.resize(count);

  for (int i = 0; i < count; i++) {
    objects[i] = proposals[picked[i]];

    // adjust offset to original unpadded
    float x0 = (objects[i].rect.x) / scale;
    float y0 = (objects[i].rect.y) / scale;
    float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
    float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

    // clip
    x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
    y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
    x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
    y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

    objects[i].rect.x = x0;
    objects[i].rect.y = y0;
    objects[i].rect.width = x1 - x0;
    objects[i].rect.height = y1 - y0;
  }
}

const float color_list[1][3] = {
    {0.000, 0.447, 0.741}};

static void draw_objects(const cv::Mat &bgr,
                         const std::vector<Object> &objects) {
  static const char *class_names[] = {"person"};

  cv::Mat image = bgr.clone();

  for (size_t i = 0; i < objects.size(); i++) {
    const Object &obj = objects[i];

    //fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
    //        obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

    cv::Scalar color =
        cv::Scalar(color_list[obj.label][0], color_list[obj.label][1],
                   color_list[obj.label][2]);
    float c_mean = cv::mean(color)[0];
    cv::Scalar txt_color;
    if (c_mean > 0.5) {
      txt_color = cv::Scalar(0, 0, 0);
    } else {
      txt_color = cv::Scalar(255, 255, 255);
    }

    cv::rectangle(image, obj.rect, color * 255, 2);

    char text[256];
    sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

    int baseLine = 0;
    cv::Size label_size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

    cv::Scalar txt_bk_color = color * 0.7 * 255;

    int x = obj.rect.x;
    int y = obj.rect.y + 1;
    // int y = obj.rect.y - label_size.height - baseLine;
    if (y > image.rows)
      y = image.rows;
    // if (x + label_size.width > image.cols)
    // x = image.cols - label_size.width;

    cv::rectangle(
        image,
        cv::Rect(cv::Point(x, y),
                 cv::Size(label_size.width, label_size.height + baseLine)),
        txt_bk_color, -1);

    cv::putText(image, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
  }

  cv::imwrite("out.jpg", image);

  if (SHOW_IMG)
  {
  cv::namedWindow("Display Image",cv::WINDOW_AUTOSIZE); 
  cv::imshow("Display Image", image); 
  cv::waitKey(0); 
  }
  std::cout << "save output to out.jpg" << std::endl;
}

class YOLOX_ONNX
{
public:
    const char* model_path;
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::Session session;


    std::vector<const char*> inputNames{"images"};
    std::vector<const char*> outputNames{"output"};
    std::vector<int64_t> inputDims = {1,3,416,416};
    size_t inputTensorSize = vectorProduct(inputDims);
    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;


    std::vector<int64_t> outputDims = {1,3549,6};//(batch_size,num_boxes,num_classes+5)
    size_t outputTensorSize = vectorProduct(outputDims);

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    const float* tensor_data;

    YOLOX_ONNX(const char* model_path)
        : env(ORT_LOGGING_LEVEL_WARNING, "YOLOXONNX"), model_path(model_path), session(env, model_path, session_options)
    {

        std::cout << "ONNX model loaded successfully from: " << model_path << std::endl;
    }


    void predict(cv::Mat preprocessed_img){

        // Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
        // auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        // ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
        
        std::vector<float> inputTensorValues(inputTensorSize);
        std::cout << "inputTensorSize: " << inputTensorSize << std::endl;
        // Make copies of the same image input.
        for (int64_t i = 0; i < batchSize; ++i)
        {
            std::copy(preprocessed_img.begin<float>(),
                    preprocessed_img.end<float>(),
                    inputTensorValues.begin() + i * inputTensorSize / batchSize);
        }
        


        //for (long int i : inputDims){
        //  std::cout << "inputDim" << i << std::endl;
        //}
        
        

        
        std::vector<float> outputTensorValues(outputTensorSize);

        
        std::cout << "outputTensorSize: " << outputTensorSize << std::endl;


        

        
        //std::cout << "Here before inference " << std::endl;


        
        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
            inputDims.size()));
        outputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, outputTensorValues.data(), outputTensorSize,
            outputDims.data(), outputDims.size()));

        
        
        auto start = std::chrono::high_resolution_clock::now();
        
        session.Run(Ort::RunOptions{}, inputNames.data(),
                        inputTensors.data(), 1, outputNames.data(),
                        outputTensors.data(), 1);
        
        auto stop = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        
        std::cout << "Execution Time: " << duration.count() << " ms" << std::endl;

        tensor_data = outputTensors[0].GetTensorData<float>();



    }
   
};

int main(int argc, char** argv) 
{
    if (argc != 3) { 
        std::cout<<"usage: ./yolox-onnx <Model_Path> <Image_Path>"<<std::endl; 
        return -1; 
    } 

    const char* img_path = argv[2];
    const char* model_path = argv[1];

    cv::Mat image = cv::imread(img_path);
    int img_w = image.cols;
    int img_h = image.rows;
    cv::Mat resized_img = static_resize(image);
    cv::Mat preprocessed_img;
    //channnels first
    cv::dnn::blobFromImage(resized_img, preprocessed_img);

    
    YOLOX_ONNX yolox_onnx(model_path);
    yolox_onnx.predict(preprocessed_img);

    float scale =std::min(INPUT_W / (image.cols * 1.0), INPUT_H / (image.rows * 1.0));
    std::vector<Object> objects;

    decode_outputs(yolox_onnx.tensor_data, objects, scale, img_w, img_h);

    draw_objects(image, objects);
 
    return 0;
}
