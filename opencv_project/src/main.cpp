#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

#ifdef OUTPUT_DIR
    const string outputDir = OUTPUT_DIR; // 使用 CMake 定义的输出目录
#else
    const string outputDir = "output"; // 默认输出目录
#endif

void saveImageToOutputFolder(const Mat& img, const string& baseFileName) {
    // 确保输出目录存在
    if (!fs::exists(outputDir)) {
        fs::create_directory(outputDir);
    }

    // 自动添加扩展名，例如使用 .png
    string fileNameWithExt = baseFileName + ".png"; // 或者 ".jpg"
    string fullPath = outputDir + "/" + fileNameWithExt;
    
    bool success = imwrite(fullPath, img);
    if (success) {
        cout << "图像已成功保存至: " << fullPath << endl;
    } else {
        cout << "错误：无法保存图像至: " << fullPath << endl;
        // 这个错误信息会帮助你定位问题
    }
}

int main() {
    
 //-----------------------TASK_1:实现基础图像处理操作，需要对于任务给定的图片进行以下操作------------------------------//
    
    // 1. 读取图像
    Mat originalImage = imread("/home/coordsys/zbx/opencv_project/resources/test_image.png");
    if (originalImage.empty()) {
        cout << "错误：无法读取图像，请检查路径！" << endl;
        return -1;
    }
    saveImageToOutputFolder(originalImage, "TASK_1原始图像");

    // --- 分离任务：创建副本用于绘制 ---
    Mat imageForDrawing = originalImage.clone(); // 创建原始图像的副本
    // --- 图像处理和特征提取使用原始图像 ---
    // 图像颜色空间转换
    Mat grayImage, hsvImage;
    cvtColor(originalImage, grayImage, COLOR_BGR2GRAY); // 使用 originalImage
    saveImageToOutputFolder(grayImage, "TASK_1灰度图");
    cvtColor(originalImage, hsvImage, COLOR_BGR2HSV); // 使用 originalImage
    saveImageToOutputFolder(hsvImage, "TASK_1hsv图");

    // 应用各种滤波操作
    Mat meanFiltered, gaussianFiltered;
    blur(originalImage, meanFiltered, Size(5, 5));
    saveImageToOutputFolder(meanFiltered, "TASK_1均值滤波");
    GaussianBlur(originalImage, gaussianFiltered, Size(5, 5), 0);
    saveImageToOutputFolder(gaussianFiltered, "TASK_1高斯滤波");

    // 提取红色区域
    vector<Scalar> lowerRed = {Scalar(0, 80, 80), Scalar(150, 80, 80)}; // 放宽后的下限
    vector<Scalar> upperRed = {Scalar(20, 255, 255), Scalar(180, 255, 255)}; // 放宽后的上限

    Mat mask1, mask2;
    inRange(hsvImage, lowerRed[0], upperRed[0], mask1);
    inRange(hsvImage, lowerRed[1], upperRed[1], mask2);

    Mat mask = mask1 + mask2; // 合并掩膜
    saveImageToOutputFolder(mask, "TASK_1提取红色区域的掩膜");

    // 寻找红色区域的轮廓
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 在副本上绘制圆形、方形和文字
    circle(imageForDrawing, Point(300, 300), 100, Scalar(0, 255, 0), 2); // 绿色圆轮廓
    rectangle(imageForDrawing, Point(400, 100), Point(600, 300), Scalar(255, 0, 0), 2); // 蓝色方形轮廓
    putText(imageForDrawing, "Hello OpenCV", Point(100, 100), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 2); // 红色文字
    saveImageToOutputFolder(imageForDrawing, "TASK_1绘制图形后的图像");

    Mat imageWithContours = originalImage.clone(); // 再次使用原始图像作为基础
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > 100) { // 忽略小面积区域
            Rect boundingBox = boundingRect(contours[i]);
            rectangle(imageWithContours, boundingBox.tl(), boundingBox.br(), Scalar(0, 255, 0), 2); // 绘制绿色bounding box
            drawContours(imageWithContours, contours, i, Scalar(255, 0, 0), 2); // 蓝色外轮廓
            cout << "轮廓 " << i + 1 << " 的面积: " << area << endl;
        }
    }
    saveImageToOutputFolder(imageWithContours, "TASK_1带有红色区域轮廓的图像"); // 显示在原始图像上的检测结果

    // 提取高亮颜色区域并对其进行形态学处理（这里是灰度图像与二值图像）
    // 分离HSV通道
    vector<Mat> channels;
    split(hsvImage, channels);
    Mat vChannel = channels[2];
    Mat binaryVChannel;
    vChannel.copyTo(binaryVChannel);

    // 对V通道的副本进行二值化处理，提取高亮区域
    threshold(binaryVChannel, binaryVChannel, 0, 255, THRESH_BINARY + THRESH_OTSU);

    // 显示原始的V通道（灰度图像）
    saveImageToOutputFolder(vChannel, "TASK_1V通道灰度图像");

    // 显示二值化的V通道（二值图像）
    saveImageToOutputFolder(binaryVChannel, "TASK_1V通道二值化图像");
    
    // 腐蚀
    Mat erodedImage;
    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));// 定义结构元素大小，例如 5x5
    erode(binaryVChannel, erodedImage, element); // 执行腐蚀
    saveImageToOutputFolder(erodedImage, "TASK_1腐蚀后的图像"); // 保存腐蚀后的图像
    
    // 膨胀（这里指的是先腐蚀后膨胀的操作，即为开操作）
    Mat dilatedImage;
    dilate(erodedImage, dilatedImage, element);// 使用相同的结构元素
    saveImageToOutputFolder(dilatedImage, "TASK_1先腐蚀后膨胀的图像");// 保存膨胀后的图像

    // 漫水
    Mat floodFilled = dilatedImage.clone();
    Point seedPoint(100, 100); 
    Scalar newVal(255); // 填充为白色
    Scalar loDiff(0), upDiff(0); // 在二值图像中，颜色差异设为 0（因为我们只关心连通性）
    int flags = 4; // 4-连通 + 填充值为 255（如果使用掩膜，可以设置 (255 << 8)）
    floodFill(floodFilled, seedPoint, newVal, nullptr, loDiff, upDiff, flags);// 执行漫水填充
    saveImageToOutputFolder(floodFilled, "TASK_1漫水填充后的图像");
    
    // 图像旋转 35 度
    Point2f centerRot(originalImage.cols / 2.0F, originalImage.rows / 2.0F);
    Mat rotMat = getRotationMatrix2D(centerRot, 35, 1.0);
    Mat rotatedImage;
    warpAffine(originalImage, rotatedImage, rotMat, originalImage.size()); // 使用 originalImage
    saveImageToOutputFolder(rotatedImage, "TASK_1旋转后的图像");

    // 图像裁剪为原图的左上角 1/4
    Mat croppedImage = originalImage(Rect(0, 0, originalImage.cols / 2, originalImage.rows / 2)); // 使用 originalImage
    saveImageToOutputFolder(croppedImage, "TASK_1裁剪后的图像");

 //-----------------------TASK_2:实际应用：识别出给定图片二的装甲板区域------------------------------//
    
    // 1. 读取图像
    Mat originalImage2 = imread("/home/coordsys/zbx/opencv_project/resources/test_image_2.png");
    if (originalImage2.empty()) {
        cout << "错误：无法读取图像，请检查路径！" << endl;
        return -1;
    }
    saveImageToOutputFolder(originalImage2, "TASK_2原始图像");

    Mat imageForDrawing2 = originalImage2.clone(); 

    // Mat hsvImage2,meanFiltered2;
    // cvtColor(originalImage2, hsvImage2, COLOR_BGR2HSV); // 使用 originalImage2
    // saveImageToOutputFolder(hsvImage2, "hsv图2");

    // blur(hsvImage2, meanFiltered2, Size(5, 5));
    // saveImageToOutputFolder(meanFiltered2, "hsv图进行均值滤波后2"); 
        
    // vector<Mat> channels2;
    // split(meanFiltered2, channels2);
    // Mat vChannel2 = channels2[2];
    // Mat binaryVChannel2;
    // vChannel2.copyTo(binaryVChannel2);

    // // 对V通道的副本进行二值化处理，提取高亮区域
    // threshold(binaryVChannel2, binaryVChannel2, 250, 255, THRESH_BINARY);
 
    // saveImageToOutputFolder(binaryVChannel2, "hsv图进行均值滤波后V通道二值化图像2");
    // 这一段当时是采取hsv思路来解决，但是似乎并不能实现，但是还是决定保留这段代码。

    Mat grayImage2;
    cvtColor(originalImage2, grayImage2, COLOR_BGR2GRAY); // 使用 originalImage
    saveImageToOutputFolder(grayImage2, "TASK_2灰度图");

    Mat meanFiltered2;
    blur(grayImage2, meanFiltered2, Size(5, 5));
    saveImageToOutputFolder(meanFiltered2, "TASK_2灰度图均值滤波");

    double thresholdValue = 225;  // 阈值
    double maxValue = 255;        // 最大值（超过阈值时设置为此值）
    Mat binaryImage2;
    threshold(grayImage2, binaryImage2, thresholdValue, maxValue, cv::THRESH_BINARY);
    saveImageToOutputFolder(binaryImage2, "TASK_2灰度图二值化结果");

    Mat erodedImage2;
    erode(binaryImage2, erodedImage2, element); // 执行腐蚀
    saveImageToOutputFolder(erodedImage2, "TASK_2腐蚀后的图像"); // 保存腐蚀后的图像
    
    // 膨胀（这里指的是先腐蚀后膨胀的操作，即为开操作）
    Mat dilatedImage2;
    dilate(erodedImage2, dilatedImage2, element);// 使用相同的结构元素
    saveImageToOutputFolder(dilatedImage2, "TASK_2先腐蚀后膨胀的图像");// 保存膨胀后的图像

    // --- 检测白色区域，绘制外接矩形，并判断长宽比与填充比例是否符合要求。

    Mat resultImage2 = originalImage2.clone(); // 用于绘制结果（彩色图上画框）
    Mat temp = dilatedImage2.clone();          // findContours 会修改原图

    std::vector<cv::Vec4i> hierarchy;

    // 1. 查找所有轮廓
    cv::findContours(temp, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 2. 遍历每个轮廓
    for (size_t i = 0; i < contours.size(); ++i) {
        // 计算轮廓面积
        double contourArea = cv::contourArea(contours[i]);
        
        // 过滤太小的区域
        if (contourArea < 50) {
            continue;
        }

        // 获取最小外接旋转矩形
        cv::RotatedRect rotatedRect = cv::minAreaRect(contours[i]);
        double rectArea = rotatedRect.size.width * rotatedRect.size.height;

        // 计算“填充比例” = 轮廓面积 / 外接矩形面积
        double fillRatio = contourArea / rectArea;

        // 计算长宽比（保证 >= 1）
        double width = rotatedRect.size.width;
        double height = rotatedRect.size.height;
        double aspectRatio = std::max(width, height) / std::min(width, height);

        // 输出信息
        std::cout << "区域 " << i 
                << "：轮廓面积=" << contourArea 
                << ", 矩形面积=" << rectArea 
                << ", 填充比例=" << std::fixed << std::setprecision(2) << fillRatio
                << ", 长宽比=" << aspectRatio << std::endl;

        // 🔍 综合判断条件（更鲁棒！）
        if (
            aspectRatio > 3.0 && aspectRatio < 3.5 &&  // 细长条
            fillRatio > 0.8 && fillRatio <= 1.0 &&     // 实心程度高（不是空壳或散点）
            contourArea > 100                            // 面积不能太小
        ) {
            // 绘制旋转矩形
            cv::Point2f rectPoints[4];
            rotatedRect.points(rectPoints);
            for (int j = 0; j < 4; ++j) {
                cv::line(resultImage2, rectPoints[j], rectPoints[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
            }

            // 标注文本
            std::ostringstream label;
            label << "AR:" << std::setprecision(2) << aspectRatio 
                << " FR:" << fillRatio;
            cv::putText(resultImage2, label.str(),
                        cv::Point(rotatedRect.center.x - 40, rotatedRect.center.y - 15),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

            // 可选：绘制中心点
            cv::circle(resultImage2, rotatedRect.center, 3, cv::Scalar(0, 0, 255), -1);
        }
        else {
            // 可选：用浅色标出未通过筛选的区域（如灰色）
            cv::Point2f rectPoints[4];
            rotatedRect.points(rectPoints);
            for (int j = 0; j < 4; ++j) {
                cv::line(resultImage2, rectPoints[j], rectPoints[(j + 1) % 4], cv::Scalar(128, 128, 128), 1);
            }
        }
    }

    // 保存结果
    saveImageToOutputFolder(resultImage2, "TASK_2最终检测结果_带旋转矩形框_增强判断");
 //-------------------------------------------------------------------------------------------//
    destroyAllWindows();
    return 0;
}