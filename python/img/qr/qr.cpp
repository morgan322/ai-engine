#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class UnionFind
{
private:
    vector<int> parent;
    vector<int> rank;

public:
    UnionFind(int n)
    {
        parent.resize(n);
        rank.resize(n, 1);
        for (int i = 0; i < n; ++i)
        {
            parent[i] = i;
        }
    }

    int find(int u)
    {
        if (parent[u] != u)
        {
            parent[u] = find(parent[u]);
        }
        return parent[u];
    }

    void union_set(int u, int v)
    {
        int root_u = find(u);
        int root_v = find(v);

        if (root_u != root_v)
        {
            if (rank[root_u] > rank[root_v])
            {
                parent[root_v] = root_u;
            }
            else if (rank[root_u] < rank[root_v])
            {
                parent[root_u] = root_v;
            }
            else
            {
                parent[root_v] = root_u;
                rank[root_u]++;
            }
        }
    }
};

// 计算两个轮廓之间的最小距离
double closestDistance(const vector<Point> &contour1, const vector<Point> &contour2)
{
    double min_dist = DBL_MAX;
    for (const auto &pt1 : contour1)
    {
        for (const auto &pt2 : contour2)
        {
            double dist = norm(pt1 - pt2);
            if (dist < min_dist)
            {
                min_dist = dist;
            }
        }
    }
    return min_dist;
}

// 扩展矩形的顶点
vector<Point> expandRectPoints(const vector<Point>& points, int expand_pixels) {
    vector<Point> expanded_points = points;
    // Calculate center of the rectangle
    Point center(0, 0);
    for (const auto& p : points) {
        center += p;
    }
    center /= static_cast<float>(points.size());

    // Expand each point
    for (size_t i = 0; i < points.size(); ++i) {
        Point direction = points[i] - center;
        direction *= (expand_pixels / norm(direction));
        expanded_points[i] += direction;
    }

    return expanded_points;
}

void mainFunc(const string &image_path)
{
    // 读取图像
    Mat image = imread(image_path);
    if (image.empty())
    {
        cerr << "Error: Could not read the image." << endl;
        return;
    }

    // 转换为灰度图像
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // 使用Canny边缘检测
    Mat edges;
    Canny(gray, edges, 100, 200);

    // 查找图像中的轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // // 筛选并排序轮廓（面积大于100）
    // vector<vector<Point>> filtered_contours;
    // for (const auto &contour : contours)
    // {
    //     if (contourArea(contour) > 100)
    //     {
    //         filtered_contours.push_back(contour);
    //     }
    // }
    // sort(filtered_contours.begin(), filtered_contours.end(), [](const vector<Point> &a, const vector<Point> &b)
    //      { return contourArea(a) > contourArea(b); });

    // 存储合并后的轮廓
    vector<vector<Point>> merged_contours;

    // 处理每个轮廓
    for (const auto &contour : contours)
    {
        if (contourArea(contour) < 100)
        {
            continue;
        }
        // 近似轮廓
        double perimeter = arcLength(contour, true);
        double epsilon = 0.02 * perimeter;
        vector<Point> approx;
        approxPolyDP(contour, approx, epsilon, true);

        // 如果近似轮廓有四个顶点，则考虑合并
        if (approx.size() == 4)
        {
            // // 合并接近的轮廓
            // if (!merged_contours.empty())
            // {
            //     Moments M1 = moments(approx);
            //     Point2f center1(M1.m10 / M1.m00, M1.m01 / M1.m00);

            //     Moments M2 = moments(merged_contours.back());
            //     Point2f center2(M2.m10 / M2.m00, M2.m01 / M2.m00);

            //     float distance = sqrt(pow(center1.x - center2.x, 2) + pow(center1.y - center2.y, 2));

            //     // Adjust this threshold based on your image scale
            //     if (distance < 50)
            //     {
            //         // Concatenate 'approx' with the last contour in 'merged_contours'
            //         vector<Point> merged_points;
            //         merged_points.reserve(merged_contours.back().size() + approx.size());
            //         merged_points.insert(merged_points.end(), merged_contours.back().begin(), merged_contours.back().end());
            //         merged_points.insert(merged_points.end(), approx.begin(), approx.end());

            //         // Update the last contour in 'merged_contours'
            //         merged_contours.back() = merged_points;
            //     }
            //     else
            //     {
            //         merged_contours.push_back(approx);
            //     }
            // }
            // else
            // {
            merged_contours.push_back(approx);
            // }
        }
    }
    vector<vector<Point>> square_merged_contours;

    for (const auto &contour : merged_contours)
    {
        int max_x = contour[0].x;
        int min_x = contour[0].x;
        int max_y = contour[0].y;
        int min_y = contour[0].y;

        // Find the bounding box coordinates of the contour
        for (const auto &point : contour)
        {
            if (point.x > max_x)
                max_x = point.x;
            if (point.x < min_x)
                min_x = point.x;
            if (point.y > max_y)
                max_y = point.y;
            if (point.y < min_y)
                min_y = point.y;
        }

        // Check if the contour forms a nearly square shape
        if (abs(max_x - min_x - max_y + min_y) < 100)
        {
            square_merged_contours.push_back(contour);
        }
    }

    // 创建并查集实例
    int n_contours = square_merged_contours.size();
    UnionFind uf(n_contours);

    // 构建轮廓之间的距离矩阵
    vector<vector<double>> distances(n_contours, vector<double>(n_contours, 0.0));
    for (int i = 0; i < n_contours; ++i)
    {
        for (int j = i + 1; j < n_contours; ++j)
        {
            distances[i][j] = closestDistance(square_merged_contours[i], square_merged_contours[j]);
            distances[j][i] = distances[i][j];
        }
    }

    // 合并轮廓成团
    for (int i = 0; i < n_contours; ++i)
    {
        for (int j = i + 1; j < n_contours; ++j)
        {
            if (distances[i][j] < 20.0)
            { // 根据距离阈值合并
                uf.union_set(i, j);
            }
        }
    }

    // 输出结果
    unordered_map<int, vector<int>> clusters;
    for (int i = 0; i < n_contours; ++i)
    {
        int root = uf.find(i);
        clusters[root].push_back(i);
    }

    // 使用OpenCV绘制最小旋转外接矩形
    for (const auto &kv : clusters)
    {
        vector<vector<Point>> contours_to_merge;
        for (int idx : kv.second)
        {
            contours_to_merge.push_back(square_merged_contours[idx]);
        }

        // 计算最小外接矩形
        vector<Point> points;
        for (const auto &contour : contours_to_merge)
        {
            points.insert(points.end(), contour.begin(), contour.end());
        }
        RotatedRect rect = minAreaRect(points);
     
        // 获取最小外接矩形的四个顶点
        vector<Point2f> box(4);
        rect.points(box.data());

        // 转换box为Point类型
        vector<Point> box_points;
        for (const auto &pt : box)
        {
            box_points.push_back(Point(pt.x, pt.y));
        }
       
        // 扩展矩形顶点
        vector<Point> expanded_box = expandRectPoints(box_points, 7);
     
        // 创建与原始图像大小相同的掩模
        Mat mask = Mat::zeros(gray.size(), CV_8UC1);

        // 在掩模上绘制扩展的矩形区域（填充白色，255）
        vector<vector<Point>> poly{expanded_box};
        fillPoly(mask, poly, Scalar(255));

        // 创建结果图像
        Mat result_image = Mat::zeros(image.size(), image.type());
        image.copyTo(result_image, mask);

        // 保存结果图像
        imwrite("./mask/" + to_string(kv.first) + ".jpg", result_image);
    }
}
//g++ -o qr_exe qr.cpp `pkg-config --cflags --libs opencv`
int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cerr << "Usage: " << argv[0] << " <image_path>" << endl;
        return 1;
    }

    string image_path = argv[1];
    mainFunc(image_path);

    return 0;
}
