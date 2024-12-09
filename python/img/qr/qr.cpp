#include <iostream>
#include <numeric>
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

double compute_iou(const vector<Point> &box1, const vector<Point> &box2)
{
    // 计算矩形的凸包
    vector<Point> poly1, poly2;
    convexHull(box1, poly1);
    convexHull(box2, poly2);

    // 获取交集
    vector<Point> intersection;
    double intersection_area = 0.0;
    if (intersectConvexConvex(poly1, poly2, intersection) > 0)
    {
        // 交集区域的面积
        intersection_area = contourArea(intersection);
    }

    // 计算两个矩形的面积
    double area1 = contourArea(poly1);
    double area2 = contourArea(poly2);

    // 计算并集的面积（并集面积 = area1 + area2 - intersection_area）
    double union_area = area1 + area2 - intersection_area;

    // 计算IoU
    double iou = intersection_area / union_area;
    return iou;
}

// 扩展矩形的顶点
vector<Point> expandRectPoints(const vector<Point> &points, int expand_pixels)
{
    vector<Point> expanded_points = points;
    // Calculate center of the rectangle
    Point center(0, 0);
    for (const auto &p : points)
    {
        center += p;
    }
    center /= static_cast<float>(points.size());

    // Expand each point
    for (size_t i = 0; i < points.size(); ++i)
    {
        Point direction = points[i] - center;
        direction *= (expand_pixels / norm(direction));
        expanded_points[i] += direction;
    }

    return expanded_points;
}

vector<vector<Point>> filter_boxes(const vector<vector<Point>> &boxes)
{
    vector<vector<Point>> filtered_boxes;
    for (const auto &box : boxes)
    {
        bool to_add = true;
        for (const auto &box0 : filtered_boxes)
        {
            if (compute_iou(box0, box) > 0.1)
            {
                to_add = false;
                break;
            }
        }
        if (to_add)
        {
            filtered_boxes.push_back(box);
        }
    }
    return filtered_boxes;
}

vector<Point> getRotatedRectBox(const vector<Point>& contour) {
    RotatedRect rect = minAreaRect(contour);
    vector<Point2f> box(4);
    rect.points(box.data());

    vector<Point> box_points;
    for (const auto& pt : box) {
        box_points.push_back(Point(pt.x, pt.y));
    }
    return box_points;
}

vector<vector<Point>> merge_contours(const vector<vector<Point>> &square_merged_contours)
{
    int n_contours = square_merged_contours.size();
    UnionFind uf(n_contours);
    vector<vector<double>> distances(n_contours, vector<double>(n_contours, 0.0));
    for (int i = 0; i < n_contours; ++i)
    {
        for (int j = i + 1; j < n_contours; ++j)
        {
            distances[i][j] = closestDistance(square_merged_contours[i], square_merged_contours[j]);
            distances[j][i] = distances[i][j];
        }
    }
    vector<double> distances_flat;
    for (int i = 0; i < n_contours; i++)
    {
        for (int j = i + 1; j < n_contours; j++)
        {
            distances_flat.push_back(distances[i][j]);
        }
    }
    double threshold = 15.0;
    if (distances_flat.size() > 24)
    {
        partial_sort(distances_flat.begin(), distances_flat.begin() + 24, distances_flat.end());
        double average_min_distance = std::accumulate(distances_flat.begin(), distances_flat.begin() + 24, 0.0) / 24.0;
        double min_distance = distances_flat[0];
        threshold = average_min_distance * 2.5;
    }

    for (int i = 0; i < n_contours; ++i)
    {
        for (int j = i + 1; j < n_contours; ++j)
        {
            if (distances[i][j] < threshold)
            {
                uf.union_set(i, j);
            }
        }
    }

    unordered_map<int, vector<int>> clusters;
    for (int i = 0; i < n_contours; ++i)
    {
        int root = uf.find(i);
        clusters[root].push_back(i);
    }

    vector<vector<Point>> boxs;
    for (const auto &cluster : clusters)
    {
        const vector<int> &indices = cluster.second;
        vector<vector<Point>> contours_to_merge;
        for (int idx : indices)
        {
            contours_to_merge.push_back(square_merged_contours[idx]);
        }

        vector<Point> points;
        for (const auto &contour : contours_to_merge)
        {
            points.insert(points.end(), contour.begin(), contour.end());
        }
        vector<Point> box_points = getRotatedRectBox(points);
        double area = contourArea(box_points);
        if (area < 80 * 80 * 3 && indices.size() < 3)
        {
            continue;
        }

        bool flag = false;
        for (int idx : indices)
        {
            vector<Point> point0 = square_merged_contours[idx];
            vector<Point> box_points0 = getRotatedRectBox(point0);;
            double area0 = contourArea(box_points0);
            if (area0 > 10000)
            {
                boxs.push_back(box_points0);
                flag = true;
            }
        }

        if (!flag)
        {
            boxs.push_back(box_points);
        }
    }

    return filter_boxes(boxs);
}

void mainFunc(const string &image_path)
{
    Mat image = imread(image_path);
    if (image.empty())
    {
        cerr << "Error: Could not read the image." << endl;
        return;
    }

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    Mat edges;
    Canny(gray, edges, 150, 200);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<vector<Point>> hulls;
    for (size_t i = 0; i < contours.size(); ++i)
    {
        vector<cv::Point> hull;
        convexHull(contours[i], hull);
        hulls.push_back(hull);
    }
    contours = hulls;
    vector<vector<Point>> merged_contours;
    for (const auto &contour : contours)
    {
        if (contourArea(contour) < 100)
        {
            continue;
        }
        double perimeter = arcLength(contour, true);
        double epsilon = 0.02 * perimeter;
        vector<Point> approx;
        approxPolyDP(contour, approx, epsilon, true);
        if (approx.size() == 4)
        {
            merged_contours.push_back(approx);
        }
    }
    vector<vector<Point>> square_merged_contours;
    for (const auto &contour : merged_contours)
    {
        int max_x = contour[0].x;
        int min_x = contour[0].x;
        int max_y = contour[0].y;
        int min_y = contour[0].y;
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
        if (abs(max_x - min_x - max_y + min_y) < 100)
        {
            square_merged_contours.push_back(contour);
        }
    }

    vector<vector<Point>> boxs = merge_contours(square_merged_contours);
    int id = 0;
    for (const auto &box : boxs)
    {
        vector<Point> expanded_box = expandRectPoints(box, 7);
        Mat mask = Mat::zeros(gray.size(), CV_8UC1);
        vector<vector<Point>> poly{expanded_box};
        fillPoly(mask, poly, Scalar(255));
        Mat result_image = Mat::zeros(image.size(), image.type());
        image.copyTo(result_image, mask);
        id++;
        string filename = "./mask/" + to_string(id) + ".jpg";
        imwrite(filename, result_image);
    }
}
// g++ -o qr_exe qr.cpp `pkg-config --cflags --libs opencv`
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
