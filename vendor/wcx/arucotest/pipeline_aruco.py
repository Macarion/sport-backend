"""Required helpers copied from the existing Web algorithm; source hashes in docs/jump-source-manifest.json.
Unused standalone CLI/model loaders are intentionally omitted.
"""


def write_result_file(result_path, measurement, image_path, mask_path, calib_image_path):
    heel = measurement.heel_result
    with open(result_path, "w", encoding="utf-8") as file:
        file.write("ArUco 无垫子立定跳远测距结果\n")
        file.write("=" * 40 + "\n")
        file.write(f"落地帧图像: {image_path}\n")
        file.write(f"人物 mask: {mask_path}\n")
        file.write(f"标定图像: {calib_image_path or image_path}\n")
        file.write(f"mask 像素值: {list(measurement.mask_unique_values)}\n")
        file.write(f"人物轮廓数量: {measurement.person_contour_count}\n")
        file.write("\n")
        file.write(f"跳远成绩: {measurement.score_cm:.2f} cm\n")
        file.write(
            "脚后跟图像坐标: "
            f"({heel.image_point[0]:.2f}, {heel.image_point[1]:.2f})\n"
        )
        file.write(
            "脚后跟平面坐标: "
            f"x={heel.world_point[0]:.2f} cm, y={heel.world_point[1]:.2f} cm\n"
        )
        file.write(f"脚后跟候选策略: {heel.candidate_strategy}\n")
        file.write(f"脚底接触带候选点数: {heel.contact_candidate_count}\n")
        file.write("\n")
        file.write(f"ArUco debug 图: {measurement.aruco_debug_path}\n")
        file.write(f"测距可视化图: {measurement.visualization_path}\n")
