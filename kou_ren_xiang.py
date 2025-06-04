import os
import cv2
import paddlehub as hub
import paddle

# def batch_human_segmentation(input_dir, output_dir):
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#     # 加载预训练人像分割模型
#     model = hub.Module(name='deeplabv3p_xception65_humanseg')
#
#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 批量处理图像
#     for filename in os.listdir(input_dir):
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             img_path = os.path.join(input_dir, filename)
#             result = model.segmentation(paths=[img_path], visualization=True, output_dir=output_dir, batch_size=32, use_gpu=True)
#
#             # 重命名输出文件（原文件名+_alpha.png）
#             orig_name = os.path.splitext(filename)[0]
#             # os.rename(
#             #     os.path.join(output_dir, f'{orig_name}_alpha.png'),
#             #     os.path.join(output_dir, f'{orig_name}.png')
#             # )
#     paddle.device.cuda.empty_cache()
#
# # 使用示例
# batch_human_segmentation(
#     input_dir='D:/code/videodetectron/data/TIC/train/4',
#     output_dir='D:/code/videodetectron/data/TIC/train/4_process'
# )


def person_bbox_crop(input_dir, output_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # 加载轻量级人脸检测模型
    detector = hub.Module(name="ultra_light_fast_generic_face_detector_1mb_640")
    # detector = hub.Module(name="pyramidbox_face_detection")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(input_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, img_name)
            frame = cv2.imread(img_path)

            # 检测人物区域（此处以人脸为例，可替换为全身检测模型）
            results = detector.face_detection(
                images=[frame],
                use_gpu=True,
                visualization=False,
                confs_threshold=0.9
            )
            # results = detector.face_detection(
            #     images=[frame],
            #     use_gpu=True,
            #     visualization=False,
            #     score_thresh=0.9
            # )

            # 提取边界框并裁剪
            for i, result in enumerate(results[0]['data']):
                left = int(result['left'])
                top = int(result['top'])
                right = int(result['right'])
                bottom = int(result['bottom'])

                # 扩展检测框至全身（需根据实际比例调整）
                height = bottom - top
                expanded_bottom = min(bottom + 2 * height, frame.shape[0])
                person_roi = frame[top:bottom, left:right]

                # 保存裁剪后图像
                output_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}.jpg")
                cv2.imwrite(output_path, person_roi)


# 调用示例
person_bbox_crop(
    input_dir='D:/code/videodetectron/data/TIC/train/7',
    output_dir='D:/code/videodetectron/data/TIC/train/7_process'
)