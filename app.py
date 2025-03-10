from yolox_onnx import YOLOX_ONNX
import albumentations as A
import gradio as gr
import cv2

def show_example(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def get_response(input_img,add_snow,add_rain,add_fog,confidence_threshold,iou_threshold):

    '''
    detects all possible pedestrians in the image and recognizes it
        Args:
            input_img (numpy array): one image of type numpy array
            add_snow (boolean) : apply random snow augmentation
            add_rain (boolean) : apply random rain augmentation
            add_fog (boolean) : apply random fog augmentation
            confidence_threshold (float) : minimum confidence prob required for bounding box candidate
            iou_threshold (float)  : intersection threshold above which bounding box will be neglected

        Returns:
            return img(numpy array): image with bounding boxes of pedestrians
    '''


    if not hasattr(input_img,'shape'):
        return "invalid input",input_img

    pedestrian_detector.predict(input_img,confidence_threshold,iou_threshold)
    out_img=pedestrian_detector.output_img

    if add_snow:
        out_img = weather_transform["add_snow"] (image=out_img)['image']
    elif add_rain:
        out_img = weather_transform["add_rain"](image=out_img)['image']
    elif add_fog:
        out_img = weather_transform["add_fog"](image=out_img)['image']
    else:
        pass


    return out_img


if __name__ == "__main__":
    weather_transform = {}
    weather_transform["add_rain"] = A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1)
    weather_transform["add_snow"] = A.RandomSnow(brightness_coeff=2.5, snow_point_lower=0.3, snow_point_upper=0.5, p=1)
    weather_transform["add_fog"] = A.RandomFog(fog_coef_lower=0.7, fog_coef_upper=0.8, alpha_coef=0.1, p=1)


    pedestrian_detector=YOLOX_ONNX('models/pedestrian-detection-best55.onnx')
    iface = gr.Interface(
        fn=get_response,
        inputs=[gr.Image(type="numpy"),  # Accepts image input
        gr.Checkbox(label="add_snow"),
        gr.Checkbox(label="add_rain"),
        gr.Checkbox(label="add_fog"),
        gr.Slider(0, 1,value=0.5, step=0.01, label="confidence_threshold"),
        gr.Slider(0, 1,value=0.45, step=0.01, label="iou_threshold")],
        examples=[[show_example('test1.jpg')],[show_example('test2.jpg')],[show_example('test3.jpg')]],
        outputs=[gr.Image(type="numpy")],
        title="Pedestrian Detection with All weather augmentation",
        description="Upload images for pedestrian detection")

    iface.launch(share=True)