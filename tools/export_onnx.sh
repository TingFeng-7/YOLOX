python /workspace/bohuang/git_res/YOLOX/tools/export_onnx.py -f exp.py -c YOLOX_outputs/exp/latest_ckpt.pth --output-name model.onnx --dynamic
python /workspace/bohuang/git_res/YOLOX/demo/ONNXRuntime/onnx_inference.py --model model.onnx \
    -i /workspace/bohuang/data/sa_v1/sa_format/0006_word_07pro/imgs/1607500755_img.png\
    --input_shape "1088,1920"