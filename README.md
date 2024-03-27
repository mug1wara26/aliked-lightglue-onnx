# ALIKED-LightGlue-ONNX

C++ Implementation of Aliked feature extractor and LightGlue feature matcher using ONNX Runtime. ALIKED ONNX model generated from [aliked-tensorrt](https://github.com/ajuric/aliked-tensorrt) and LightGlue models generated from [LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX)

Code also serves as a reference for implementing other ONNX feature extractors in C++, tried to implement SuperPoint but ran into Cudnn errors :p

### Running the feature extractor:
```
./test_aliked <directory containing models> <path to image> <model_type>
```
See include/ExtractorType.h for the corresponding model types, for example, to run aliked-n16rot-top1k:
```
./test_aliked models/ images/test.png 0
```


### Running the featue extractor with LightGlue
```
./lightglue <directory containing models> <path to left image> <path to right image> <extractor_type> <score_threshold>
```
For example, to run lightglue with aliked-n16rot-top1k with 0.9 score threshold:
```
./lightglue models/ images/1.png images/2.png 0 0.9
```
