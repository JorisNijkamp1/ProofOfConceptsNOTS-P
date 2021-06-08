# Installatie guide
## Object detection 
**TinyYoloV2**: Het model dat er gedownload moet worden is met het ONNX version 1.3. Dit model moet vervolgens in het mapje **assets/Model** gezet worden. Het model moet **TinyYolo2_model.onnx** heten om te kunnen gebruiken. [Model download link](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny-yolov2)

**TinyYoloV4**: Het model dat er gedownload moet worden is hier te [verkrijgen](https://blog.roboflow.com/train-yolov4-tiny-on-custom-data-lighting-fast-detection/). Je moet een mapje in de assets aanmaken genaamd Model en hier moet het model komen. Het model moet heten **yolov4.onnx**.

**YoloV3**: Het model dat er gedownload moet worden is met het ONNX version 1.5. Dit model moet vervolgens in het mapje **assets/Model** gezet worden. Het model moet **yolov3-10.onnx** heten om te kunnen gebruiken. [Model download link](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov3)

**YoloV4**: Het model dat er gedownload moet worden is met het ONNX version 1.6. Dit model moet vervolgens in het mapje **assets/Model** gezet worden. Het model moet **yolov4.onnx** heten om te kunnen gebruiken. [Model download link](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4)

## Dataset
De dataset die is gebruikt in het [3BrandRecognizer.ipynb](https://github.com/JorisNijkamp1/ProofOfConceptsNOTS-P/blob/master/3BrandRecognizer.ipynb "3BrandRecognizer.ipynb") notebook is: [LogoDet-3K](https://www.kaggle.com/lyly99/logodet3k)

## Google colab transfer learning start up guide
De benodigdheden voor het zelf trainen van een model, via de Google Colab notebook zijn als volgt. Maak folder aan op je eigen Google Drive, in de folder wordt de dataset(afbeeldingen) opgeslagen. Zorg ervoor dat de mappen waar de afbeeldingen in staan, de juiste naam hebben(label van afbeeldingen) Dit is de enigste stap die nodig is voordat je kunt starten met het trainen. Importeer 3BrandRecognizer.ipynb in Google Colab en hierin staat stap voor stap beschreven welke acties er uitgevoerd worden om een model te trainen.
