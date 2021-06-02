using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using Microsoft.ML;
using YoloV3_ObjectDetection.DataStructure;
using static Microsoft.ML.Transforms.Image.ImageResizingEstimator;

namespace YoloV3_ObjectDetection
{
    class Program
    {
        // MAPPEN STRUCTUUR
        const string modelPath =
            @"C:\Users\joris\source\repos\MlOnnxPOC\POC-YoloV3\YoloV3_ObjectDetection\Assets\Model\yolov3-10.onnx";

        const string imageFolder =
            @"C:\Users\joris\source\repos\MlOnnxPOC\POC-YoloV3\YoloV3_ObjectDetection\Assets\Images\";

        const string imageOutputFolder =
            @"C:\Users\joris\source\repos\MlOnnxPOC\POC-YoloV3\YoloV3_ObjectDetection\Assets\Output\";

        // LABELS
        static readonly string[] classesNames = new string[]
        {
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
            "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"
        };

        static void Main(string[] args)
        {
            Directory.CreateDirectory(imageOutputFolder);
            MLContext mlContext = new MLContext();


            // pipeline met input, output en dimensies. Deze is interresant
            var pipeline = mlContext.Transforms.ResizeImages(
                    inputColumnName: "bitmap",
                    outputColumnName: "input_1",
                    imageWidth: 416,
                    imageHeight: 416,
                    resizing: ResizingKind.IsoPad
                )
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input_1", scaleImage: 1f / 255f))
                .Append(mlContext.Transforms.Concatenate("image_shape", "height", "width"))
                .Append(mlContext.Transforms.ApplyOnnxModel(
                    shapeDictionary: new Dictionary<string, int[]>() {{"input_1", new[] {1, 3, 416, 416}}},
                    inputColumnNames: new[]
                    {
                        "input_1",
                        "image_shape"
                    },
                    outputColumnNames: new[]
                    {
                        "yolonms_layer_1/ExpandDims_1:0",
                        "yolonms_layer_1/ExpandDims_3:0",
                        "yolonms_layer_1/concat_2:0"
                    },
                    modelFile: modelPath));

            // Lege lijst toevoegen aan het model
            var model = pipeline.Fit(mlContext.Data.LoadFromEnumerable(new List<YoloV3BitmapData>()));

            // Prediction engine aanmaken en toevoegen aan het model
            var predictionEngine = mlContext.Model.CreatePredictionEngine<YoloV3BitmapData, YoloV3Prediction>(model);

            // Image inladen waar je op wilt detecteren.
            string imageName = "imageFestivalJoris.jpg";

            using (var bitmap = new Bitmap(Image.FromFile(Path.Combine(imageFolder, imageName))))
            {
                // predict
                var predict = predictionEngine.Predict(new YoloV3BitmapData() {Image = bitmap});
                var results = GetResults(predict, classesNames);

                // Predictions tekenen op de fotos
                using (var g = Graphics.FromImage(bitmap))
                {
                    foreach (var result in results)
                    {
                        var y1 = result.BBox[0];
                        var x1 = result.BBox[1];
                        var y2 = result.BBox[2];
                        var x2 = result.BBox[3];

                        g.DrawRectangle(Pens.Red, x1, y1, x2 - x1, y2 - y1);
                        using (var brushes = new SolidBrush(Color.FromArgb(50, Color.Red)))
                        {
                            g.FillRectangle(brushes, x1, y1, x2 - x1, y2 - y1);
                        }

                        g.DrawString(result.Label + " " + result.Confidence.ToString("0.00"),
                            new Font("Arial", 12), Brushes.Blue, new PointF(x1, y1));
                    }

                    // Afbeeldingnaam veranderd 
                    bitmap.Save(Path.Combine(imageOutputFolder,
                        Path.ChangeExtension(imageName, "_processed" + Path.GetExtension(imageName))));
                }
            }
        }

        public static IReadOnlyList<YoloV3Result> GetResults(YoloV3Prediction prediction, string[] categories)
        {
            // Checks of prediction wel aanwezig en correct is.
            if (prediction.Concat == null || prediction.Concat.Length == 0) return new List<YoloV3Result>();
            if (prediction.Boxes.Length != YoloV3Prediction.YoloV3BboxPredictionCount * 4)
                throw new ArgumentException();
            if (prediction.Scores.Length != YoloV3Prediction.YoloV3BboxPredictionCount * categories.Length)
                throw new ArgumentException();

            // Return object aanmaken.
            var results = new List<YoloV3Result>();

            var resulstCount = prediction.Concat.Length / 3;
            for (var c = 0; c < resulstCount; c++)
            {
                var res = prediction.Concat.Skip(c * 3).Take(3).ToArray();

                var classIndex = res[1];
                var boxIndex = res[2];

                var label = categories[classIndex];
                var bbox = new float[]
                {
                    prediction.Boxes[boxIndex * 4],
                    prediction.Boxes[boxIndex * 4 + 1],
                    prediction.Boxes[boxIndex * 4 + 2],
                    prediction.Boxes[boxIndex * 4 + 3],
                };
                var score = prediction.Scores[boxIndex + classIndex * YoloV3Prediction.YoloV3BboxPredictionCount];

                results.Add(new YoloV3Result(bbox, label, score));
            }

            return results;
        }
    }
}