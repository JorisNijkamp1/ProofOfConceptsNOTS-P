using Microsoft.ML.Data;

namespace MlOnnxPOC.DataStructures
{
    public class ImageNetPrediction
    {
        [ColumnName("grid")] public float[] PredictedLabels;
    }
}
