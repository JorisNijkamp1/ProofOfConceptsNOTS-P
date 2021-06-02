using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

namespace YoloV3_ObjectDetection.DataStructure
{
    public class YoloV3BitmapData
    {
        [ColumnName("bitmap")]
        [ImageType(416, 416)]
        public Bitmap Image { get; set; }

        [ColumnName("width")]
        public float ImageWidth => Image.Width;

        [ColumnName("height")]
        public float ImageHeight => Image.Height;
    }
}
