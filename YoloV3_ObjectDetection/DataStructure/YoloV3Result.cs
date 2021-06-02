﻿using System;
using System.Collections.Generic;
using System.Text;

namespace YoloV3_ObjectDetection.DataStructure
{
    public class YoloV3Result
    {
        /// <summary>
        /// x1, y1, x2, y2 in page coordinates.
        /// </summary>
        public float[] BBox { get; }

        /// <summary>
        /// The Bbox category.
        /// </summary>
        public string Label { get; }

        /// <summary>
        /// Category's confidence level.
        /// </summary>
        public float Confidence { get; }

        public YoloV3Result(float[] bbox, string label, float confidence)
        {
            BBox = bbox;
            Label = label;
            Confidence = confidence;
        }
    }
}