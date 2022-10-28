using OpenCvSharp;
using System.Runtime.InteropServices;
using TorchSharp;

namespace MicroOCR
{
    public static class CollateFn
    {
        private static Mat ResizeWithSpecificHeight(int inputHeight, Mat img)
        {
            double resizeRatio = (double)inputHeight / img.Size().Height;
            Mat outImg = new Mat();
            Cv2.Resize(img, outImg, new OpenCvSharp.Size(0, 0), resizeRatio, resizeRatio, InterpolationFlags.Linear);
            return outImg;
        }

        private static byte[] PadImageWidth(Mat img, int targetWidth, int padValue = 0)
        {
            byte[] retData = new byte[0];
            var size = img.Size();
            if (targetWidth > size.Width)
            {
                Mat returnImg = new Mat();
                Cv2.CopyMakeBorder(img, returnImg, 0, 0, 0, targetWidth - size.Width, BorderTypes.Constant, padValue);
                var bytes = new byte[returnImg.Total() * 3];
                Marshal.Copy(returnImg.Data, bytes, 0, bytes.Length);       
                retData = bytes;
            }
            else
            {
                var bytes = new byte[img.Total() * 3];
                Marshal.Copy(img.Data, bytes, 0, bytes.Length);
                retData = bytes;
            }
            return retData;
        }

        public static BatchItem Collate(IEnumerable<TextLineDataSetItem> items, torch.Device device)
        {
            List<Mat> allSameHeightImage = new List<Mat>();
            var itemList = items.ToList<TextLineDataSetItem>();
            var transform = torchvision.transforms.ConvertImageDType(torch.ScalarType.Float32);
            foreach (var item in items)
            {
                allSameHeightImage.Add(ResizeWithSpecificHeight(32, item.image));
            }
            var maxImgWidth = allSameHeightImage.Max(mat => mat.Size().Width);
            maxImgWidth = (int)Math.Ceiling((double)maxImgWidth / 8) * 8;
            List<torch.Tensor> resizeImgTensor = new List<torch.Tensor>();
            List<string> labels = new List<string>();
            for (int i = 0; i < items.Count(); i++)
            {
                labels.Add(itemList[i].label);
                var img = PadImageWidth(allSameHeightImage[i], maxImgWidth);
                var imgTensor = transform.forward(img).reshape(3, 32, maxImgWidth);
                //var imgTensor = torch.tensor(img, torch.uint8).div(255).reshape(3, 32, maxImgWidth).to(torch.float32);
                resizeImgTensor.Add(imgTensor);
            }
            var images = torch.stack(resizeImgTensor).to(device);
            return new BatchItem
            {
                labels = labels,
                images = images
            };
        }
    }
}
