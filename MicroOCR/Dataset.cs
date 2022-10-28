using TorchSharp;
using OpenCvSharp;

namespace MicroOCR
{
    public class TextLineDataset : torch.utils.data.Dataset<TextLineDataSetItem>
    {
        private string[] _dataLines;
        private string _dataDir;
        
        public TextLineDataset(string dataDir, string labelFile)
        {
            GetImageInfoList(labelFile);
            _dataDir = dataDir;
        }

        private void GetImageInfoList(string labelFile)
        {
            _dataLines = File.ReadAllLines(labelFile);
        }

        public override long Count
        {
            get
            {
                return _dataLines.Length;
            }
        }

        public override TextLineDataSetItem GetTensor(long index)
        {
            var subStrings = _dataLines[index].Split('\t');
            var imgName = subStrings[0];
            var label = subStrings[1];
            var imgPath = Path.Combine(_dataDir, imgName);
            var image = Cv2.ImRead(imgPath);
            Cv2.CvtColor(image, image, ColorConversionCodes.BGR2RGB);
            return new TextLineDataSetItem
            {
                label = label,
                image = image,
            };
        }
    }

    public class TextLineDataSetItem
    {
        public string label;
        public Mat image;
    }
    public class BatchItem
    {
        public torch.Tensor images;
        public List<string> labels;
    }
}
