using TorchSharp;

namespace MicroOCR
{
    public static class CollateFn
    {
        public static BatchItem Collate(IEnumerable<TextLineDataSetItem> items, torch.Device device)
        {
            var itemList = items.ToList<TextLineDataSetItem>();
            var transform = torchvision.transforms.ConvertImageDType(torch.ScalarType.Float32);
            List<int> allImageWidth = new List<int>();
            foreach (var item in items)
            {
                allImageWidth.Add((int)((double)32 / item.image.shape[1] * item.image.shape[2]));
            }
            var maxImgWidth = allImageWidth.Max();
            maxImgWidth = (int)Math.Ceiling((double)maxImgWidth / 8) * 8;

            List<torch.Tensor> resizeImgTensor = new List<torch.Tensor>();
            List<string> labels = new List<string>();
            for (int i = 0; i < items.Count(); i++)
            {
                labels.Add(itemList[i].label);
                int newWidth = (int)((double)32 / itemList[i].image.shape[1] * itemList[i].image.shape[2]);
                var resizeOperator = torchvision.transforms.Resize(32, newWidth);
                long[] padding = { 0, (long)maxImgWidth - newWidth, 0, 0 };
                var padOperator = torchvision.transforms.Pad(padding);
                var img = itemList[i].image;
                var imgTensor = padOperator.forward(resizeOperator.forward(transform.forward(img)));
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
