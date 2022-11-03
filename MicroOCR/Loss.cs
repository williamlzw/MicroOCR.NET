using TorchSharp;
using TorchSharp.Modules;

namespace MicroOCR
{
    public interface LossInterface
    {
        public torch.Tensor Call(torch.Tensor pred, torch.Tensor label, torch.Tensor labelLength);
    }
    public class CtcLoss : LossInterface
    {
        private static CTCLoss _lossFunc;
        private static torch.Device _device;
        public CtcLoss(torch.Device device, int blank = 0, torch.nn.Reduction reduction = torch.nn.Reduction.Sum)
        {
            _lossFunc = torch.nn.CTCLoss(blank, reduction: reduction, zero_infinity:true);
            _device = device;
        }

        public torch.Tensor Call(torch.Tensor pred, torch.Tensor label, torch.Tensor labelLength)
        {
            pred = pred.permute(1, 0, 2);
            var batchSize = pred.size()[1];
            pred = torch.nn.functional.log_softmax(pred, 2);
            var value = pred.size()[0];
            var array = Enumerable.Repeat(value, (int)batchSize).ToArray();
            var predsLength = torch.from_array(array);
            predsLength = predsLength.to(_device);
            return _lossFunc.forward(pred, label, predsLength, labelLength);
        }
    }
}
