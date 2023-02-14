using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace MicroOCR
{
    public class ConvBNACT : Sequential
    {
        public ConvBNACT(long inChannels, long outChannels, long kernelSize, long stride = 1, long padding = 0, long groups = 1) : base(
            Conv2d(inChannels, outChannels, kernelSize, stride, padding, dilation: 1, groups: groups, bias: false),
            BatchNorm2d(outChannels),
            GELU())
        {
        }
    }

    public class MicroBlock : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> conv1;
        private readonly Module<Tensor, Tensor> conv2;

        public MicroBlock(long nh) : base("MicroBlock")
        {
            conv1 = new ConvBNACT(nh, nh, 1);
            conv2 = new ConvBNACT(nh, nh, 3, 1, 1, nh);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            using var _ = NewDisposeScope();
            var x = conv1.forward(input);
            x = x + conv2.forward(x);
            return x.MoveToOuterDisposeScope();
        }
    }

    public class MicroStage : Sequential
    {
        public MicroStage(int depth, long nh) : base(Enumerable.Range(0, depth).Select(_ => new MicroBlock(nh)).ToArray())
        {
        }
    }

    public class MLP : Sequential
    {
        public MLP(long inputDim, long hiddenDim) : base(
            Linear(inputDim, hiddenDim),
            GELU(),
            Linear(hiddenDim, inputDim),
            Dropout(0.5))
        {
        }
    }

    public class MLPBlock : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> layerNorm1;
        private readonly Module<Tensor, Tensor> layerNorm2;
        private readonly Module<Tensor, Tensor> layerNorm3;
        private readonly Module<Tensor, Tensor> mlp1;
        private readonly Module<Tensor, Tensor> mlp2;
        private readonly Module<Tensor, Tensor> conv;

        public MLPBlock(long inputDim, long hiddenDim) : base("MLPBlock")
        {
            layerNorm1 = LayerNorm(inputDim);
            mlp1 = new MLP(inputDim, hiddenDim);
            layerNorm2 = LayerNorm(inputDim);
            conv = Conv2d(inputDim, inputDim, 3, stride: 1, padding: 1, groups: inputDim, bias: false);
            layerNorm3 = LayerNorm(inputDim);
            mlp2 = new MLP(inputDim, hiddenDim);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            using var _ = NewDisposeScope();
            var x = input.permute(0, 3, 2, 1);
            x = layerNorm1.forward(x);
            x = x + mlp1.forward(x);
            x = layerNorm2.forward(x);
            x = x.permute(0, 3, 2, 1);
            x = conv.forward(x);
            x = x.permute(0, 3, 2, 1);
            x = layerNorm3.forward(x);
            x = x + mlp2.forward(x);
            x = x.permute(0, 3, 2, 1);
            return x.MoveToOuterDisposeScope();
        }
    }

    public class MLPStage : Sequential
    {
        public MLPStage(int depth, long inputDim, long hiddenDim) : base(Enumerable.Range(0, depth).Select(_ => new MLPBlock(inputDim, hiddenDim)).ToArray())
        {
        }
    }

    public class Tokenizer : Sequential
    {
        public Tokenizer(long inputChannels, long hiddenDim, long outDim) : base(
            new ConvBNACT(inputChannels, hiddenDim / 2, 3, 2, 1),
            new ConvBNACT(hiddenDim / 2, hiddenDim / 2, 3, 1, 1),
            new ConvBNACT(hiddenDim / 2, outDim, 3, 1, 1),
            MaxPool2d(3, 2, 1))
        {
        }
    }

    public class MicroMLPNet : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> embed;
        private readonly Module<Tensor, Tensor> microstage;
        private readonly Module<Tensor, Tensor> mlpstages;
        private readonly Module<Tensor, Tensor> flatten;
        private readonly Module<Tensor, Tensor> dropout;
        private readonly Module<Tensor, Tensor> fc;
        public MicroMLPNet(long inputChannels = 3, long nh = 64, int depth = 2, long numClass = 60, long imgHeight = 32) : base("MicroMLPNet")
        {
            embed = new Tokenizer(inputChannels, nh, nh);
            microstage = new MicroStage(depth, nh);
            mlpstages = new MLPStage(depth, nh, nh);
            flatten = Flatten(1, 2);
            dropout = Dropout(0.5);
            long linearIn = nh * imgHeight / 4;
            fc = Linear(linearIn, numClass);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            using var _ = NewDisposeScope();
            var xShape = input.size();
            var x = embed.forward(input);
            x = microstage.forward(x);
            x = mlpstages.forward(x);
            x = flatten.forward(x);
            x = nn.functional.adaptive_avg_pool1d(x, xShape[3] / 4);
            x = x.permute(0, 2, 1);
            x = dropout.forward(x);
            x = fc.forward(x);
            return x.MoveToOuterDisposeScope();
        }
    }
}
