using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp;
using static MicroOCR.CollateFn;
using static TorchSharp.torch.optim.lr_scheduler.impl;
using TorchSharp.Modules;

namespace MicroOCR
{
    public static class TrainTool
    {
        public static void TrainModel(TrainConfig cfg)
        {
            string description = "cpu";
            if (torch.cuda.is_available())
            {
                description = $"cuda:{cfg.GpuIndex}";
            }

            var device = torch.device(description);
            var lines = File.ReadAllLines(cfg.VocabularyPath);
            string character = "";
            foreach (var line in lines)
            {
                character += line.TrimEnd();
            }
            var trainLoader = BuildDataloader(cfg.TrainRoot, cfg.TrainLabel, cfg.BatchSize, true, device);
            var testLoader = BuildDataloader(cfg.TestRoot, cfg.TestLabel, cfg.BatchSize, false, device);
            var converter = BuildConverter(character);
            var lossFun = BuildLoss(device);
            var average = BuildAverageMeter();
            var metric = BuildMetric(converter);
            var model = BuildModel(cfg.InChannels, cfg.Nh, cfg.Depth, converter._numOfClass);
            if (cfg.ModelPath != null && cfg.ModelPath != "")
            {
                LoadModel(cfg.ModelPath, model);
            }
            model = model.to(device);
            var optimizer = BuildOptimizer(model, cfg.Lr);
            var scheduler = BuildScheduler(optimizer);

            float lastWordAcc = 0;
            float bestWordAcc = 0;
            var writer = torch.utils.tensorboard.SummaryWriter();

            foreach (var epoch in Enumerable.Range(0, cfg.Epochs))
            {
                model.train();
                int charCorrects = 0, wordCorrects = 0, allWord = 1, allChar = 1;
                var since = Environment.TickCount;
                int batchIdx = 0;
                foreach (var batchItem in trainLoader)
                {
                    using var _ = torch.NewDisposeScope();
                    var (targets, targetsLength) = converter.Encode(batchItem.labels);
                    var images = batchItem.images;
                    targets = targets.to(device);
                    targetsLength = targetsLength.to(device);
                    var pred = model.forward(images);
                    var loss = lossFun.Call(pred, targets, targetsLength);
                    optimizer.zero_grad();
                    loss.backward();
                    optimizer.step();
                    average.Update(loss.ToSingle());
                    var costTime = (Environment.TickCount - since) / 1000;
                    if ((batchIdx + 1) % cfg.DisplayStepInterval == 0 || (batchIdx + 1) == trainLoader.Count)
                    {
                        var showDict = metric.Call(pred, batchItem.labels);
                        charCorrects += (int)showDict[ShowNode.CharCorrect];
                        wordCorrects += (int)showDict[ShowNode.WordCorrect];
                        var sum = torch.sum(targetsLength);
                        allChar += (int)sum.ToInt64();
                        allWord += (int)images.size(0);
                        lastWordAcc = (float)wordCorrects / allWord;
                        string logStr = $"Train:[epoch {epoch + 1}/{cfg.Epochs}][step {batchIdx + 1}/{trainLoader.Count} ({(float)100 * (batchIdx + 1) / trainLoader.Count:F0}%] lr:{optimizer.ParamGroups.First().LearningRate:F5} Loss:{average._avg:F4} Word Acc:{(float)wordCorrects / allWord:F4} Char Acc:{(float)charCorrects / allChar:F4} Cost time:{costTime}s Estimated time:{costTime * trainLoader.Count / (batchIdx + 1) - costTime}s";
                        Console.WriteLine(logStr);
                    }
                    if ((batchIdx + 1) % cfg.EvalStepInterval == 0)
                    {
                        var (wordAcc, charAcc, _) = TestModel(model, device, testLoader, converter, metric, lossFun, cfg.ShowStrSize);
                        if (wordAcc > bestWordAcc)
                        {
                            bestWordAcc = wordAcc;
                            //SaveModel(model, cfg[CfgNode.ModelType].ToString(), (epoch + 1).ToString(), (int)cfg[CfgNode.Nh], (int)cfg[CfgNode.Depth], wordAcc, charAcc);
                        }
                    }
                    batchIdx += 1;
                    batchItem.images.Dispose();
                }
                if ((epoch + 1) % cfg.SaveEpochInterval == 0)
                {
                    var (wordAcc, charAcc, valLoss) = TestModel(model, device, testLoader, converter, metric, lossFun, cfg.ShowStrSize);

                    writer.add_scalar("Validation Word Acc", wordAcc, epoch + 1);
                    writer.add_scalar("Validation Loss", valLoss, epoch + 1);

                    string epochStr = (epoch + 1).ToString();
                    if (wordAcc > bestWordAcc)
                    {
                        bestWordAcc = wordAcc;
                        epochStr = "best";
                    }
                    SaveModel(model, cfg.ModelType, epochStr, cfg.Nh, cfg.Depth, wordAcc, charAcc);
                }
                average.Reset();
                scheduler.step(lastWordAcc, epoch);
            }
        }

        public static (float, float, float) TestModel(Module<Tensor, Tensor> model, torch.Device device, DataLoader<TextLineDataSetItem, BatchItem> testLoader, ConverterInterface converter, MetricInterface metric, LossInterface lossFun, int showStrSize)
        {
            model.eval();
            var lss = 0.0f;
            List<string> showStr = new List<string>();
            int charCorrects = 0, wordCorrects = 0, allWord = 0, allChar = 0;
            using (torch.no_grad())
            {
                using var _ = torch.NewDisposeScope();
                var since = Environment.TickCount;
                int batchIdx = 0;
                foreach (var batchItem in testLoader)
                {
                    var (targets, targetsLength) = converter.Encode(batchItem.labels);
                    var images = batchItem.images;
                    targets = targets.to(device);
                    targetsLength = targetsLength.to(device);
                    var pred = model.forward(images);
                    var loss = lossFun.Call(pred, targets, targetsLength);
                    var showDict = metric.Call(pred, batchItem.labels);
                    var listStr = showDict[ShowNode.ShowStr] as List<string>;
                    showStr.AddRange(listStr);
                    charCorrects += (int)showDict[ShowNode.CharCorrect];
                    wordCorrects += (int)showDict[ShowNode.WordCorrect];
                    var sum = torch.sum(targetsLength);
                    allChar += (int)sum.ToInt64();
                    allWord += (int)images.size(0);
                    var costTime = (Environment.TickCount - since) / 1000;
                    if ((batchIdx + 1) == testLoader.Count)
                    {
                        string logStr = $"Eval:[step {batchIdx + 1}/{testLoader.Count} ({(float)100 * (batchIdx + 1) / testLoader.Count:F0})%] Loss:{loss.ToSingle():F4} Word Acc:{(float)wordCorrects / allWord:F4} Char Acc:{(float)charCorrects / allChar:F4} Cost time:{costTime}s";
                        lss = loss.ToSingle();
                        Console.WriteLine(logStr);
                    }
                    batchIdx += 1;
                    batchItem.images.Dispose();
                }
            }

            foreach (var index in showStr.Take(showStrSize))
            {
                Console.WriteLine(index);
            }
            model.train();
            float wordAcc = (float)wordCorrects / allWord;
            float charAcc = (float)charCorrects / allChar;
            return (wordAcc, charAcc, lss);
        }

        public static AverageMeter BuildAverageMeter()
        {
            return new AverageMeter();
        }

        public static CTCLabelConverter BuildConverter(string character)
        {
            return new CTCLabelConverter(character);
        }

        public static RecMetric BuildMetric(ConverterInterface conveter)
        {
            return new RecMetric(conveter);
        }

        public static LossInterface BuildLoss(torch.Device device)
        {
            return new CtcLoss(device);
        }

        public static optim.Optimizer BuildOptimizer(Module<Tensor, Tensor> model, double lr = 0.0001)
        {
            //return optim.RMSProp(model.parameters(), lr);
            return optim.Adam(model.parameters(), lr, 0.5, 0.999, weight_decay: 0.0001);
        }

        public static torch.utils.data.Dataset<TextLineDataSetItem> BuildDataset(string dataDir, string labelFile)
        {
            return new TextLineDataset(dataDir, labelFile);
        }

        public static DataLoader<TextLineDataSetItem, BatchItem> BuildDataloader(string dataDir, string labelFile, int batchSize, bool doShuffle, torch.Device device)
        {
            var dataset = BuildDataset(dataDir, labelFile);
            var dataloader = new DataLoader<TextLineDataSetItem, BatchItem>(dataset, batchSize, Collate, doShuffle, device);
            
            return dataloader;
        }

        public static ReduceLROnPlateau BuildScheduler(optim.Optimizer optimizer)
        {
            //var milestones = new List<int>();
            //milestones.Add(30);
            //var scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma: 0.1);
            //var scheduler = optim.lr_scheduler.StepLR(optimizer, 15, gamma: 0.90);
            var scheduler = (ReduceLROnPlateau)optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience: 3, threshold: 0.01, threshold_mode: "abs");
            return scheduler;
        }

        public static void SaveModel(Module<Tensor, Tensor> model, string modelType, string epoch, int nh, int depth, float wordAcc, float charAcc)
        {
            string dirPath = Environment.CurrentDirectory + "/save_model";
            if (!File.Exists(dirPath))
            {
                Directory.CreateDirectory(dirPath);
            }
            string savePath = $"{dirPath}/{modelType}_nh{nh}_depth{depth}_epoch{epoch}_wordAcc{wordAcc:F4}_charAcc{charAcc:F4}.pth";
            model.save(savePath);
        }

        public static void LoadModel(string modelPath, Module<Tensor, Tensor> model)
        {
            model.load(modelPath);
        }

        public static Module<Tensor, Tensor> BuildModel(int inChannels, int nh, int depth, int nclass)
        {
            var net = new MicroMLPNet(inChannels, nh, depth, nclass, 32);
            return net;
        }

        public static void Infer(TrainConfig cfg)
        {
            string description = "cpu";
            if (torch.cuda.is_available())
            {
                description = $"cuda:{cfg.GpuIndex}";
            }

            var device = torch.device(description);
            var lines = File.ReadAllLines(cfg.VocabularyPath);
            string character = "";
            foreach (var line in lines)
            {
                character += line.TrimEnd();
            }
            var converter = new CTCLabelConverter(character);
            var model = new MicroMLPNet(cfg.InChannels, cfg.Nh, cfg.Depth, converter._numOfClass, 32);
            if (cfg.ModelPath != null && cfg.ModelPath != "")
            {
                model.load(cfg.ModelPath);
            }
           
            model = model.to(device);
            model.eval();
            torchvision.io.DefaultImager = new torchvision.io.SkiaImager();
            var image = torchvision.io.read_image("E:\\dataset\\ocr10\\train\\0.jpg");
            var transformOperator = torchvision.transforms.ConvertImageDtype(ScalarType.Float32);
            var resizeOperator = torchvision.transforms.Resize(32, 140);
            image = transformOperator.call(image);
            image = resizeOperator.call(image).unsqueeze(0);
            image = image.to(device);
            var pred = model.call(image);
            var predList = converter.Decode(pred);
            var (predStr, predScore) = predList[0];
            Console.WriteLine(predStr);
        }

        public class TrainConfig
        {
            public string TrainRoot;
            public string TestRoot;
            public string TrainLabel;
            public string TestLabel;
            public string VocabularyPath;
            public string ModelPath;
            public string ModelType;
            public int InChannels;
            public double Lr;
            public int BatchSize;
            public int Epochs;
            public int DisplayStepInterval;
            public int EvalStepInterval;
            public int SaveEpochInterval;
            public int ShowStrSize;
            public int GpuIndex;
            public int Nh;
            public int Depth;
        }
    }
}
