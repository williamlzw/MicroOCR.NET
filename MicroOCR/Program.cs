using TorchSharp;
using static MicroOCR.TrainTool;

namespace MicroOCR
{
    static class Program
    {
        public static void Main()
        {  
            var cfg = BuildCfg();
            TrainModel(cfg);
            Infer(cfg);
        }

        public static TrainConfig BuildCfg()
        {
            TrainConfig cfg = new TrainConfig();
            cfg.TrainRoot = "e:/dataset/ocr10/";
            cfg.TestRoot = "e:/dataset/ocr10/";
            cfg.TrainLabel = "e:/dataset/ocr10/train.txt";
            cfg.TestLabel = "e:/dataset/ocr10/test.txt";
            cfg.VocabularyPath = "D:\\MicroOcr\\MicroOCR\\english.txt";
            cfg.ModelPath = "D:\\MicroOcr\\MicroOCR\\bin\\x64\\Debug\\net6.0\\save_model\\MicroOcr_nh64_depth2_epoch19_wordAcc0.5833_charAcc0.8118.pth";
            cfg.ModelType = "MicroOcr";
            cfg.Nh = 64;
            cfg.Depth = 2;
            cfg.InChannels = 3;
            cfg.Lr = 0.0001;
            cfg.BatchSize = 32;
            cfg.Epochs = 20;
            cfg.DisplayStepInterval = 50;
            cfg.EvalStepInterval = 250;
            cfg.SaveEpochInterval = 1;
            cfg.ShowStrSize = 10;
            cfg.GpuIndex = 0;
            return cfg;
        }
    }
};