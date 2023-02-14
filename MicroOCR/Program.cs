using static MicroOCR.TrainTool;

namespace MicroOCR
{
    static class Program
    {
        public static void Main()
        {  
            var cfg = BuildCfg();
            TrainModel(cfg);
        }

        public static TrainConfig BuildCfg()
        {
            TrainConfig cfg = new TrainConfig();
            cfg.TrainRoot = "D:/dataset/gen/train/";
            cfg.TestRoot = "D:/dataset/gen/test/";
            cfg.TrainLabel = "D:/dataset/gen/train.txt";
            cfg.TestLabel = "D:/dataset/gen/test.txt";
            cfg.VocabularyPath = "D:\\MicroOcr\\MicroOCR\\english.txt";
            //cfg.ModelPath = "D:\\MicroOcr\\MicroOCR\\bin\\Debug\\net6.0\\save_model\\MicroOcr_nh64_depth2_epoch3_wordAcc0.6575.pth";
            cfg.ModelType = "MicroOcr";
            cfg.Nh = 64;
            cfg.Depth = 2;
            cfg.InChannels = 3;
            cfg.Lr = 0.001;
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