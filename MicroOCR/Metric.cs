using TorchSharp;

namespace MicroOCR
{
    public enum ShowNode
    {
        WordCorrect,
        CharCorrect,
        ShowStr
    }
    public interface MetricInterface
    {
        public Dictionary<ShowNode, object> Call(torch.Tensor pred, List<string> labels);
    }
    public class RecMetric: MetricInterface
    {
        private ConverterInterface _converter;
        public RecMetric(ConverterInterface converter) 
        {
            _converter = converter;
        }

        public Dictionary<ShowNode, object> Call(torch.Tensor pred, List<string> labels)
        {
            var predList = _converter.Decode(pred);
            var rawList = _converter.Decode(pred, true);
            Dictionary<ShowNode, object> ret = new Dictionary<ShowNode, object>();
            List<string> showStr = new List<string>();
            int wordCorrect = 0;
            int charCorrect = 0;
            string rawStr, predStr;
            List<float> rawScore = new List<float>();
            List<float> predScore = new List<float>();

            for (int i = 0; i < predList.Count; i++)
            {
                (rawStr, rawScore) = rawList[i];
                if(predList.Count >= rawList.Count)
                {
                    (predStr, predScore) = predList[i];
                }
                else
                {
                    predStr = "_";
                    predScore.Add(0);
                }
                
                var targetStr = labels[i];
                showStr.Add($"{rawStr} -> {predStr} -> {targetStr}");
                if (predStr == targetStr)
                {
                    wordCorrect += 1;
                }
                for(int j = 0; j < predStr.Length;j++)
                {
                    var predChar = predStr[j];
                    char targetChar;
                    if (targetStr.Length >= predStr.Length)
                    {
                        targetChar = targetStr[j];
                    }
                    else
                    {
                        targetChar = '_';
                    }
                    
                    if(predChar == targetChar)
                    {
                        charCorrect += 1;
                    }
                }
            }
            ret.Add(ShowNode.WordCorrect, wordCorrect);
            ret.Add(ShowNode.CharCorrect, charCorrect);
            ret.Add(ShowNode.ShowStr, showStr);
            return ret;
        }
    }
}
