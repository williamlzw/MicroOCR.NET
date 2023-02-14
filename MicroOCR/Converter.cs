using System.Collections;
using TorchSharp;
using static TorchSharp.torch;

namespace MicroOCR
{
    public interface ConverterInterface
    {
        public Tuple<torch.Tensor, torch.Tensor> Encode(List<string> strs);
        public List<Tuple<string, List<float>>> Decode(torch.Tensor preds, bool raw = false);
    }
    public class CTCLabelConverter : ConverterInterface
    {
        public int _numOfClass;
        private static List<char> _idx2char = new List<char>();
        private static Dictionary<char, int> _char2idx = new Dictionary<char, int>();
        public CTCLabelConverter(string character)
        {
            _numOfClass = character.Length + 2;
            _idx2char.Add('_');
            foreach (var ch in character)
            {
                _idx2char.Add(ch);
            }
            _idx2char.Add(' ');
            for (int i = 0; i < _idx2char.Count; i++)
            {
                _char2idx.Add(_idx2char[i], i);
            }
        }
        private static List<long> str2idx(List<string> strs)
        {
            List<long> list = new List<long>();
            foreach (var str in strs)
            {
                foreach (var ch in str)
                {
                    var idx = _char2idx[ch];
                    list.Add(idx);
                }
            }
            return list;
        }

        public Tuple<torch.Tensor, torch.Tensor> Encode(List<string> strs)
        {
            List<long> targetsLength = new List<long>();
            foreach (var str in strs)
            {
                targetsLength.Add(str.Length);
            }
            var targets = str2idx(strs);
            return new Tuple<torch.Tensor, torch.Tensor>(torch.from_array(targets.ToArray()), torch.from_array(targetsLength.ToArray()));
        }

        public List<Tuple<string, List<float>>> Decode(torch.Tensor preds, bool raw = false)
        {
            preds = torch.nn.functional.softmax(preds, 2);
            var ret = preds.max(2);
            var preds_idx = ret.indexes.detach().to(CPU);
            var preds_score = ret.values.detach().to(CPU);
            List<Tuple<string, List<float>>> retList = new List<Tuple<string, List<float>>>();
            for (int i = 0; i < preds_idx.shape[0]; i++)
            {
                var word = preds_idx[i].to_type(ScalarType.Int32).data<int>().ToArray();
                if (raw)
                {
                    var scoreList = preds_score[i].data<float>().ToArray();
                    var score = scoreList[0];
                    string retStr = "";
                    List<float> retScoreList = new List<float>();
                    for (int j = 0; j < word.Length; j++)
                    {
                        var charIdx = word[j];
                        retStr = retStr + _idx2char[charIdx];
                    }
                    retScoreList.Add(score);
                    var tuple = new Tuple<string, List<float>>(retStr, retScoreList);
                    retList.Add(tuple);
                }
                else
                {
                    var scoreList = preds_score[i].data<float>().ToArray();
                    string retStr = "";
                    List<float> retScoreList = new List<float>();
                    for (int j = 0; j < word.Length; j++)
                    {
                        var charIdx = word[j];
                        if (charIdx != 0 && !(j > 0 && (word[j - 1]) == charIdx))
                        {
                            var bb = charIdx;
                            retStr = retStr + _idx2char[bb];
                            var score = scoreList[j];
                            retScoreList.Add(score);
                        }
                    }
                    var tuple = new Tuple<string, List<float>>(retStr, retScoreList);
                    retList.Add(tuple);
                }
            }
            return retList;
        }
    }
}
