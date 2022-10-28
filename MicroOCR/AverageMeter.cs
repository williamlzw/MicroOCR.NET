using System;

namespace MicroOCR
{
    public class AverageMeter
    {
        private float _val;
        public float _avg;
        private float _sum;
        private int _count;
        public AverageMeter()
        {
            Reset();
        }

        public void Reset()
        {
            _val = 0;
            _avg = 0;
            _sum = 0;
            _count = 0;
        }

        public void Update(float val, int n = 1)
        {
            _val += val;
            _sum += val * n;
            _count += n;
            _avg = (float)_sum / _count;
        }
    }

}
