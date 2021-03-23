import os
from cal_score import cal_score

def test_model(num,model_name='final'):
    print('testing {}'.format(model_name))
    command = 'python main.py --mode=test --model=results/run-{}/models/{}.pth --test_fold=results/run-{}-sal-e --sal_mode=e ' \
              '--test_function=hardtanh'.format(
        num,model_name, num)
    os.system(command)
    cal_score(num)
if __name__=='__main__':
    from main import test_all_model
    test_all_model()