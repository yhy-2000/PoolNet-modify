import os
from cal_score import cal_score

def test_model(num,model_name='final'):
    command = 'python main.py --mode=test --model=results/run-{}/models/{}.pth --test_fold=results/run-{}-sal-e --sal_mode=e'.format(
        num,model_name, num)
    os.system(command)
    print('testing {}'.format(model_name))
    cal_score(num)
if __name__=='__main__':
    test_model(15,'final')