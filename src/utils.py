import time

from tqdm import tqdm

from nn.loss_functions.mse_loss import mse_loss

def progress_bar(iterable, text='Epoch progress', end=''):
    """Мониториг выполнения эпохи

    ---------
    Параметры
    ---------
    iterable
        Что-то по чему можно итерироваться

    text: str (default='Epoch progress')
        Текст, выводящийся в начале

    end : str (default='')
        Что вывести в конце выполнения
    """
    max_num = len(iterable)
    iterable = iter(iterable)

    start_time = time.time()
    cur_time = 0
    approx_time = 0

    print('\r', end='')

    it = 0
    while it < max_num:
        it += 1
        print(f"{text}: [", end='')

        progress = int((it - 1) / max_num * 50)
        print('=' * progress, end='')
        if progress != 50:
            print('>', end='')
            print(' ' * (50 - progress - 1), end='')
        print('] ', end='')

        print(f'{it - 1}/{max_num}', end='')
        print(' ', end='')

        print(f'{cur_time}s>{approx_time}s', end='')

        yield next(iterable)

        print('\r', end='')
        print(' ' * (60 + len(text) + len(str(max_num)) + len(str(it))
                     + len(str(cur_time)) + len(str(approx_time))),
              end='')
        print('\r', end='')

        cur_time = time.time() - start_time

        approx_time = int(cur_time / it * (max_num - it))
        cur_time = int(cur_time)
        print(end, end='')


def gradient_check(x, y, neural_net):
    h = 1e-4
    eps = 0.5
    neural_net.train()
    y_pred = neural_net.forward(x)

    l = mse_loss(y_pred, y)
    l.backward()
    diff_sum = 0
#     neural_net.eval()
    for params in neural_net.parameters():
        for i in tqdm(range(params.params.shape[0])):
            if len(params.params.shape) == 2:
                for j in range(params.params.shape[1]):
                    #print('HELO')
                    params.params[i, j] += h
                    y_plus = neural_net.forward(x)
                    l_plus = mse_loss(y_plus, y).loss
                    #print(l_plus)
                    params.params[i, j] -= 2 * h
                    y_minus = neural_net.forward(x)
                    l_minus = mse_loss(y_minus, y).loss
                    #print(l_minus)
                    params.params[i, j] += h
                    grad_mechanic = (l_plus - l_minus) / (2 * h)
                    
                    diff = abs(grad_mechanic - params.grads[i, j])
           
                    diff_sum += diff
                    #print(diff)
                    if diff > eps:
                        print(grad_mechanic)
                        print(params.grads[0, i])
                        print("BAD DIFF = ", diff)
                        return -1
            else:
                params.params[i] += h
                y_plus = neural_net.forward(x)
                l_plus = mse_loss(y_plus, y).loss
                params.params[i] -= 2 * h
                y_minus = neural_net.forward(x)
                l_minus = mse_loss(y_minus, y).loss
                params.params[i] += h
                grad_mechanic = (l_plus - l_minus) / (2 * h)

                diff = abs(grad_mechanic - params.grads[0, i])
                diff_sum += diff
                
                if diff > eps:
                    
                    print("BAD DIFF = ", diff)
                    return -1
       
    print("Good Gradient")
    print('SUM DIFF = ', diff_sum)
    return diff_sum

