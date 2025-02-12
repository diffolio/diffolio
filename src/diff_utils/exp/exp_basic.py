import os
import torch


class Exp_Basic(object):
    def __init__(self, args, env=None):
        self.args = args
        self.env = env
        self.device = self._acquire_device()
        model, _, _ = self._build_model()
        self.model = model.to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None, None, None

    def _acquire_device(self):
        if torch.cuda.is_available():
            if self.args.use_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
                device = torch.device('cuda:{}'.format(self.args.gpu))
                print('Use GPU: cuda:{}'.format(self.args.gpu))
            else:
                device = torch.device('cpu')
                print('Use CPU')
        else:
            device_name = f'mps:0'
            device = torch.device(device_name)
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
