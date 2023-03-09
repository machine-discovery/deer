import torch


def get_avail_devices():
    devices = [torch.device('cpu')]
    # if torch.cuda.is_available():
    #     devices.append(torch.device('cuda'))
    return devices
