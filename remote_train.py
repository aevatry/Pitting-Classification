from src.Config import Config_remote
from src.train import train_net, get_device

if __name__=='__main__':

    device = get_device()
    print(f"Device for current machine is: {device}")

    config_path = 'experiment/LSTMClasdifier1/Configs/exp_1.json'
    train_dir = 'data/Classify_data/train'
    eval_dir = 'data/Classify_data/eval'
    epochs_wanted = 100

    config = Config_remote(config_path, train_dir, eval_dir)

    train_net(config, device, epochs_wanted)