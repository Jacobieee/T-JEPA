import sys
import logging
import argparse

from config import Config
from utils import tool_funcs
from task.baseline_simi import TrajSimi
# from model.trajcl import TrajCL
# from model.jepa import JEPA_base


def parse_args():
    # dont set default value here! -- it will incorrectly overwrite the values in config.py.
    # config.py is the correct place for default values.
    parser = argparse.ArgumentParser(description="baseline/train_trajsimi.py")
    parser.add_argument('--dumpfile_uniqueid', type=str, help='see config.py')
    parser.add_argument('--seed', type=int, help='')
    parser.add_argument('--dataset', type=str, help='')

    parser.add_argument('--trajsimi_measure_fn_name', type=str, help='')

    args = parser.parse_args()
    return dict(filter(lambda kv: kv[1] is not None, vars(args).items()))


def main():
    # enc_name = Config.trajsimi_encoder_name
    fn_name = Config.trajsimi_measure_fn_name
    metrics = tool_funcs.Metrics()

    # jepa = JEPA_base()
    # jepa.load_checkpoint()
    # jepa.to(Config.device)

    # print("freezing encoder")
    # # Freeze all parameters first
    # for name, param in jepa.named_parameters():
    #     param.requires_grad = False
    #     print(f"{name} is frozen.")

    # # Assuming 'layers.2.self_attn' and 'layers.2.norm' denote the last attention layers,
    # # we unfreeze them here.
    # # Please adjust 'spatial_attn' and '2' according to your model's structure.
    # for name, param in jepa.named_parameters():
    #     if 'context_encoder.structural_attn.layers.1' in name:
    #         param.requires_grad = True
    #
    # # Verify what has been unfrozen
    # for name, param in jepa.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name} is unfrozen and trainable.")
    #     else:
    #         print(f"{name} is frozen.")

    task = TrajSimi()
    metrics.add(task.train())

    logging.info('[EXPFlag]dataset={},fn={},{}'.format(Config.dataset_prefix, fn_name, str(metrics)))
    return


# nohup python train_trajsimi.py --dataset porto --trajsimi_measure_fn_name hausdorff &> result &
if __name__ == '__main__':
    Config.update(parse_args())

    logging.basicConfig(level=logging.DEBUG if Config.debug else logging.INFO,
                        format="[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
                        handlers=[
                            logging.FileHandler(Config.root_dir + '/exp/log/' + tool_funcs.log_file_name(), mode='w'),
                            logging.StreamHandler()]
                        )

    logging.info('python ' + ' '.join(sys.argv))
    logging.info('=================================')
    logging.info(Config.to_str())
    logging.info('=================================')

    main()
