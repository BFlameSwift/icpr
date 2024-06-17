import sys
import os

# 获取当前工作目录
current_dir = os.getcwd()

# 添加项目根目录到 sys.path
project_dir = os.path.abspath(os.path.join(current_dir, './'))
# print(project_dir)
sys.path.append(project_dir)


import os

import typer
from comer.datamodule import CROHMEDatamodule
from comer.lit_comer import LitCoMER
from pytorch_lightning import Trainer, seed_everything

seed_everything(7)


def main(version: str, test_year: str):
    # generate output latex in result.zip
    ckp_folder = os.path.join("lightning_logs", f"version_{version}", "checkpoints")
    fnames = os.listdir(ckp_folder)
    assert len(fnames) == 1
    ckp_path = os.path.join(ckp_folder, fnames[0])
    print(f"Test with fname: {fnames[0]}")

    trainer = Trainer(logger=False, gpus=[4])

    dm = CROHMEDatamodule(test_year=test_year, eval_batch_size=4)

    model = LitCoMER.load_from_checkpoint(ckp_path)

    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    typer.run(main)
