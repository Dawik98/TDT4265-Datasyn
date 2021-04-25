import pathlib
import torch
import os
from ssd.config.defaults import cfg
from train import get_parser

if __name__ == "__main__":
   parser = get_parser()
   args = parser.parse_args()
   cfg.merge_from_file(args.config_file)
   #cfg.merge_from_list(args.opts)
   output_dir = pathlib.Path(cfg.OUTPUT_DIR)
   # Define new output directory to 
   new_output_dir = pathlib.Path(
       output_dir.parent,
       output_dir.stem + "_transfer_learn"
   )

   # Copy checkpoint
   #new_output_dir.mkdir()
   new_checkpoint_path = new_output_dir.joinpath("rdd2020_model.pth")
   previous_checkpoint_path = pathlib.Path(cfg.OUTPUT_DIR, "model_017500.pth")
   assert previous_checkpoint_path.is_file()
   # Only keep the parameters for the model
   new_checkpoint = {
       "model": torch.load(previous_checkpoint_path)["model"]
   }
   torch.save(new_checkpoint, str(new_checkpoint_path))
   del new_checkpoint