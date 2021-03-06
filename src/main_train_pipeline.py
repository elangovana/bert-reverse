import argparse
import json
import logging
import os
import sys

from dataset_builder import DatasetBuilder
from train_builder import TrainBuilder

logger = logging.getLogger(__name__)


class TrainPipeline:

    def _persist_config(self, additional_args, model_dir):
        # Persist params
        output_config = os.path.join(model_dir, "training_config_parameters.json")
        logger.info(f"Writing config to {output_config}")
        with open(output_config, "w") as f:
            json.dump(additional_args, f)

    def run_train(self, train_dir, val_dir, model_dir, checkpoint_dir, additional_args):
        self._persist_config(additional_args, model_dir)

        # Builder
        dataset_builder = DatasetBuilder(val_data=val_dir, dataset_factory_name=additional_args["datasetfactory"],
                                         tokenisor_factory_name=additional_args["modelfactory"], train_data=train_dir,
                                         num_workers=additional_args["numworkers"], batch_size=additional_args["batch"],
                                         addition_args_dict=additional_args)

        train_builder = TrainBuilder(model_factory_name=additional_args["modelfactory"],
                                     scorers=dataset_builder.get_scorers(),
                                     num_classes=dataset_builder.num_classes(),
                                     checkpoint_dir=checkpoint_dir, epochs=additional_args["epochs"],
                                     grad_accumulation_steps=additional_args.get("gradientaccumulationsteps", 1),
                                     learning_rate=additional_args.get("learningrate", 0.0001),
                                     use_loss_eval=additional_args.get("uselosseval", 0),
                                     early_stopping_patience=additional_args["earlystoppingpatience"],
                                     model_dir=model_dir,
                                     addition_args_dict=additional_args)

        trainer = train_builder.get_trainer()

        # Run training
        result = trainer.run_train(train_iter=dataset_builder.get_train_dataloader(),
                                   validation_iter=dataset_builder.get_val_dataloader(),
                                   model_network=train_builder.get_network(),
                                   loss_function=train_builder.get_loss_function(),
                                   optimizer=train_builder.get_optimiser(),
                                   pos_label=dataset_builder.positive_label_index()
                                   )
        return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasetfactory",
                        help="The dataset type e.g. datasets.aimed_dataset_factory.AimedDatasetFactory",
                        required=True)

    parser.add_argument("--traindir",
                        help="The input train  dir. If kfoldrootdir is set, then pass the ",
                        default=os.environ.get("SM_CHANNEL_TRAIN", "."))

    parser.add_argument("--valdir",
                        help="The input val dir. If kfoldrootdir is set then this directory is ignored, else mandatory",
                        default=os.environ.get("SM_CHANNEL_VAL", None))

    parser.add_argument("--testdir",
                        help="The input test dir", default=os.environ.get("SM_CHANNEL_TEST", None))

    parser.add_argument("--modelfactory",
                        help="The model factory type e.g. models.bert_model_factory.BertModelFactory",
                        default="models.bert_model_factory.BertModelFactory")

    parser.add_argument("--pretrained_model_dir",
                        help="The pretrained model dir",
                        default=os.environ.get("SM_CHANNEL_PRETRAINED_MODEL", None))

    parser.add_argument("--outdir", help="The output dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "."))
    parser.add_argument("--modeldir", help="The output dir", default=os.environ.get("SM_MODEL_DIR", "."))
    parser.add_argument("--checkpointdir", help="The checkpoint dir", default=None)
    parser.add_argument("--checkpointfreq", help="The checkpoint frequency, number of epochs", default=1)

    parser.add_argument("--gradientaccumulationsteps", help="The number of gradient accumulation steps", type=int,
                        default=1)
    parser.add_argument("--learningrate", help="The learningrate", type=float, default=0.0001)

    parser.add_argument("--batch", help="The batchsize", type=int, default=32)
    parser.add_argument("--epochs", help="The number of epochs", type=int, default=10)
    parser.add_argument("--earlystoppingpatience", help="The number of patience epochs epochs", type=int, default=10)
    parser.add_argument("--numworkers", help="The number of workers to use", type=int, default=None)

    parser.add_argument("--uselosseval", help="Whether the best model should be optimised for lowest loss", default=0,
                        choices={0, 1}, type=int)

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args, additional = parser.parse_known_args()

    # Convert additional args into dict
    additional_dict = {}
    for i in range(0, len(additional), 2):
        additional_dict[additional[i].lstrip("--")] = additional[i + 1]
    additional_dict["pretrained_model"] = args.pretrained_model_dir
    additional_dict["vocab_file"] = os.path.join(args.pretrained_model_dir, "vocab.txt")

    return args, additional_dict


def main_run():
    args, additional_args = parse_args()
    print(args.__dict__)
    print(additional_args)

    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Runs
    TrainPipeline().run_train(args.traindir,
                              args.valdir,
                              model_dir=args.modeldir,
                              checkpoint_dir=args.checkpointdir,
                              additional_args={**vars(args), **additional_args}
                              )


if __name__ == '__main__':
    main_run()
