import logging

import torch


class Predictor:

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def predict(self, model_network, dataloader, device=None):
        device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.logger.info("Using device {}".format(device))

        arr_prediction_conf = []
        arr_prediction_classes = []
        with torch.no_grad():
            model_network.to(device)
            # switch model to evaluation mode
            model_network.eval()

            soft_max_func = torch.nn.Softmax(dim=-1)
            for i, (batch_x, batch_y) in enumerate(dataloader):
                self.logger.debug("running batch {}".format(i))
                # TODO: CLean this up
                if isinstance(batch_x, list):
                    val_batch_idx = [t.to(device=device) for t in batch_x]
                else:
                    val_batch_idx = batch_x.to(device=device)
                self.logger.debug("predict batch {} {}".format(device, i))

                pred_batch_y = model_network(val_batch_idx)[0]

                self.logger.debug("softmax batch {} {}".format(device, i))

                # Soft max the predictions
                pred_batch_y = soft_max_func(pred_batch_y)
                pred_conf, pred_class = torch.max(pred_batch_y, dim=-1)

                # Move to CPU so we can scale for large datasets
                arr_prediction_conf.append(pred_conf.cpu())
                arr_prediction_classes.append(pred_class.cpu())
                self.logger.debug("Completed cpu {} {}".format(device, i))
            self.logger.debug("In grad {}".format(device))

        predicted_conf = torch.cat(arr_prediction_conf, dim=0)
        predicted_classes = torch.cat(arr_prediction_classes, dim=0)
        self.logger.info("Completed inference {}".format(device))

        return predicted_classes, predicted_conf
