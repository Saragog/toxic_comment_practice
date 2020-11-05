from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from module.model import NaiveBayer, TextCNN, TransformerClassifier

class Trainer(object):
    def __init__(self, config, logger, classes, pretrained_embedding):
        self.config = config
        self.logger = logger
        self.classes = classes
        self.pretrained_embedding = pretrained_embedding
        self._create_model(classes)

    def _create_model(self, classes):
        if self.config['model_name'] == 'naivebayse':
            self.model = NaiveBayer(classes)
        elif self.config['model_name'] == 'textcnn':
            self.model = TextCNN(classes, self.config, self.pretrained_embedding)
        elif self.config['model_name'] == 'transformer':
            self.model = TransformerClassifier(classes, self.config, self.pretrained_embedding)
        else:
            self.logger.warning("Model Type:{} is not support yet".format(self.config['model_name']))

    def metrics(self, predictions, labels):
        accuracy = accuracy_score(labels, predictions)
        cls_report = classification_report(labels, predictions, zero_division=1)
        return accuracy, cls_report

    def fit(self, train_x, train_y):
        self.model.fit(train_x, train_y)
        return self.model

    def validate(self, validate_x, validate_y):
        predictions = self.model.predict(validate_x)
        return self.metrics(predictions, validate_y)

    def fit_and_validate(self, train_x, train_y, validate_x, validate_y):
        predictions, history = self.model.fit_and_validate(train_x, train_y, validate_x, validate_y)
        accuracy, cls_report = self.metrics(predictions, validate_y)
        return self.model, accuracy, cls_report, history
