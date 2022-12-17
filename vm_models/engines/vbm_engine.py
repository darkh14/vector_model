import numpy as np
from typing import Any, Optional
import os
import zipfile
import pickle
import shutil

from sklearn.metrics import mean_squared_error

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam

from ..engines.base_engine import BaseEngine
from ..model_parameters.base_parameters import ModelParameters, FittingParameters

class VbmNeuralNetwork(BaseEngine):
    service_name = 'vbm'
    model_type = 'neural_network'

    def __init__(self, model_parameters: ModelParameters, fitting_parameters: FittingParameters, db_path: str,
                 model_id: str, **kwargs) -> None:

        super().__init__(model_parameters, fitting_parameters, db_path, model_id, **kwargs)

        self._inner_engine: Optional[Sequential] = None
        self._validation_split: float = 0.2

        self._read_from_db()

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int,
            parameters: Optional[dict[str, Any]] = None) -> dict[str, Any]:

        history = self._inner_engine.fit(x, y, epochs=epochs, verbose=2, validation_split=self._validation_split)

        y_pred = self._inner_engine.predict(x)

        metrics = dict()
        metrics['rsme'] = self._calculate_rsme(y, y_pred)
        metrics['mspe'] = self._calculate_mspe(y, y_pred)

        self.metrics = metrics

        self._write_to_db()

        return {'description': 'Fit OK', 'metrics': self.metrics, 'history': history.history}

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._inner_engine.predict(x)

    def _create_inner_engine(self) -> Sequential:

        model = Sequential()

        model.add(Dense(300, activation="relu", input_shape=(self._input_number,), name='dense_1'))
        model.add(Dense(250, activation="relu", name='dense_2'))
        model.add(Dense(100, activation="relu",  name='dense_3'))
        model.add(Dense(30, activation="relu", name='dense_4'))
        model.add(Dense(self._output_number, activation="linear", name='dense_last'))

        self._compile_model(model)

        return model

    def _read_from_db(self) -> None:

        if self._fitting_parameters.is_first_fitting():
            self._inner_engine = self._create_inner_engine()
        else:
            line_from_db = self._db_connector.get_line('engines', {'model_id': self._model_id})

            if line_from_db:
                self._inner_engine = self._get_inner_engine_from_binary_data(line_from_db['inner_engine'])
            else:
                self._inner_engine = self._create_inner_engine()

    def _get_inner_engine_from_binary_data(self, model_data: bytes, use_pickle: bool=False) -> Sequential:

        if use_pickle:
            inner_model = pickle.loads(model_data)
        else:

            if not os.path.isdir('tmp'):
                os.mkdir('tmp')

            with open('tmp/model.zip', 'wb') as f:
                f.write(model_data)

            with zipfile.ZipFile('tmp/model.zip', 'r') as zip_h:
                zip_h.extractall('tmp/model')

            inner_model = load_model('tmp/model')

            os.remove('tmp/model.zip')
            shutil.rmtree('tmp/model')

        return inner_model

    def _write_to_db(self):

        line_to_db = {'model_id': self._model_id, 'inner_engine': self._get_binary_data_from_inner_engine()}
        self._db_connector.set_line('engines', line_to_db, {'model_id': self._model_id})

    def _get_binary_data_from_inner_engine(self, use_pickle: bool = False) -> bytes:

        if use_pickle:
            engine_packed = pickle.dumps(self._inner_engine, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            if not os.path.isdir('tmp'):
                os.mkdir('tmp')

            self._inner_engine.save('tmp/model')

            zipf = zipfile.ZipFile('tmp/model.zip', 'w', zipfile.ZIP_DEFLATED)
            self._zipdir('tmp/model', zipf)
            zipf.close()

            with open('tmp/model.zip', 'rb') as f:
                engine_packed = f.read()

            os.remove('tmp/model.zip')
            shutil.rmtree('tmp/model')

        return engine_packed

    @staticmethod
    def _zipdir(path, zipf):

        for root, dirs, files in os.walk(path):
            c_dir = root
            c_dir = 'tmp/' + c_dir[10:]

            for file in files:
                zipf.write(os.path.join(root, file),
                           os.path.relpath(os.path.join(c_dir, file),
                                           os.path.join(path, '..')))

    @staticmethod
    def _compile_model(model):
        model.compile(optimizer=Adam(learning_rate=0.001), loss='MeanSquaredError',
                      metrics=['RootMeanSquaredError'])

    @staticmethod
    def _calculate_mspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = np.zeros(y_true.shape)
        eps[:] = 0.0001
        y_p = np.c_[abs(y_true), abs(y_pred), eps]
        y_p = np.max(y_p, axis=1).reshape(-1, 1)

        return np.sqrt(np.nanmean(np.square(((y_true - y_pred) / y_p))))

    @staticmethod
    def _calculate_rsme(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))


class VbmLinearModel(VbmNeuralNetwork):
    model_type = 'linear_regression'

    def _create_inner_engine(self) -> Sequential:

        model = Sequential()

        model.add(Dense(self._input_number, activation="relu", input_shape=(self._input_number,), name='dense_1'))
        model.add(Dense(self._output_number, activation="linear", name='dense_last'))

        self._compile_model(model)

        return model

    @staticmethod
    def _compile_model(model):
        model.compile(optimizer=Adam(learning_rate=0.01), loss='MeanSquaredError',
                      metrics=['RootMeanSquaredError'])