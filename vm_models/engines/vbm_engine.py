"""
    VBM (Vector budget model)
    Module for ML engine classes.
    Classes:
        VbmNeuralNetwork -  3 layer direct distribution NN
        VbmLinearModel - linear regression based on sklearn.linear_model
        PolynomialModel - 2 degree model based on sklearn.linear_model with polynomial transformation
"""

import numpy as np
from typing import Any, Optional, ClassVar
import os
import zipfile
import pickle
import shutil

from sklearn.linear_model import LinearRegression

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam

from ..engines.base_engine import BaseEngine


class VbmNeuralNetwork(BaseEngine):
    """ 3 layer direct distribution NN. Powered by tensorflow.keras """
    service_name: ClassVar[str] = 'vbm'
    model_type: ClassVar[str] = 'neural_network'

    def __init__(self, model_id: str, input_number: int, output_number: int, new_engine: bool = False,
                 **kwargs) -> None:
        """
        Defines inner engine = None, _validation_split and reads from db if necessary
        :param model_id: id of model object
        :param input_number: number of inputs
        :param output_number: umber of outputs (labels)
        :param new_engine: if is True  - not need to read from db
        :param kwargs: additional parameters
        """
        super().__init__(model_id, input_number, output_number, new_engine, **kwargs)

        self._inner_engine: Optional[Sequential] = None
        self._validation_split: float = 0.0

        self._read_from_db()

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int,
            parameters: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """
        For fitting ML engine
        :param x:  input data
        :param y: output data (labels)
        :param epochs: nuber of epochs of fitting
        :param parameters: additional parameters
        :return: history of fitting
        """
        history = self._inner_engine.fit(x, y, epochs=epochs, verbose=2, validation_split=self._validation_split)

        self._write_to_db()

        return {'description': 'Fit OK', 'history': history.history}

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        For predicting using inner engine
        :param x: input data for predicting
        :return: predicted output data
        """
        return self._inner_engine.predict(x)

    def create_inner_engine(self) -> Sequential:
        """
        For creating new inner engine object and compile it
        :return: new inner engine object (keras Sequential)
        """
        model = Sequential()

        model.add(Dense(300, activation="relu", input_shape=(self._input_number,), name='dense_1'))
        model.add(Dense(250, activation="relu", name='dense_2'))
        model.add(Dense(100, activation="relu",  name='dense_3'))
        model.add(Dense(30, activation="relu", name='dense_4'))
        model.add(Dense(self._output_number, activation="linear", name='dense_last'))

        self.compile_engine(model)

        return model

    def _read_from_db(self) -> None:
        """
        For read inner engine from db and save it to self._inner_engine
        """
        if self._new_engine or not self._model_id:
            self._inner_engine = self.create_inner_engine()
        else:
            line_from_db = self._db_connector.get_line('engines', {'model_id': self._model_id})

            if line_from_db:
                self._inner_engine = self._get_inner_engine_from_binary_data(line_from_db['inner_engine'])
            else:
                self._inner_engine = self.create_inner_engine()

    # noinspection PyMethodMayBeStatic
    def _get_inner_engine_from_binary_data(self, model_data: bytes, use_pickle: bool = False) -> Sequential:
        """
        Gets inner engine object from its binary data
        :param model_data: binary engine data
        :param use_pickle: need to use pickle module to load engine
        :return: inner engine object
        """
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

    def _write_to_db(self) -> None:
        """
        For writing inner engine object to db
        """
        if self._model_id:
            line_to_db = {'model_id': self._model_id, 'inner_engine': self._get_binary_data_from_inner_engine()}
            self._db_connector.set_line('engines', line_to_db, {'model_id': self._model_id})

    def _get_binary_data_from_inner_engine(self, use_pickle: bool = False) -> bytes:
        """
        Gets binary data of inner engine object from inner object itself
        :param use_pickle: use pickle module if True
        :return: binary data of inner engine object
        """
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
    def _zipdir(path, zipf) -> None:
        """
        Makes and writes zip file from dir
        :param path: path to dir, from which wwe make zipfile
        :param zipf: zip file object
        """
        for root, dirs, files in os.walk(path):
            c_dir = root
            c_dir = 'tmp/' + c_dir[10:]

            for file in files:
                zipf.write(os.path.join(root, file),
                           os.path.relpath(os.path.join(c_dir, file),
                                           os.path.join(path, '..')))

    @staticmethod
    def compile_engine(engine):
        """
        Method to compile engine
        :param engine: ML engine object
        """
        engine.compile(optimizer=Adam(learning_rate=0.001), loss='MeanSquaredError',
                      metrics=['RootMeanSquaredError'])

    @property
    def inner_engine(self) -> Sequential:
        """
        Property for self._inner_engine
        :return: value of self._inner_engine
        """
        return self._inner_engine


class VbmLinearModel(BaseEngine):
    """ Linear regression based on 1 layer NN inherited by VbmNeuralNetwork """
    service_name: ClassVar[str] = 'vbm'
    model_type: ClassVar[str] = 'linear_regression'

    def __init__(self, model_id: str, input_number: int, output_number: int, new_engine: bool = False,
                 **kwargs) -> None:
        """
        Defines inner engine = None, _validation_split and reads from db if necessary
        :param model_id: id of model object
        :param input_number: number of inputs
        :param output_number: umber of outputs (labels)
        :param new_engine: if is True  - not need to read from db
        :param kwargs: additional parameters
        """
        super().__init__(model_id, input_number, output_number, new_engine, **kwargs)

        self._inner_engine: Optional[LinearRegression] = None

        self._read_from_db()

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int,
            parameters: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """
        For fitting ML engine
        :param x:  input data
        :param y: output data (labels)
        :param epochs: nuber of epochs of fitting
        :param parameters: additional parameters
        :return: history of fitting
        """
        self._inner_engine.fit(x, y)

        self._write_to_db()

        return {'description': 'Fit OK', 'history': []}

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        For predicting using inner engine
        :param x: input data for predicting
        :return: predicted output data
        """
        return self._inner_engine.predict(x)

    # noinspection PyMethodMayBeStatic
    def create_inner_engine(self) -> LinearRegression:
        """
        For creating new inner engine object and compile it
        :return: new inner engine object (LinearRegression)
        """
        model = LinearRegression()

        return model

    def _read_from_db(self) -> None:
        """
        For read inner engine from db and save it to self._inner_engine
        """
        if self._new_engine or not self._model_id:
            self._inner_engine = self.create_inner_engine()
        else:
            line_from_db = self._db_connector.get_line('engines', {'model_id': self._model_id})

            if line_from_db:
                self._inner_engine = self._get_inner_engine_from_binary_data(line_from_db['inner_engine'])
            else:
                self._inner_engine = self.create_inner_engine()

    # noinspection PyMethodMayBeStatic
    def _get_inner_engine_from_binary_data(self, model_data: bytes) -> LinearRegression:
        """
        Gets inner engine object from its binary data
        :param model_data: binary engine data
        :return: inner engine object
        """

        inner_model = pickle.loads(model_data)

        return inner_model

    def _write_to_db(self) -> None:
        """
        For writing inner engine object to db
        """
        if self._model_id:
            line_to_db = {'model_id': self._model_id, 'inner_engine': self._get_binary_data_from_inner_engine()}
            self._db_connector.set_line('engines', line_to_db, {'model_id': self._model_id})

    def _get_binary_data_from_inner_engine(self) -> bytes:
        """
        Gets binary data of inner engine object from inner object itself
        :return: binary data of inner engine object
        """

        engine_packed = pickle.dumps(self._inner_engine, protocol=pickle.HIGHEST_PROTOCOL)

        return engine_packed

    @property
    def inner_engine(self) -> LinearRegression:
        """
        Property for self._inner_engine
        :return: value of self._inner_engine
        """
        return self._inner_engine
        