from vm_admin_console.view.base_view import BaseView
from vm_admin_console.model.base_model import BaseModel


class BaseController:
    """
    Класс BaseController представляет реализацию контроллера.
    Согласовывает работу представления с моделью.
    """
    def __init__(self, model: BaseModel):
        """
        Конструктор принимает ссылку на модель.
        Конструктор создаёт и отображает представление.
        """
        self._model = model
        self._view = BaseView(self, self._model)

        self._view.show()
        self._view.set_connection(False)

        self._model.add_observer(self._view)
        self._model.notify_observers()

    def connect(self):

        if self._model.is_connected:
            self._model.disconnect()
            self._view.set_connection(False)
        else:
            self._model.connect()
            if self._model.is_connected:
                self._view.set_status('Connected')
                self._view.set_connection(True)
            else:
                self._view.set_connection(False)
                self._view.set_status(self._model.connector.error_text, color='red')

        
