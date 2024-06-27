from PyQt6.QtWidgets import QMainWindow

# noinspection PyUnresolvedReferences
from vm_admin_console.utility.base_observer import BaseObserver
from vm_admin_console.utility.base_meta import BaseMeta
from vm_admin_console.view.base_main_window import MainWindow
from vm_admin_console.model.base_model import BaseModel

from PyQt6.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget, QLineEdit
from PyQt6.QtCore import QSize, QRegularExpression
from PyQt6.QtGui import QRegularExpressionValidator, QIntValidator


class BaseView(QMainWindow, BaseObserver, metaclass=BaseMeta):
    """
    Класс ...
    """
    def __init__(self, controller, model: BaseModel, parent=None):
        """
        Конструктор принимает ссылки на модель и контроллер.
        """
        super(QMainWindow, self).__init__(parent)
        self._controller = controller
        self._model = model

        # подключаем визуальное представление
        self.ui = MainWindow()
        self.ui.setupUi(self)

        # регистрируем представление в качестве наблюдателя
        self._model.add_observer(self)

        self.ui.ConnectionHostLineEdit.setPlaceholderText('0.0.0.0 or https://example.com')

        port_validator = QIntValidator(0, 65535)
        self.ui.ConnectionPortLineEdit.setValidator(port_validator)
        self.ui.ConnectionPortLineEdit.setPlaceholderText('00000')

        self.ui.ConnectionHostLineEdit.textChanged.connect(self._ConnectionHostLineEditOnChanged)
        self.ui.ConnectionPortLineEdit.textChanged.connect(self._ConnectionPortLineEditOnChanged)

        # связываем событие завершения редактирования с методом контроллера
        self.ui.ConnectButton.clicked.connect(self._controller.connect)

        self.set_status('')

    # noinspection PyPep8Naming
    def _ConnectionPortLineEditOnChanged(self, value):

        if not value or int(value) <= 0 or int(value) > 65535:
            self.ui.ConnectionPortLineEdit.setText(value[:-1])

        self._model.port = int(value)

    # noinspection PyPep8Naming
    def _ConnectionHostLineEditOnChanged(self, value):
        self._model.host = value

    def set_status(self, status: str, color: str = ''):

        self.ui.StatusLabel.setText(status)
        if color:
            self._set_status_color(color)
        else:
            self._set_status_color('white')

    def set_connection(self, is_connected: bool):

        if is_connected:
            self.ui.ConnectionStatusLabel.setText('Connected')
            new_stylesheet = self.ui.ConnectionStatusLabel.styleSheet()
            new_stylesheet = new_stylesheet.replace('color: white', 'color: #00CC00')
            self.ui.ConnectionStatusLabel.setStyleSheet(new_stylesheet)
        else:
            self.ui.ConnectionStatusLabel.setText('Disconnected')
            new_stylesheet = self.ui.ConnectionStatusLabel.styleSheet()
            new_stylesheet = new_stylesheet.replace('color: white', 'color: #FF0000')
            self.ui.ConnectionStatusLabel.setStyleSheet(new_stylesheet)

    def _set_status_color(self, color):

        if color == 'white':
            color = '#FFFFFF'
        elif color == 'red':
            color = '#FF0000'
        elif color == 'blue':
            color = '#0000CC'
        elif color == 'green':
            color = '#00CC00'
        elif color == 'black':
            color = '#000000'

        style_sheet = self.ui.StatusLabel.styleSheet().replace('\n', '')

        style_els = style_sheet.split(';')
        new_style_els = []

        found = False
        for style_el in style_els:
            if style_el.find('color:') != -1:
                new_style_el = f'color: {color}'
                new_style_els.append(new_style_el)
                found = True
            else:
                new_style_els.append(style_el)

        if not found:
            new_style_el = f'color: {color}'
            new_style_els.append(new_style_el)

        new_style_sheet = ';'.join(new_style_els)
        self.ui.StatusLabel.setStyleSheet(new_style_sheet)

    def closeEvent(self, a0):
        super(QMainWindow, self).closeEvent(a0)
        self._model.write_settings()

    def model_changed(self):
        """
        Метод вызывается при изменении модели.
        """
        if 'host' in self._model.properties_changed or 'port' in self._model.properties_changed:
            self.ui.ConnectionHostLineEdit.setText(self._model.host)
            self.ui.ConnectionPortLineEdit.setText(str(self._model.port))



