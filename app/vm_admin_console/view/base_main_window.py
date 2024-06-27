from vm_admin_console.view.ui_main_window import Ui_MainWindow

from PyQt6 import QtWidgets


class MainWindow(Ui_MainWindow):
    # Переопределяем конструктор класса
    # noinspection PyPep8Naming
    def setupUi(self, MainWindowView):
        # Обязательно нужно вызвать метод супер класса
        super().setupUi(MainWindowView)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    mw = MainWindow()
