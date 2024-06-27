import sys
from PyQt6.QtWidgets import QApplication

from vm_admin_console.model.base_model import BaseModel, Connector
from vm_admin_console.controller.base_controller import BaseController


def main():
    app = QApplication(sys.argv)


    connector = Connector()
    # создаём модель
    model = BaseModel(connector)

    # создаём контроллер и передаём ему ссылку на модель
    controller = BaseController(model)

    app.exec()


if __name__ == '__main__':
    sys.exit(main())
