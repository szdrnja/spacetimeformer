import logging
import threading

MainLogHandler = logging.StreamHandler()
MainLogHandler.setLevel(logging.INFO)


class Logger(logging.Logger):
    def __init__(self, name="", msg_format=r"{log_tag} {{{thread}}} {level}: {msg}"):
        super().__init__(name)
        self.__log_tag = name
        self.__format = msg_format
        self.setLevel(logging.NOTSET)
        self.addHandler(MainLogHandler)

    def error(self, msg):
        super().error(
            self.__format.format(
                thread=threading.current_thread().ident,
                log_tag=self.__log_tag,
                level="ERROR",
                msg=msg,
            )
        )

    def warning(self, msg):
        super().warning(
            self.__format.format(
                thread=threading.current_thread().ident,
                log_tag=self.__log_tag,
                level="W",
                msg=msg,
            )
        )

    def warn(self, msg):
        super().warn(
            self.__format.format(
                thread=threading.current_thread().ident,
                log_tag=self.__log_tag,
                level="w",
                msg=msg,
            )
        )

    def info(self, msg):
        super().info(
            self.__format.format(
                thread=threading.current_thread().ident,
                log_tag=self.__log_tag,
                level="I",
                msg=msg,
            )
        )

    def debug(self, msg):
        super().debug(
            self.__format.format(
                thread=threading.current_thread().ident,
                log_tag=self.__log_tag,
                level="D",
                msg=msg,
            )
        )
