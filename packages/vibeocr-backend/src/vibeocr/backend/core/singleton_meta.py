"""单例元类

提供线程安全的单例模式实现。
"""

import threading
from typing import Any


class SingletonMeta(type):
    """线程安全的单例元类

    使用方法:
        class MyClass(metaclass=SingletonMeta):
            def __init__(self):
                pass
    """

    _instances: dict[type, Any] = {}
    _meta_lock = threading.Lock()  # 元类级别的锁，避免与子类的 _lock 冲突

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._meta_lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    @classmethod
    def reset_instance(cls, target_class: type) -> None:
        """重置指定类的单例实例

        主要用于测试清理。

        Args:
            target_class: 要重置的类
        """
        with cls._meta_lock:
            if target_class in cls._instances:
                instance = cls._instances[target_class]
                if hasattr(instance, "_reset"):
                    instance._reset()
                del cls._instances[target_class]

    @classmethod
    def reset_all(cls) -> None:
        """重置所有单例实例

        主要用于测试清理。
        """
        with cls._meta_lock:
            for instance in cls._instances.values():
                if hasattr(instance, "_reset"):
                    instance._reset()
            cls._instances.clear()
