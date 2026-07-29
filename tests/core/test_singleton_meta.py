"""测试 SingletonMeta 元类"""

import threading
import time

from vibeocr.backend.core import SingletonMeta


class TestSingletonMeta:
    """测试单例元类"""

    def test_singleton_creation(self):
        """测试单例创建"""

        class TestClass(metaclass=SingletonMeta):
            def __init__(self):
                self.value = 0

        # 创建两个实例
        instance1 = TestClass()
        instance2 = TestClass()

        # 应该是同一个实例
        assert instance1 is instance2

    def test_singleton_state_shared(self):
        """测试单例状态共享"""

        class Counter(metaclass=SingletonMeta):
            def __init__(self):
                self.count = 0

        # 清除之前的实例
        SingletonMeta.reset_instance(Counter)

        counter1 = Counter()
        counter1.count = 10

        counter2 = Counter()
        assert counter2.count == 10

        # 清理
        SingletonMeta.reset_instance(Counter)

    def test_thread_safety(self):
        """测试线程安全"""

        class ThreadSafeClass(metaclass=SingletonMeta):
            def __init__(self):
                time.sleep(0.01)  # 模拟初始化延迟
                self.id = id(self)

        # 清除之前的实例
        SingletonMeta.reset_instance(ThreadSafeClass)

        instances = []

        def create_instance():
            instances.append(ThreadSafeClass())

        # 创建多个线程同时创建实例
        threads = [threading.Thread(target=create_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 所有实例应该是同一个
        assert len({id(i) for i in instances}) == 1

        # 清理
        SingletonMeta.reset_instance(ThreadSafeClass)

    def test_reset_instance(self):
        """测试重置实例"""

        class ResettableClass(metaclass=SingletonMeta):
            def __init__(self):
                self.value = 42

        # 清除之前的实例
        SingletonMeta.reset_instance(ResettableClass)

        instance1 = ResettableClass()
        instance1.value = 100

        # 重置实例
        SingletonMeta.reset_instance(ResettableClass)

        instance2 = ResettableClass()
        assert instance2.value == 42  # 重新初始化
        assert instance1 is not instance2

    def test_reset_all(self):
        """测试重置所有实例"""

        class ClassA(metaclass=SingletonMeta):
            pass

        class ClassB(metaclass=SingletonMeta):
            pass

        # 清除之前的实例
        SingletonMeta.reset_instance(ClassA)
        SingletonMeta.reset_instance(ClassB)

        instance_a1 = ClassA()
        instance_b1 = ClassB()

        # 重置所有
        SingletonMeta.reset_all()

        instance_a2 = ClassA()
        instance_b2 = ClassB()

        assert instance_a1 is not instance_a2
        assert instance_b1 is not instance_b2

    def test_reset_with_custom_cleanup(self):
        """测试带自定义清理的重置"""

        class CleanupClass(metaclass=SingletonMeta):
            def __init__(self):
                self.resources = ["resource1", "resource2"]
                self.cleaned = False

            def _reset(self):
                self.resources.clear()
                self.cleaned = True

        # 清除之前的实例
        SingletonMeta.reset_instance(CleanupClass)

        instance = CleanupClass()
        assert len(instance.resources) == 2

        SingletonMeta.reset_instance(CleanupClass)

        # 创建新实例
        new_instance = CleanupClass()
        assert new_instance.cleaned is False  # 新实例，未清理


class TestSingletonMetaDifferentClasses:
    """测试不同类的单例独立"""

    def setup_method(self):
        """每个测试前清理"""
        SingletonMeta.reset_all()

    def teardown_method(self):
        """每个测试后清理"""
        SingletonMeta.reset_all()

    def test_different_classes_independent(self):
        """测试不同类的单例相互独立"""

        class ClassA(metaclass=SingletonMeta):
            def __init__(self):
                self.name = "A"

        class ClassB(metaclass=SingletonMeta):
            def __init__(self):
                self.name = "B"

        a1 = ClassA()
        b1 = ClassB()

        a2 = ClassA()
        b2 = ClassB()

        # 同类实例相同
        assert a1 is a2
        assert b1 is b2

        # 不同类实例不同
        assert a1 is not b1
        assert a1.name == "A"
        assert b1.name == "B"
